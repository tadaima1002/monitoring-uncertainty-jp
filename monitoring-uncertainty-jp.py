# Copyright 2026 tadaima1002
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See: http://www.apache.org/licenses/LICENSE-2.0

# feature_extraction_pipeline.py
# requirements: torch torchvision numpy scikit-learn scipy tqdm pandas matplotlib
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize
from scipy.stats import spearmanr, ks_2samp, ttest_ind
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32          # GTX1050Ti 4GB 向け
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 200
OUTPUT_CSV = 's_d_results_features.csv'
RANDOM_STATE = 0

# Repro
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Data
# -------------------------
transform = T.Compose([
    T.Resize(224),                # ImageNet pretrained に合わせる
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
])

train_full = torchvision.datasets.CIFAR10(root='D:/datasets/cifar10', train=True, download=True, transform=transform)
test_full  = torchvision.datasets.CIFAR10(root='D:/datasets/cifar10', train=False, download=True, transform=transform)

def sample_per_class(dataset, n_per_class, seed=SEED):
    labels = np.array(dataset.targets)
    idxs = []
    rng = np.random.RandomState(seed)
    for c in range(NUM_CLASSES):
        c_idx = np.where(labels == c)[0]
        chosen = rng.choice(c_idx, size=min(n_per_class, len(c_idx)), replace=False)
        idxs.extend(chosen.tolist())
    return Subset(dataset, idxs)

train_subset = sample_per_class(train_full, SAMPLES_PER_CLASS)
test_subset  = sample_per_class(test_full, SAMPLES_PER_CLASS)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print("train_subset len:", len(train_subset))
print("test_subset len:", len(test_subset))
print("Device", DEVICE)

# -------------------------
# Model as feature extractor
# -------------------------
model = torchvision.models.resnet18(pretrained=True)
# replace final fc with identity to get features
model.fc = nn.Identity()
model = model.to(DEVICE)
model.eval()

def extract_features(model, loader):
    feats = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader):
            imgs = imgs.to(DEVICE)
            out = model(imgs)            # (N, feat_dim)
            feats.append(out.cpu().numpy())
            labels.append(labs.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels

train_X, train_y = extract_features(model, train_loader)
test_X, test_y   = extract_features(model, test_loader)
print("feature dim", train_X.shape)

# -------------------------
# Train lightweight classifier
# -------------------------
clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
clf.fit(train_X, train_y)
# -------------------------
# ここから追加コード（clf.fit の直後に貼る）
# -------------------------
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_auc_score

# 安全な AUC 計算ラッパー
def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)

# ブートストラップで AUC の 95% CI を計算
def bootstrap_auc(y_true, score, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], score[idx]))
    if len(aucs) == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.percentile(aucs, [2.5, 50, 97.5])

# 1) cross_val_predict で train 側の確率を得る（5-fold）
print("Computing cross-validated train probabilities (5-fold)...")
probs_cv = cross_val_predict(clf, train_X, train_y, cv=5, method='predict_proba', n_jobs=1)

# 2) CalibratedClassifierCV で校正（train 全体で校正）
print("Fitting calibrated classifier (sigmoid)...")
cal = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
cal.fit(train_X, train_y)
test_probs_cal = cal.predict_proba(test_X)
train_probs_cal = cal.predict_proba(train_X)

# 3) 使う確率と対応する予測ラベルを明示的に作る
#    - train: CV 確率からの予測ラベル
#    - test: 校正済確率からの予測ラベル
train_probs_used = probs_cv
train_preds_used = np.argmax(train_probs_used, axis=1)

test_probs_used = test_probs_cal
test_preds_used = cal.predict(test_X)   # 校正器の predict を使う（predict_proba と整合）

# 4) s/d を再計算（エントロピーとマージン）
eps = 1e-12
def compute_ent_margin(probs):
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    return ent, margin

train_ent_cv, train_margin_cv = compute_ent_margin(train_probs_used)
test_ent_cal, test_margin_cal = compute_ent_margin(test_probs_used)

# 5) AUC とブートストラップ CI（test）
test_wrong = (test_preds_used != test_y).astype(int)
ci_ent = bootstrap_auc(test_wrong, test_ent_cal, n_boot=1000, seed=0)
ci_margin = bootstrap_auc(test_wrong, -test_margin_cal, n_boot=1000, seed=0)
print("Test AUC-s 95% CI:", ci_ent)
print("Test AUC-d 95% CI:", ci_margin)

# 6) クラス別 AUC（test）
print("\n--- Per-class AUC (Test Set) ---")
y_bin = label_binarize(test_y, classes=range(NUM_CLASSES))
for c in range(NUM_CLASSES):
    try:
        auc_c = roc_auc_score(y_bin[:, c], test_probs_used[:, c])
    except Exception:
        auc_c = np.nan
    print(f"Class {c} AUC: {auc_c:.4f}")

# 7) train 側の CV を使った安全な AUC 表示
train_wrong_cv = (train_preds_used != train_y).astype(int)
train_auc_s_cv = safe_auc(train_wrong_cv, train_ent_cv)
print("\nTrain AUC s (CV) :", train_auc_s_cv)

# -------------------------
# ここまで追加コード
# -------------------------

cal = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
cal.fit(train_X, train_y)
test_probs_cal = cal.predict_proba(test_X)

# 例: 5-fold CV で確率を得る（scikit-learn）
from sklearn.model_selection import cross_val_predict
probs_cv = cross_val_predict(clf, train_X, train_y, cv=5, method='predict_proba', n_jobs=1)

# Predict probabilities
train_probs = clf.predict_proba(train_X)
test_probs  = clf.predict_proba(test_X)
train_preds = clf.predict(train_X)
test_preds  = clf.predict(test_X)

# -------------------------
# s and d computation (NumPy)
# -------------------------
def compute_s_d(probs: np.ndarray):
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)   # s proxy
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_p[:, 0] - sorted_p[:, 1]             # d proxy
    s_z = (ent - ent.mean()) / (ent.std() + eps)
    d_z = (margin - margin.mean()) / (margin.std() + eps)
    return ent, margin, s_z, d_z

train_ent, train_margin, train_s, train_d = compute_s_d(train_probs)
test_ent, test_margin, test_s, test_d = compute_s_d(test_probs)

# -------------------------
# Metrics
# -------------------------
def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(mask):
            acc = np.mean(predictions[mask] == labels[mask])
            conf = np.mean(confidences[mask])
            ece += np.abs(acc - conf) * mask.mean()
    return ece

def bootstrap_auc(y_true, score, n_boot=1000):
    rng = np.random.RandomState(0)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], score[idx]))
    return np.percentile(aucs, [2.5, 50, 97.5])


train_acc = accuracy_score(train_y, train_preds)
test_acc  = accuracy_score(test_y, test_preds)
train_ece = expected_calibration_error(train_probs, train_y, n_bins=10)
test_ece  = expected_calibration_error(test_probs, test_y, n_bins=10)

train_wrong = (train_preds != train_y).astype(int)
test_wrong  = (test_preds != test_y).astype(int)

def auc_for_detection(wrong_flags, score):
    try:
        return roc_auc_score(wrong_flags, score)
    except ValueError:
        return np.nan

train_auc_s = auc_for_detection(train_wrong, train_ent)
test_auc_s  = auc_for_detection(test_wrong, test_ent)
train_auc_d = auc_for_detection(train_wrong, -train_margin)
test_auc_d  = auc_for_detection(test_wrong, -test_margin)

train_rho_s, train_p_s = spearmanr(train_ent, train_wrong)
test_rho_s, test_p_s   = spearmanr(test_ent, test_wrong)

ks_ent_stat, ks_ent_p = ks_2samp(train_ent, test_ent)
t_ent_stat, t_ent_p = ttest_ind(train_ent, test_ent)

# -------------------------
# Save results
# -------------------------
df = pd.DataFrame({
    'split': ['train']*len(train_ent) + ['test']*len(test_ent),
    'label': np.concatenate([train_y, test_y]),
    'pred': np.concatenate([train_preds, test_preds]),
    'prob_max': np.concatenate([train_probs.max(axis=1), test_probs.max(axis=1)]),
    'entropy': np.concatenate([train_ent, test_ent]),
    'margin': np.concatenate([train_margin, test_margin]),
    's_z': np.concatenate([train_s, test_s]),
    'd_z': np.concatenate([train_d, test_d]),
    'is_wrong': np.concatenate([train_wrong, test_wrong])
})
df.to_csv(OUTPUT_CSV, index=False)

# -------------------------
# Print summary
# -------------------------
# ↓ ここに挿入
# per-class AUC (one-vs-rest)
print("\n--- Per-class AUC (Test Set) ---")
y_bin = label_binarize(test_y, classes=range(NUM_CLASSES))
for c in range(NUM_CLASSES):
    auc_c = roc_auc_score(y_bin[:, c], test_probs[:, c])
    print(f"Class {c} AUC: {auc_c:.4f}")

# Bootstrap Confidence Intervals for Uncertainty Detection
ci_s = bootstrap_auc(test_wrong, test_ent)
print(f"\nTest AUC-s 95% CI: [{ci_s[0]:.4f}, {ci_s[1]:.4f}, {ci_s[2]:.4f}]")

print("Train Acc", train_acc, "Test Acc", test_acc)
print("Train ECE", train_ece, "Test ECE", test_ece)
print("Train AUC s(entropy)", train_auc_s, "Test AUC s(entropy)", test_auc_s)
print("Train AUC d(margin)", train_auc_d, "Test AUC d(margin)", test_auc_d)
print("Spearman train s vs wrong", train_rho_s, train_p_s)
print("Spearman test s vs wrong", test_rho_s, test_p_s)
print("KS test entropy train vs test p", ks_ent_p)
print("t-test entropy train vs test p", t_ent_p)
print("Results saved to", OUTPUT_CSV)

# -------------------------
# Simple visualizations
# -------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(test_ent[test_wrong==0], bins=30, alpha=0.6, label='correct')
plt.hist(test_ent[test_wrong==1], bins=30, alpha=0.6, label='wrong')
plt.title('Entropy distribution test split')
plt.legend()
plt.subplot(1,2,2)
plt.hist(test_margin[test_wrong==0], bins=30, alpha=0.6, label='correct')
plt.hist(test_margin[test_wrong==1], bins=30, alpha=0.6, label='wrong')
plt.title('Margin distribution test split')
plt.legend()
plt.tight_layout()
plt.savefig('s_d_histograms.png')

print("Saved histogram s_d_histograms.png")

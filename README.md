# monitoring-uncertainty-jp
AI-based monitoring and uncertainty detection for autonomous control
# Project Title
AIベースのモニタリングと不確実性検出の最小再現実装。

## 再現手順
1. Python 3.8+ を用意
2. pip install -r requirements.txt
3. python monitoring-uncertainty-jp.py

### 環境構築（推奨手順）

1. 仮想環境を作成して有効化（例: venv）
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

#incase using cpu-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#incase using cuda11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#incase using cuda12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#### インストール確認
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

##### 実行
python monitoring-uncertainty-jp.py

### 補足（短く）
- `requirements.txt` に `torch` とだけ書いておくと環境依存で失敗することがあるため、上のように別コマンドで入れるのが安全です。  
- GPU を使う場合は事前に NVIDIA ドライバと対応する CUDA ランタイムが正しく入っていることを確認してください。  


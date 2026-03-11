# Fake News Robustness Project / 假新聞魯棒性專案

## English
RoBERTa-based robustness project for fake news detection.

Implemented strategy scope (locked):
- `vanilla`
- `lexical_mhc_lite` (main mHC-lite line)
- `style_invariance`
- `sentiment_invariance`

### Dataset
- LIAR (binary mapping)
  - fake = {pants-fire, false, barely-true}
  - real = {half-true, mostly-true, true}

### Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Train
```bash
python train.py --strategy vanilla --device cpu
python train.py --strategy lexical_mhc_lite --device cpu
python train.py --strategy style_invariance --device cpu
python train.py --strategy sentiment_invariance --device cpu
```

### Full Evaluation Matrix
```bash
python evaluate_matrix.py --epochs 2 --device cpu --out-dir outputs
```

Outputs:
- `outputs/results_matrix.csv`
- `outputs/f1_pivot.csv`

### Simple manifold-constrained hyper-connections (mHC-lite structure)
A lightweight structural variant is enabled by default in `train.py`:
- Fuse `[CLS]` and masked mean-pooled sequence features via a gated hyper-connection
- Add manifold bottleneck projection/reconstruction regularization
- Training loss includes optional `manifold_weight * manifold_loss`

Controls:
```bash
python train.py --strategy lexical_mhc_lite --use-mhc-lite-structure --manifold-weight 0.05
python train.py --strategy lexical_mhc_lite --no-mhc-lite-structure
```

---

## 繁體中文
本專案為 RoBERTa 假新聞偵測魯棒性研究。

目前策略範圍（已鎖定）：
- `vanilla`
- `lexical_mhc_lite`（mHC-lite 主線）
- `style_invariance`
- `sentiment_invariance`

### 資料集
- LIAR（二元化）
  - fake = {pants-fire, false, barely-true}
  - real = {half-true, mostly-true, true}

### 快速開始
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 訓練
```bash
python train.py --strategy vanilla --device cpu
python train.py --strategy lexical_mhc_lite --device cpu
python train.py --strategy style_invariance --device cpu
python train.py --strategy sentiment_invariance --device cpu
```

### 完整評估矩陣
```bash
python evaluate_matrix.py --epochs 2 --device cpu --out-dir outputs
```

輸出：
- `outputs/results_matrix.csv`
- `outputs/f1_pivot.csv`

### 簡易 manifold-constrained hyper-connections（mHC-lite 結構）
`train.py` 預設啟用輕量結構版：
- 用 gated hyper-connection 融合 `[CLS]` 與 masked mean-pooled 特徵
- 加入 manifold bottleneck 投影/重建正則
- 總 loss 可加入 `manifold_weight * manifold_loss`

控制參數：
```bash
python train.py --strategy lexical_mhc_lite --use-mhc-lite-structure --manifold-weight 0.05
python train.py --strategy lexical_mhc_lite --no-mhc-lite-structure
```

# Fake News Robustness Project (RoBERTa + mHC-lite/SheepDog-style/AdSent-style)

## What is included now
- Experiment blueprint: `experiment_plan.md`
- Training scaffold: `train.py`
- Core modules:
  - `src/modeling.py`
  - `src/training_strategies.py`
  - `src/evaluate.py`
  - `src/datasets.py` (placeholder loader)

## Dataset choice
I selected **LIAR** as the first implementation dataset because it has standard train/validation/test splits and is easy to reproduce.

Binary mapping used in code:
- fake = {pants-fire, false, barely-true}
- real = {half-true, mostly-true, true}

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# baseline
python train.py --strategy vanilla --device cpu

# lexical robustness (mHC-lite)
python train.py --strategy lexical_mhc_lite --device cpu

# style invariance
python train.py --strategy style_invariance --device cpu

# sentiment invariance
python train.py --strategy sentiment_invariance --device cpu

# run full 4x4 evaluation matrix (train+eval all strategies)
python evaluate_matrix.py --epochs 2 --device cpu --out-dir outputs
```

Outputs:
- `outputs/results_matrix.csv`
- `outputs/f1_pivot.csv`

## Implemented strategies
- `vanilla`: CE on clean text
- `lexical_mhc_lite`: CE(clean) + consistency KL(clean || synonym-perturbed)
- `style_invariance`: CE(clean) + consistency KL(clean || style-reframed)
- `sentiment_invariance`: CE(clean) + consistency KL(clean || sentiment-shifted)

## Scope lock (do not exceed unless explicitly requested)
Future iterations should stay within this strategy scope only:
- vanilla
- lexical_mhc_lite (mHC-lite main line)
- style_invariance
- sentiment_invariance

## Notes
- style/sentiment augmenters are lightweight local proxies (no external LLM API yet)
- mHC-related code paths should keep rich comments for readability and reviewability
- if needed, next step is replacing style/sentiment proxies with LLM-generated rewrites for closer reproduction of SheepDog/AdSent

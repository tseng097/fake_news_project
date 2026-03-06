# Fake News Robustness Experiment Plan (RoBERTa + 3 Robust Strategies)

## Models
1. RoBERTa-base (vanilla)
2. RoBERTa + Lexical robustness (mHC-lite)
3. RoBERTa + Style robustness (SheepDog-style invariance)
4. RoBERTa + Sentiment robustness (AdSent-style invariance)

## Evaluation Matrix
Each model is evaluated on:
- Clean test
- Lexical/synonym attack
- Style attack
- Sentiment attack

Metrics:
- Accuracy
- Macro-F1
- Attack Success Rate (ASR)
- Delta-F1 (clean - attacked)

## Suggested Fixed Hyperparameters
- Backbone: roberta-base
- Max length: 256
- Batch size: 16
- LR: 2e-5
- Epochs: 4
- Warmup ratio: 0.1
- Weight decay: 0.01
- Early stopping patience: 2
- Seed: 42

## Flow (One-page text diagram)
1) Load dataset and split (train/val/test)
2) Build base RoBERTa classifier
3) Select strategy: vanilla / lexical_mhc_lite / style_invariance / sentiment_invariance
4) Train with corresponding objective
5) Generate attacked test sets (lexical/style/sentiment)
6) Evaluate all models on all test variants
7) Produce comparison table and discussion points

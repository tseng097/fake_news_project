from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def attack_success_rate(clean_pred, attacked_pred, y_true):
    """ASR: originally correct -> attacked wrong proportion."""
    clean_correct = np.array(clean_pred) == np.array(y_true)
    attacked_wrong = np.array(attacked_pred) != np.array(y_true)
    denom = max(clean_correct.sum(), 1)
    return float((clean_correct & attacked_wrong).sum() / denom)

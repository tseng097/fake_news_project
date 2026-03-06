from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict, load_dataset


@dataclass
class DataConfig:
    dataset_name: str = "liar"
    text_col: str = "statement"
    label_col: str = "label"


# LIAR labels (common ordering in HF mirror):
# 0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true
# Binary mapping for this project:
# fake = {0,1,2}, real = {3,4,5}
FAKE_LABELS = {0, 1, 2}
REAL_LABELS = {3, 4, 5}


def _to_binary_label(label: int) -> int:
    if label in FAKE_LABELS:
        return 1  # fake
    if label in REAL_LABELS:
        return 0  # real
    # fallback (shouldn't happen)
    return int(label > 2)


def load_data(cfg: DataConfig) -> DatasetDict:
    ds = load_dataset(cfg.dataset_name)

    def _map(batch):
        labels = batch[cfg.label_col]
        out = [_to_binary_label(int(x)) for x in labels]
        return {"labels": out, "text": batch[cfg.text_col]}

    keep_cols = [cfg.text_col, cfg.label_col]
    ds = ds.map(_map, batched=True)

    # Keep only unified columns expected by training loop
    for split in ds.keys():
        remove_cols = [c for c in ds[split].column_names if c not in {"text", "labels"}]
        ds[split] = ds[split].remove_columns(remove_cols)

    return ds

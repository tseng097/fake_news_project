from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.augment import lexical_synonym_perturb, sentiment_shift_simple, style_reframe_simple
from src.datasets import DataConfig, load_data
from src.evaluate import attack_success_rate, compute_metrics
from src.modeling import ModelConfig, RobertaFakeNewsClassifier
from src.training_strategies import StrategyConfig, total_loss

STRATEGIES = ["vanilla", "lexical_mhc_lite", "style_invariance", "sentiment_invariance"]
ATTACKS = ["clean", "lexical", "style", "sentiment"]


def augment_text(text: str, mode: str) -> str:
    if mode == "lexical":
        return lexical_synonym_perturb(text)
    if mode == "style":
        return style_reframe_simple(text)
    if mode == "sentiment":
        return sentiment_shift_simple(text)
    return text


def build_collate(tokenizer, max_len: int, train_strategy: str = "vanilla"):
    def collate(examples):
        texts = [e["text"] for e in examples]
        labels = torch.tensor([int(e["labels"]) for e in examples], dtype=torch.long)

        clean = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

        if train_strategy == "vanilla":
            aug = None
        else:
            aug_mode = {
                "lexical_mhc_lite": "lexical",
                "style_invariance": "style",
                "sentiment_invariance": "sentiment",
            }[train_strategy]
            aug_texts = [augment_text(t, aug_mode) for t in texts]
            aug = tokenizer(aug_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

        return clean, aug, labels

    return collate


def build_eval_collate(tokenizer, max_len: int, attack_mode: str):
    def collate(examples):
        texts = [augment_text(e["text"], attack_mode) for e in examples]
        labels = torch.tensor([int(e["labels"]) for e in examples], dtype=torch.long)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc, labels

    return collate


def predict(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for enc, labels in dl:
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(labels.numpy().tolist())
    model.train()
    return y_true, y_pred


def train_one(strategy: str, backbone: str, epochs: int, lr: float, batch_size: int, max_len: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    ds = load_data(DataConfig())
    train_split = ds["train"]

    train_dl = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate(tokenizer, max_len=max_len, train_strategy=strategy),
    )

    model = RobertaFakeNewsClassifier(ModelConfig(backbone=backbone)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scfg = StrategyConfig(name=strategy)

    for _ in tqdm(range(epochs), desc=f"train:{strategy}"):
        for clean, aug, labels in train_dl:
            labels = labels.to(device)
            clean_ids = clean["input_ids"].to(device)
            clean_mask = clean["attention_mask"].to(device)
            clean_logits = model(clean_ids, clean_mask)

            if aug is not None:
                aug_logits = model(aug["input_ids"].to(device), aug["attention_mask"].to(device))
            else:
                aug_logits = None
            loss = total_loss(scfg, clean_logits, labels, aug_logits=aug_logits)
            optim.zero_grad()
            loss.backward()
            optim.step()

    return model, tokenizer, ds["test"]


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for strategy in STRATEGIES:
        model, tokenizer, test_split = train_one(
            strategy=strategy,
            backbone=args.backbone,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=args.device,
        )

        clean_dl = DataLoader(test_split, batch_size=args.batch_size, shuffle=False, collate_fn=build_eval_collate(tokenizer, args.max_len, "clean"))
        y_true, y_clean = predict(model, clean_dl, args.device)
        clean_metrics = compute_metrics(y_true, y_clean)

        for attack in ATTACKS:
            attack_dl = DataLoader(test_split, batch_size=args.batch_size, shuffle=False, collate_fn=build_eval_collate(tokenizer, args.max_len, attack))
            y_true_a, y_att = predict(model, attack_dl, args.device)
            m = compute_metrics(y_true_a, y_att)

            row = {
                "strategy": strategy,
                "attack": attack,
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "delta_f1_vs_clean": clean_metrics["f1_macro"] - m["f1_macro"],
            }
            if attack == "clean":
                row["ASR"] = 0.0
            else:
                row["ASR"] = attack_success_rate(y_clean, y_att, y_true)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_matrix.csv", index=False)

    pivot = df.pivot(index="strategy", columns="attack", values="f1_macro")
    pivot.to_csv(out_dir / "f1_pivot.csv")

    print(df)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="roberta-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="outputs")
    args = p.parse_args()
    main(args)

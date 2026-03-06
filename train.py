from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.augment import lexical_synonym_perturb, sentiment_shift_simple, style_reframe_simple
from src.datasets import DataConfig, load_data
from src.modeling import ModelConfig, RobertaFakeNewsClassifier
from src.training_strategies import StrategyConfig, total_loss


@dataclass
class TrainConfig:
    backbone: str = "roberta-base"
    strategy: str = "vanilla"  # vanilla|lexical_mhc_lite|style_invariance|sentiment_invariance
    epochs: int = 4
    lr: float = 2e-5
    batch_size: int = 16
    max_len: int = 256
    device: str = "cpu"


def _augment_text(text: str, strategy: str) -> str:
    if strategy == "lexical_mhc_lite":
        return lexical_synonym_perturb(text)
    if strategy == "style_invariance":
        return style_reframe_simple(text)
    if strategy == "sentiment_invariance":
        return sentiment_shift_simple(text)
    return text


def build_collate_fn(tokenizer, max_len: int, strategy: str):
    def collate(examples):
        texts = [e["text"] for e in examples]
        labels = torch.tensor([int(e["labels"]) for e in examples], dtype=torch.long)

        clean = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

        if strategy == "vanilla":
            aug = None
        else:
            aug_texts = [_augment_text(t, strategy) for t in texts]
            aug = tokenizer(aug_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

        return clean, aug, labels

    return collate


def evaluate(model, dl, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for clean, _, labels in dl:
            input_ids = clean["input_ids"].to(device)
            attention_mask = clean["attention_mask"].to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / max(total, 1)


def train(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    ds = load_data(DataConfig())

    train_split = ds["train"]
    val_split = ds.get("validation", ds["test"])

    collate = build_collate_fn(tokenizer, cfg.max_len, cfg.strategy)
    train_dl = DataLoader(train_split, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_split, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    model = RobertaFakeNewsClassifier(ModelConfig(backbone=cfg.backbone)).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    strategy = StrategyConfig(name=cfg.strategy)

    for epoch in tqdm(range(cfg.epochs), desc=f"Training ({cfg.strategy})"):
        running_loss = 0.0
        for clean, aug, labels in train_dl:
            input_ids = clean["input_ids"].to(cfg.device)
            attention_mask = clean["attention_mask"].to(cfg.device)
            labels = labels.to(cfg.device)

            clean_logits = model(input_ids, attention_mask)

            if aug is not None:
                aug_ids = aug["input_ids"].to(cfg.device)
                aug_mask = aug["attention_mask"].to(cfg.device)
                aug_logits = model(aug_ids, aug_mask)
            else:
                aug_logits = None

            loss = total_loss(strategy, clean_logits, labels, aug_logits=aug_logits)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_dl, cfg.device)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | loss={running_loss / max(len(train_dl), 1):.4f} | val_acc={val_acc:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="roberta-base")
    p.add_argument(
        "--strategy",
        type=str,
        default="vanilla",
        choices=["vanilla", "lexical_mhc_lite", "style_invariance", "sentiment_invariance"],
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    train(
        TrainConfig(
            backbone=args.backbone,
            strategy=args.strategy,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=args.device,
        )
    )

"""Strong custom CNN for room-occupancy classification.

Design choices that fix the prior pipeline:
- Train at native resolution padded to 48x64 (no destructive resize from 45x51 to 128).
- Treat the 3 PNG channels as 3 colormap encodings of the underlying scalar field.
- Safe augmentations only: horizontal flip (Doppler symmetry), light Gaussian noise,
  random row/column dropout (delay/Doppler bin masking). NO rotation/translation/scale.
- 5-fold StratifiedKFold x N seeds, with TTA (orig + horizontal flip).
- AdamW + CosineAnnealingLR, label smoothing, MixUp on logits (optional).
- Saves OOF probs and averaged test probs per (model, seed) for downstream stacking.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 4
PAD_H, PAD_W = 48, 64  # native 45x51 padded
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_image_tensor(path: Path) -> np.ndarray:
    """Load PNG as float32 RGB array in [0,1] padded to PAD_H x PAD_W."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    assert h <= PAD_H and w <= PAD_W
    pad_top = (PAD_H - h) // 2
    pad_bottom = PAD_H - h - pad_top
    pad_left = (PAD_W - w) // 2
    pad_right = PAD_W - w - pad_left
    arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
    return arr.transpose(2, 0, 1)  # CHW


def preload_all(image_dir: Path, ids: Iterable[int]) -> np.ndarray:
    arrays = []
    for sample_id in ids:
        arrays.append(load_image_tensor(image_dir / f"img_{sample_id + 1}.png"))
    return np.stack(arrays).astype(np.float32)


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None, augment: bool):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return self.X.shape[0]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Horizontal flip with p=0.5 (Doppler symmetry)
        if random.random() < 0.5:
            img = img[:, :, ::-1].copy()
        # Light Gaussian noise
        if random.random() < 0.5:
            img = img + np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
        # Random row/col dropout (mask one delay/Doppler bin) — light
        if random.random() < 0.25:
            r = random.randint(0, img.shape[1] - 1)
            img[:, r, :] = 0.0
        if random.random() < 0.25:
            c = random.randint(0, img.shape[2] - 1)
            img[:, :, c] = 0.0
        return img

    def __getitem__(self, idx: int):
        img = self.X[idx]
        if self.augment:
            img = self._augment(img)
        x = torch.from_numpy(img.copy())
        if self.y is None:
            return x, idx
        return x, int(self.y[idx]), idx


class StrongCNN(nn.Module):
    """Compact CNN sized for 48x64 inputs."""

    def __init__(self, num_classes: int = NUM_CLASSES, drop: float = 0.3, base: int = 48):
        super().__init__()

        def block(c_in, c_out, pool=True):
            layers = [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.b1 = block(3, base)            # 48x64 -> 24x32
        self.b2 = block(base, base * 2)      # -> 12x16
        self.b3 = block(base * 2, base * 4)  # -> 6x8
        self.b4 = block(base * 4, base * 8, pool=False)  # 6x8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        feat = base * 8 * 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(feat, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        a = self.gap(x).flatten(1)
        m = self.gmp(x).flatten(1)
        return self.head(torch.cat([a, m], dim=1))


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def train_one(model, train_ds, val_ds, epochs, batch, lr, wd, label_smoothing, mixup_alpha):
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = -1.0
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_seen = 0
        tot_correct = 0
        for x, y, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x_mixed, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
            logits = model(x_mixed)
            loss = lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot_loss += loss.item() * x.size(0)
            tot_seen += x.size(0)
            with torch.no_grad():
                tot_correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        train_loss = tot_loss / tot_seen
        train_acc = tot_correct / tot_seen

        # eval
        model.eval()
        v_correct = 0
        v_seen = 0
        v_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                v_loss += crit(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_seen += x.size(0)
        val_acc = v_correct / v_seen
        val_loss = v_loss / v_seen
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"   epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    return best_acc, best_epoch, history


@torch.no_grad()
def predict_with_tta(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    n = X.shape[0]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        chunk = torch.from_numpy(X[s:e]).to(DEVICE)
        chunk_flip = torch.from_numpy(X[s:e, :, :, ::-1].copy()).to(DEVICE)
        p1 = torch.softmax(model(chunk), dim=1)
        p2 = torch.softmax(model(chunk_flip), dim=1)
        out[s:e] = ((p1 + p2) / 2).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 7])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--drop", type=float, default=0.3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device = {DEVICE}", flush=True)

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)

    print("Preloading train images...", flush=True)
    t0 = time.time()
    X_train = preload_all(args.data_root / "train_images", train_df["id"].tolist())
    X_test = preload_all(args.data_root / "test_images", test_df["id"].tolist())
    print(f"  preloaded in {time.time() - t0:.1f}s | X_train {X_train.shape} X_test {X_test.shape}", flush=True)

    y = train_df["target"].to_numpy()
    n_train = len(train_df)
    n_test = len(test_df)

    seed_oof = []   # list of (n_train, 4)
    seed_test = []  # list of (n_test, 4)
    seed_summaries = []

    for seed in args.seeds:
        print(f"\n=== SEED {seed} ===", flush=True)
        seed_everything(seed)
        oof = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)
        test_acc = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y), start=1):
            print(f"-- fold {fold} (seed {seed}) --", flush=True)
            train_ds = ArrayDataset(X_train[tr_idx], y[tr_idx], augment=True)
            val_ds = ArrayDataset(X_train[va_idx], y[va_idx], augment=False)
            model = StrongCNN(drop=args.drop, base=args.base).to(DEVICE)
            best_acc, best_epoch, history = train_one(
                model, train_ds, val_ds,
                epochs=args.epochs, batch=args.batch,
                lr=args.lr, wd=args.wd,
                label_smoothing=args.label_smoothing,
                mixup_alpha=args.mixup,
            )
            val_probs = predict_with_tta(model, X_train[va_idx])
            test_probs = predict_with_tta(model, X_test)
            oof[va_idx] = val_probs
            test_acc += test_probs / args.folds
            fold_scores.append(float(best_acc))
            print(f"   fold {fold} best_val_acc={best_acc:.4f} (epoch {best_epoch})", flush=True)

        oof_acc = accuracy_score(y, oof.argmax(1))
        seed_oof.append(oof)
        seed_test.append(test_acc)
        seed_summaries.append({
            "seed": seed,
            "fold_scores": fold_scores,
            "oof_accuracy": float(oof_acc),
        })
        np.save(args.output_dir / f"oof_seed{seed}.npy", oof)
        np.save(args.output_dir / f"test_seed{seed}.npy", test_acc)
        print(f"=== seed {seed} OOF acc = {oof_acc:.4f} ===", flush=True)

    # average across seeds
    oof_avg = np.mean(seed_oof, axis=0)
    test_avg = np.mean(seed_test, axis=0)
    np.save(args.output_dir / "oof_probs.npy", oof_avg)
    np.save(args.output_dir / "test_probs.npy", test_avg)
    final_acc = accuracy_score(y, oof_avg.argmax(1))
    cm = confusion_matrix(y, oof_avg.argmax(1))

    submission = test_df.copy()
    submission["target"] = test_avg.argmax(1).astype(int)
    sub_path = args.output_dir / "submission_strong.csv"
    submission.to_csv(sub_path, index=False)

    summary = {
        "device": str(DEVICE),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "seed_summaries": seed_summaries,
        "ensemble_oof_accuracy": float(final_acc),
        "confusion_matrix": cm.tolist(),
        "submission_path": str(sub_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"final_oof": final_acc, "submission": str(sub_path)}, indent=2))


if __name__ == "__main__":
    main()

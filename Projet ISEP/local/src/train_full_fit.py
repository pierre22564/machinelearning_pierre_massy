"""Final fit on 100% of training data (no holdout) with multiple seeds.

Self-contained (no cross-import of train_strong) to avoid import-hang issues.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 4
PAD_H, PAD_W = 48, 64
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_image_tensor(path: Path) -> np.ndarray:
    # Keep uint8 to minimize memory (float32 conversion happens in Dataset/__getitem__).
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    pt = (PAD_H - h) // 2; pb = PAD_H - h - pt
    pl = (PAD_W - w) // 2; pr = PAD_W - w - pl
    arr = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)))
    return arr.transpose(2, 0, 1)  # uint8 CHW


def preload_all(image_dir: Path, ids: Iterable[int]) -> np.ndarray:
    n = len(list(ids)) if not hasattr(ids, "__len__") else len(ids)
    ids_list = list(ids)
    out = np.zeros((len(ids_list), 3, PAD_H, PAD_W), dtype=np.uint8)
    for i, sid in enumerate(ids_list):
        out[i] = load_image_tensor(image_dir / f"img_{sid + 1}.png")
    return out


class ArrayDataset(Dataset):
    def __init__(self, X, y, augment):
        self.X = X; self.y = y; self.augment = augment
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0  # uint8 → float32 on-the-fly
        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
            if random.random() < 0.5:
                img = img + np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
            if random.random() < 0.25:
                r = random.randint(0, img.shape[1] - 1); img[:, r, :] = 0.0
            if random.random() < 0.25:
                c = random.randint(0, img.shape[2] - 1); img[:, :, c] = 0.0
        x = torch.from_numpy(img.copy())
        if self.y is None:
            return x, idx
        return x, int(self.y[idx]), idx


class StrongCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, drop=0.3, base=48):
        super().__init__()
        def block(ci, co, pool=True):
            layers = [
                nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.SiLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.SiLU(inplace=True),
            ]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        self.b1 = block(3, base)
        self.b2 = block(base, base * 2)
        self.b3 = block(base * 2, base * 4)
        self.b4 = block(base * 4, base * 8, pool=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        feat = base * 8 * 2
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(drop), nn.Linear(feat, 128),
            nn.SiLU(inplace=True), nn.Dropout(drop), nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        a = self.gap(x).flatten(1); m = self.gmp(x).flatten(1)
        return self.head(torch.cat([a, m], dim=1))


def mixup_batch(x, y, alpha):
    if alpha <= 0: return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha)); lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def predict_with_tta(model, X, batch=256):
    model.eval()
    n = X.shape[0]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        xb = X[s:e].astype(np.float32) / 255.0  # uint8 → float32
        c1 = torch.from_numpy(xb).to(DEVICE)
        c2 = torch.from_numpy(xb[:, :, :, ::-1].copy()).to(DEVICE)
        p = (torch.softmax(model(c1), 1) + torch.softmax(model(c2), 1)) / 2
        out[s:e] = p.cpu().numpy()
    return out


def main() -> None:
    print("BOOT", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 7, 555, 999])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.1)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--drop", type=float, default=0.3)
    args = p.parse_args()

    print(f"Device = {DEVICE}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if DEVICE.type == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    print(f"dfs: train={len(train_df)} test={len(test_df)}", flush=True)

    t0 = time.time()
    # Fast path: load pre-built npy bundles (single open, bypasses macOS provenance hang).
    npy_train = Path('/tmp/mpa_data/train_X.npy')
    npy_test = Path('/tmp/mpa_data/test_X.npy')
    if npy_train.exists() and npy_test.exists():
        print("Loading pre-built npy bundles...", flush=True)
        X_train = np.load(npy_train)
        X_test = np.load(npy_test)
        print(f"  train {X_train.shape} test {X_test.shape} in {time.time()-t0:.1f}s", flush=True)
    else:
        print("Preloading train...", flush=True)
        X_train = preload_all(args.data_root / "train_images", train_df["id"].tolist())
        print(f"  train {X_train.shape} in {time.time()-t0:.1f}s", flush=True)
        print("Preloading test...", flush=True)
        X_test = preload_all(args.data_root / "test_images", test_df["id"].tolist())
        print(f"  test {X_test.shape} total {time.time()-t0:.1f}s", flush=True)
    y = train_df["target"].to_numpy()

    test_avg = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
    for seed in args.seeds:
        print(f"\n=== full-fit seed {seed} (base={args.base}) ===", flush=True)
        seed_everything(seed)
        ds = ArrayDataset(X_train, y, augment=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
        model = StrongCNN(drop=args.drop, base=args.base).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        for epoch in range(1, args.epochs + 1):
            model.train()
            tot, seen, correct = 0.0, 0, 0
            for x, y_b, _ in loader:
                x = x.to(DEVICE); y_b = y_b.to(DEVICE)
                xm, ya, yb, lam = mixup_batch(x, y_b, args.mixup)
                logits = model(xm)
                loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
                optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
                tot += loss.item() * x.size(0); seen += x.size(0)
                with torch.no_grad():
                    correct += (logits.argmax(1) == y_b).sum().item()
            sched.step()
            print(f"  epoch={epoch:02d} loss={tot/seen:.4f} acc={correct/seen:.4f}", flush=True)
        probs = predict_with_tta(model, X_test)
        test_avg += probs / len(args.seeds)
        np.save(args.output_dir / f"test_full_seed{seed}_base{args.base}.npy", probs)

    np.save(args.output_dir / "test_full_probs.npy", test_avg)
    sub = test_df.copy()
    sub["target"] = test_avg.argmax(1).astype(int)
    sp = args.output_dir / "submission_full_fit.csv"
    sub.to_csv(sp, index=False)
    print(json.dumps({"submission": str(sp)}, indent=2))


if __name__ == "__main__":
    main()

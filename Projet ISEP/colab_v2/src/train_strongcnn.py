"""StrongCNN full-fit with heavy TTA. Supports big batches and multiple bases.

Heavy TTA : orig + hflip + shift±1 + shift±2 (5-way avg) pour les predicts test.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArrayDataset(Dataset):
    def __init__(self, X, y, augment):
        self.X = X; self.y = y; self.augment = augment
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
            if random.random() < 0.5:
                img = img + np.random.normal(0.0, 0.012, img.shape).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
            if random.random() < 0.3:
                r = random.randint(0, img.shape[1] - 1); img[:, r, :] = 0.0
            if random.random() < 0.3:
                c = random.randint(0, img.shape[2] - 1); img[:, :, c] = 0.0
            # small shift
            if random.random() < 0.4:
                sh = random.randint(-2, 2); sw = random.randint(-2, 2)
                if sh != 0 or sw != 0:
                    img = np.roll(img, shift=(sh, sw), axis=(1, 2))
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
        self.b1 = block(3, base); self.b2 = block(base, base * 2)
        self.b3 = block(base * 2, base * 4); self.b4 = block(base * 4, base * 8, pool=False)
        self.gap = nn.AdaptiveAvgPool2d(1); self.gmp = nn.AdaptiveMaxPool2d(1)
        feat = base * 8 * 2
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(drop), nn.Linear(feat, 256),
            nn.SiLU(inplace=True), nn.Dropout(drop), nn.Linear(256, num_classes),
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
def predict_heavy_tta(model, X, batch=512):
    """5-way TTA: orig + hflip + 4 shifts (±1 horiz/vert)."""
    model.eval(); n = X.shape[0]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    def shift(arr, sh, sw):
        return np.roll(arr, shift=(sh, sw), axis=(2, 3))

    for s in range(0, n, batch):
        e = min(n, s + batch)
        xb = X[s:e].astype(np.float32) / 255.0
        variants = [
            xb,
            xb[:, :, :, ::-1].copy(),
            shift(xb, 0, 1),
            shift(xb, 0, -1),
            shift(xb, 1, 0),
        ]
        acc = None
        for v in variants:
            t = torch.from_numpy(np.ascontiguousarray(v)).to(DEVICE)
            p = torch.softmax(model(t), 1)
            acc = p if acc is None else acc + p
        out[s:e] = (acc / len(variants)).cpu().numpy()
    return out


def main() -> None:
    print("STRONGCNN BOOT", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--train-npy", type=Path, required=True)
    p.add_argument("--test-npy", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 7, 555, 999, 2024, 31337, 101])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--drop", type=float, default=0.3)
    p.add_argument("--extra-data-npy", type=Path, default=None,
                   help="Optional extra uint8 npy to concat to train (for pseudo)")
    p.add_argument("--extra-labels-npy", type=Path, default=None)
    args = p.parse_args()

    print(f"Device = {DEVICE} | base={args.base} batch={args.batch}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    X_train = np.load(args.train_npy)
    X_test = np.load(args.test_npy)
    y = train_df["target"].to_numpy()

    if args.extra_data_npy is not None and args.extra_labels_npy is not None:
        X_extra = np.load(args.extra_data_npy)
        y_extra = np.load(args.extra_labels_npy)
        X_train = np.concatenate([X_train, X_extra], axis=0)
        y = np.concatenate([y, y_extra], axis=0)
        print(f"  added {len(X_extra)} extra samples → total {len(X_train)}", flush=True)

    print(f"Loaded: train {X_train.shape}, test {X_test.shape}", flush=True)

    test_avg = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
    for seed in args.seeds:
        print(f"\n=== seed {seed} (base={args.base}) ===", flush=True)
        seed_everything(seed)
        ds = ArrayDataset(X_train, y, augment=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
        model = StrongCNN(drop=args.drop, base=args.base).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        t_seed = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            tot, seen, correct = 0.0, 0, 0
            for x, y_b, _ in loader:
                x = x.to(DEVICE, non_blocking=True); y_b = y_b.to(DEVICE, non_blocking=True)
                xm, ya, yb, lam = mixup_batch(x, y_b, args.mixup)
                logits = model(xm)
                loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
                optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
                tot += loss.item() * x.size(0); seen += x.size(0)
                with torch.no_grad():
                    correct += (logits.argmax(1) == y_b).sum().item()
            sched.step()
            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"  epoch={epoch:02d} loss={tot/seen:.4f} acc={correct/seen:.4f}", flush=True)
        probs = predict_heavy_tta(model, X_test)
        test_avg += probs / len(args.seeds)
        np.save(args.output_dir / f"test_seed{seed}_base{args.base}.npy", probs)
        print(f"  seed {seed} done in {time.time()-t_seed:.1f}s", flush=True)

    np.save(args.output_dir / "test_probs.npy", test_avg)
    sub = test_df.copy()
    sub["target"] = test_avg.argmax(1).astype(int)
    sp = args.output_dir / f"submission_strongcnn_base{args.base}.csv"
    sub.to_csv(sp, index=False)
    print(json.dumps({"submission": str(sp)}, indent=2))


if __name__ == "__main__":
    main()

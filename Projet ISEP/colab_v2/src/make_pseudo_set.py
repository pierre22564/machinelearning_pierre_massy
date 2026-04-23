"""Build pseudo-label dataset from an ensemble test probs file.

Given a blended test probs npy and a threshold, extract high-confidence test
samples + predicted labels → save as uint8 npy + int64 npy, ready to concat
to train. Prints class distribution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-probs", type=Path, required=True)
    p.add_argument("--test-npy", type=Path, required=True)
    p.add_argument("--output-X", type=Path, required=True)
    p.add_argument("--output-y", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.98)
    args = p.parse_args()

    probs = np.load(args.test_probs)
    X_test = np.load(args.test_npy)
    conf = probs.max(1)
    labels = probs.argmax(1)
    mask = conf > args.threshold
    print(f"threshold={args.threshold} → keeping {mask.sum()}/{len(mask)}")
    print(f"class dist: {dict(Counter(labels[mask].tolist()))}")
    print(f"min kept conf: {conf[mask].min():.4f}  mean: {conf[mask].mean():.4f}")

    args.output_X.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_X, X_test[mask])
    np.save(args.output_y, labels[mask].astype(np.int64))
    print(f"saved {args.output_X} shape={X_test[mask].shape}")
    print(f"saved {args.output_y} shape={labels[mask].shape}")


if __name__ == "__main__":
    main()

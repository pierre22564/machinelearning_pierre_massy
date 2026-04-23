"""Build stack_test.npy = 10-fold LogReg stacker test probs on CNN + features OOFs.

Used as pseudo-labeling source. Reads every oof_seed*.npy / test_seed*.npy in the
given --cnn-dir(s), plus the features dir oof_probs.npy/test_probs.npy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--cnn-dir", type=Path, action="append", required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    y = train_df["target"].to_numpy()

    cnn_oofs, cnn_tests = [], []
    for d in args.cnn_dir:
        for oof_path in sorted(d.glob("oof_seed*.npy")):
            seed = oof_path.stem.replace("oof_seed", "")
            tpath = d / f"test_seed{seed}.npy"
            if tpath.exists():
                cnn_oofs.append(np.load(oof_path))
                cnn_tests.append(np.load(tpath))
                print(f"  loaded {d.name}/seed{seed}")
    cnn_oof = np.mean(cnn_oofs, axis=0)
    cnn_test = np.mean(cnn_tests, axis=0)
    print(f"CNN avg OOF acc = {accuracy_score(y, cnn_oof.argmax(1)):.4f} ({len(cnn_oofs)} seeds)")

    feat_oof = np.load(args.features_dir / "oof_probs.npy")
    feat_test = np.load(args.features_dir / "test_probs.npy")
    print(f"FEAT OOF acc = {accuracy_score(y, feat_oof.argmax(1)):.4f}")

    X_meta_tr = np.concatenate([cnn_oof, feat_oof], axis=1)
    X_meta_te = np.concatenate([cnn_test, feat_test], axis=1)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    stack_oof = np.zeros((len(train_df), 4), dtype=np.float32)
    stack_test = np.zeros((len(test_df), 4), dtype=np.float32)
    for tr, va in skf.split(X_meta_tr, y):
        clf = LogisticRegression(max_iter=4000, C=args.C)
        clf.fit(X_meta_tr[tr], y[tr])
        stack_oof[va] = clf.predict_proba(X_meta_tr[va])
        stack_test += clf.predict_proba(X_meta_te) / 10
    print(f"STACK OOF acc = {accuracy_score(y, stack_oof.argmax(1)):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, stack_test)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

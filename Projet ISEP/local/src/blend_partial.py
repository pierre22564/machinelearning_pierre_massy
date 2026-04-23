"""Quick partial blend using whatever seed-level CNN files exist + features model.

Used to get an early Kaggle LB read while the full multi-seed CNN is still running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--cnn-dir", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    y = train_df["target"].to_numpy()

    # average available CNN seeds
    cnn_oofs, cnn_tests = [], []
    for s in args.seeds:
        oof_path = args.cnn_dir / f"oof_seed{s}.npy"
        test_path = args.cnn_dir / f"test_seed{s}.npy"
        if not (oof_path.exists() and test_path.exists()):
            print(f"  skip seed {s} (files missing)")
            continue
        cnn_oofs.append(np.load(oof_path))
        cnn_tests.append(np.load(test_path))
        print(f"  loaded CNN seed {s} acc={accuracy_score(y, cnn_oofs[-1].argmax(1)):.4f}")
    cnn_oof = np.mean(cnn_oofs, axis=0)
    cnn_test = np.mean(cnn_tests, axis=0)
    cnn_acc = float(accuracy_score(y, cnn_oof.argmax(1)))
    print(f"CNN avg ({len(cnn_oofs)} seeds) OOF = {cnn_acc:.4f}")

    feat_oof = np.load(args.features_dir / "oof_probs.npy")
    feat_test = np.load(args.features_dir / "test_probs.npy")
    feat_acc = float(accuracy_score(y, feat_oof.argmax(1)))
    print(f"FEATURES OOF = {feat_acc:.4f}")

    # weighted blend by acc^k
    k = 6.0
    accs = np.array([cnn_acc, feat_acc])
    w = (accs ** k); w /= w.sum()
    weighted_oof = w[0] * cnn_oof + w[1] * feat_oof
    weighted_test = w[0] * cnn_test + w[1] * feat_test
    weighted_acc = float(accuracy_score(y, weighted_oof.argmax(1)))
    print(f"weighted ({w}) OOF = {weighted_acc:.4f}")

    # stacking with logistic
    X_meta_train = np.concatenate([cnn_oof, feat_oof], axis=1)
    X_meta_test = np.concatenate([cnn_test, feat_test], axis=1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    stack_oof = np.zeros((len(train_df), 4), dtype=np.float32)
    stack_test = np.zeros((len(test_df), 4), dtype=np.float32)
    for tr, va in skf.split(X_meta_train, y):
        clf = LogisticRegression(max_iter=4000, C=2.0)
        clf.fit(X_meta_train[tr], y[tr])
        stack_oof[va] = clf.predict_proba(X_meta_train[va])
        stack_test += clf.predict_proba(X_meta_test) / 10
    stack_acc = float(accuracy_score(y, stack_oof.argmax(1)))
    print(f"stack OOF = {stack_acc:.4f}")

    # final = 0.5 stack + 0.5 weighted
    final_oof = 0.5 * stack_oof + 0.5 * weighted_oof
    final_test = 0.5 * stack_test + 0.5 * weighted_test
    final_acc = float(accuracy_score(y, final_oof.argmax(1)))
    print(f"FINAL OOF = {final_acc:.4f}")

    for tag, probs in [
        ("cnn_only", cnn_test),
        ("features_only", feat_test),
        ("weighted", weighted_test),
        ("stack", stack_test),
        ("final", final_test),
    ]:
        sub = test_df.copy()
        sub["target"] = probs.argmax(1).astype(int)
        sub.to_csv(args.output_dir / f"submission_partial_{tag}.csv", index=False)
    summary = {
        "seeds_used": args.seeds,
        "cnn_acc": cnn_acc,
        "features_acc": feat_acc,
        "weighted_acc": weighted_acc,
        "weighted_weights": w.tolist(),
        "stack_acc": stack_acc,
        "final_acc": final_acc,
        "confusion_matrix_final": confusion_matrix(y, final_oof.argmax(1)).tolist(),
    }
    (args.output_dir / "partial_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

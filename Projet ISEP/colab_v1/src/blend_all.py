"""Blend all available CNN seeds (from any run dirs) + features model.

Pulls oof_seed{S}.npy / test_seed{S}.npy from every dir given via --cnn-dir, plus
the engineered features model dir. Stacks with 10-fold logistic and emits multiple
candidate submissions.
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


def load_seed_pairs(d: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Find oof_seed{S}.npy + test_seed{S}.npy pairs."""
    pairs = []
    for oof_path in sorted(d.glob("oof_seed*.npy")):
        seed = oof_path.stem.replace("oof_seed", "")
        test_path = d / f"test_seed{seed}.npy"
        if test_path.exists():
            pairs.append((f"{d.name}/seed{seed}", np.load(oof_path), np.load(test_path)))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--cnn-dir", type=Path, action="append", required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--full-test", type=Path, action="append", default=[],
                   help="Optional npy files of full-fit test probs (used in test-only blend)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    y = train_df["target"].to_numpy()

    all_cnn_oofs, all_cnn_tests, names = [], [], []
    per_seed_acc: dict[str, float] = {}
    for d in args.cnn_dir:
        for name, oof, test in load_seed_pairs(d):
            all_cnn_oofs.append(oof)
            all_cnn_tests.append(test)
            names.append(name)
            per_seed_acc[name] = float(accuracy_score(y, oof.argmax(1)))
            print(f"  {name} acc={per_seed_acc[name]:.4f}")
    cnn_oof_avg = np.mean(all_cnn_oofs, axis=0)
    cnn_test_avg = np.mean(all_cnn_tests, axis=0)
    cnn_acc = float(accuracy_score(y, cnn_oof_avg.argmax(1)))
    print(f"CNN avg ({len(names)} seeds) OOF = {cnn_acc:.4f}")

    feat_oof = np.load(args.features_dir / "oof_probs.npy")
    feat_test = np.load(args.features_dir / "test_probs.npy")
    feat_acc = float(accuracy_score(y, feat_oof.argmax(1)))
    print(f"FEATURES OOF = {feat_acc:.4f}")

    # weighted blend (acc^k)
    k = 6.0
    accs = np.array([cnn_acc, feat_acc])
    w = (accs ** k); w /= w.sum()
    weighted_oof = w[0] * cnn_oof_avg + w[1] * feat_oof
    weighted_test = w[0] * cnn_test_avg + w[1] * feat_test
    weighted_acc = float(accuracy_score(y, weighted_oof.argmax(1)))
    print(f"weighted OOF = {weighted_acc:.4f} (w={w})")

    # stacking
    X_meta_train = np.concatenate([cnn_oof_avg, feat_oof], axis=1)
    X_meta_test = np.concatenate([cnn_test_avg, feat_test], axis=1)
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

    final_oof = 0.5 * stack_oof + 0.5 * weighted_oof
    final_test = 0.5 * stack_test + 0.5 * weighted_test
    final_acc = float(accuracy_score(y, final_oof.argmax(1)))
    print(f"FINAL (stack+weighted) OOF = {final_acc:.4f}")

    # if full-fit test probs given, blend them into the test side only
    full_test_blend = None
    if args.full_test:
        ft = np.mean([np.load(p) for p in args.full_test], axis=0)
        full_test_blend = 0.5 * stack_test + 0.5 * ft
        sub = test_df.copy()
        sub["target"] = full_test_blend.argmax(1).astype(int)
        sub.to_csv(args.output_dir / "submission_with_fullfit.csv", index=False)

    for tag, probs in [
        ("cnn_only", cnn_test_avg),
        ("features_only", feat_test),
        ("weighted", weighted_test),
        ("stack", stack_test),
        ("final", final_test),
    ]:
        sub = test_df.copy()
        sub["target"] = probs.argmax(1).astype(int)
        sub.to_csv(args.output_dir / f"submission_{tag}.csv", index=False)
    summary = {
        "per_seed_acc": per_seed_acc,
        "cnn_avg_acc": cnn_acc,
        "features_acc": feat_acc,
        "weighted_acc": weighted_acc,
        "weighted_weights": w.tolist(),
        "stack_acc": stack_acc,
        "final_acc": final_acc,
        "confusion_matrix_final": confusion_matrix(y, final_oof.argmax(1)).tolist(),
        "n_seeds_used": len(names),
        "full_fit_test_files": [str(p) for p in args.full_test],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

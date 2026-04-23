"""Logistic-regression stacking + plain blend across multiple OOF/test prob files.

Usage:
  python stack_blend.py \
    --train-csv ... --sample-submission ... --output-dir ... \
    --component cnn /path/to/run_strong \
    --component features /path/to/run_features_v2
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


def load_component(name: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    oof = np.load(path / "oof_probs.npy")
    test = np.load(path / "test_probs.npy")
    return oof.astype(np.float32), test.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", action="append", nargs=2, metavar=("name", "dir"), required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    y = train_df["target"].to_numpy()

    components = []
    individual_accs = {}
    for name, p in args.component:
        oof, test = load_component(name, Path(p))
        components.append((name, oof, test))
        individual_accs[name] = float(accuracy_score(y, oof.argmax(1)))

    # 1) plain mean blend (equal weights)
    plain_oof = np.mean([c[1] for c in components], axis=0)
    plain_test = np.mean([c[2] for c in components], axis=0)
    plain_acc = float(accuracy_score(y, plain_oof.argmax(1)))

    # 2) weighted blend by OOF accuracy^k
    k = 6.0
    raw_w = np.array([individual_accs[c[0]] ** k for c in components], dtype=np.float64)
    w = raw_w / raw_w.sum()
    weighted_oof = np.einsum("i,ijk->jk", w, np.stack([c[1] for c in components]))
    weighted_test = np.einsum("i,ijk->jk", w, np.stack([c[2] for c in components]))
    weighted_acc = float(accuracy_score(y, weighted_oof.argmax(1)))

    # 3) logistic stacking on OOF probs
    X_meta_train = np.concatenate([c[1] for c in components], axis=1)
    X_meta_test = np.concatenate([c[2] for c in components], axis=1)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    stack_oof = np.zeros((len(train_df), 4), dtype=np.float32)
    stack_test = np.zeros((len(test_df), 4), dtype=np.float32)
    for tr, va in skf.split(X_meta_train, y):
        clf = LogisticRegression(max_iter=4000, C=2.0, multi_class="multinomial")
        clf.fit(X_meta_train[tr], y[tr])
        stack_oof[va] = clf.predict_proba(X_meta_train[va])
        stack_test += clf.predict_proba(X_meta_test) / args.folds
    stack_acc = float(accuracy_score(y, stack_oof.argmax(1)))

    # blend stack + weighted (often robust)
    final_oof = 0.5 * stack_oof + 0.5 * weighted_oof
    final_test = 0.5 * stack_test + 0.5 * weighted_test
    final_acc = float(accuracy_score(y, final_oof.argmax(1)))

    np.save(args.output_dir / "stack_oof.npy", stack_oof)
    np.save(args.output_dir / "stack_test.npy", stack_test)
    np.save(args.output_dir / "weighted_oof.npy", weighted_oof)
    np.save(args.output_dir / "weighted_test.npy", weighted_test)
    np.save(args.output_dir / "final_oof.npy", final_oof)
    np.save(args.output_dir / "final_test.npy", final_test)

    # write 3 candidate submissions
    for tag, probs in [("plain", plain_test), ("weighted", weighted_test),
                       ("stack", stack_test), ("final", final_test)]:
        sub = test_df.copy()
        sub["target"] = probs.argmax(1).astype(int)
        sub.to_csv(args.output_dir / f"submission_{tag}.csv", index=False)

    summary = {
        "components": individual_accs,
        "plain_oof_acc": plain_acc,
        "weighted_oof_acc": weighted_acc,
        "weighted_weights": w.tolist(),
        "stack_oof_acc": stack_acc,
        "final_oof_acc": final_acc,
        "confusion_matrix_final": confusion_matrix(y, final_oof.argmax(1)).tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

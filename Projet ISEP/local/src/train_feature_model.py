from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["0 person", "1 person", "2 persons", "3 persons"],
        yticklabels=["0 person", "1 person", "2 persons", "3 persons"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def extract_features(image_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    row = arr.mean(axis=1)
    col = arr.mean(axis=0)
    row_max = arr.max(axis=1)
    col_max = arr.max(axis=0)
    flat = arr.ravel()
    topk = np.sort(flat)[-24:]

    feature_list: list[float] = [flat.mean(), flat.std(), flat.max()]
    for q in [0.95, 0.98, 0.99, 0.995, 0.999]:
        feature_list.append(float(np.quantile(flat, q)))

    for thr in [0.985, 0.99, 0.995, 0.998]:
        mask = arr > np.quantile(flat, thr)
        labeled, num = ndi.label(mask)
        objects = ndi.find_objects(labeled)
        areas = sorted(
            [
                (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
                for sl in objects
                if sl is not None
            ],
            reverse=True,
        )
        coords = np.argwhere(mask)
        if len(coords):
            cy, cx = coords.mean(axis=0)
            sy, sx = coords.std(axis=0)
        else:
            cy = cx = sy = sx = 0.0

        feature_list.extend([float(mask.sum()), float(num), float(cy), float(cx), float(sy), float(sx)])
        feature_list.extend([float(val) for val in (areas[:4] + [0] * (4 - len(areas[:4])))])

    max_filtered = ndi.maximum_filter(arr, size=3)
    peaks = (arr == max_filtered) & (arr > np.quantile(flat, 0.995))
    peak_coords = np.argwhere(peaks)
    feature_list.append(float(peaks.sum()))
    if len(peak_coords):
        feature_list.extend([float(v) for v in peak_coords.mean(axis=0)])
        feature_list.extend([float(v) for v in peak_coords.std(axis=0)])
    else:
        feature_list.extend([0.0, 0.0, 0.0, 0.0])

    return np.concatenate([np.asarray(feature_list, dtype=np.float32), row, col, row_max, col_max, topk]).astype(
        np.float32
    )


def build_feature_matrix(image_dir: Path, ids: list[int], cache_path: Path | None = None) -> np.ndarray:
    if cache_path and cache_path.exists():
        return np.load(cache_path)

    features = []
    for idx, sample_id in enumerate(ids, start=1):
        image_path = image_dir / f"img_{sample_id + 1}.png"
        features.append(extract_features(image_path))
        if idx % 2000 == 0 or idx == len(ids):
            print(f"extracted {idx}/{len(ids)} features from {image_dir.name}", flush=True)
    matrix = np.stack(features)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a feature-based histogram gradient boosting model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--l2", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)

    train_dir = args.data_root / "train_images"
    test_dir = args.data_root / "test_images"

    X_train = build_feature_matrix(train_dir, train_df["id"].tolist(), output_dir / "train_features.npy")
    X_test = build_feature_matrix(test_dir, test_df["id"].tolist(), output_dir / "test_features.npy")
    y = train_df["target"].to_numpy()

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(train_df), 4), dtype=np.float32)
    test_probs = np.zeros((len(test_df), 4), dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train, y), start=1):
        model = HistGradientBoostingClassifier(
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            l2_regularization=args.l2,
            random_state=args.seed + fold,
        )
        model.fit(X_train[train_idx], y[train_idx])
        val_probs = model.predict_proba(X_train[val_idx])
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test) / args.folds
        fold_acc = accuracy_score(y[val_idx], val_probs.argmax(axis=1))
        fold_scores.append(float(fold_acc))
        print(f"fold={fold} val_acc={fold_acc:.4f}", flush=True)

    oof_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    cm = confusion_matrix(y, oof_probs.argmax(axis=1))
    plot_confusion(cm, output_dir / "confusion_matrix.png", "Feature model OOF confusion matrix")

    np.save(output_dir / "oof_probs.npy", oof_probs)
    np.save(output_dir / "test_probs.npy", test_probs)

    submission = test_df.copy()
    submission["target"] = test_probs.argmax(axis=1).astype(int)
    submission_path = output_dir / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    save_json(
        {
            "experiment_name": "feature_hgb",
            "model_name": "HistGradientBoostingClassifier",
            "input_mode": "engineered_features",
            "fold_scores": fold_scores,
            "oof_accuracy": float(oof_acc),
            "submission_path": str(submission_path),
            "hyperparameters": {
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "max_iter": args.max_iter,
                "l2_regularization": args.l2,
            },
        },
        output_dir / "summary.json",
    )

    save_json(
        {
            "device": "cpu-feature-model",
            "experiments": [
                {
                    "experiment_name": "feature_hgb",
                    "model_name": "HistGradientBoostingClassifier",
                    "input_mode": "engineered_features",
                    "fold_scores": fold_scores,
                    "oof_accuracy": float(oof_acc),
                }
            ],
            "ensemble_weights": [{"experiment_name": "feature_hgb", "weight": 1.0}],
            "ensemble_oof_accuracy": float(oof_acc),
            "submission_path": str(submission_path),
        },
        output_dir / "run_summary.json",
    )

    print(json.dumps({"ensemble_oof_accuracy": oof_acc, "submission_path": str(submission_path)}, indent=2))


if __name__ == "__main__":
    main()

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
from sklearn.metrics import accuracy_score, confusion_matrix


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend saved OOF and test probabilities from multiple runs.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        nargs=3,
        action="append",
        metavar=("NAME", "EXPERIMENT_DIR", "WEIGHT"),
        required=True,
        help="Component label, directory with oof_probs.npy/test_probs.npy, and raw weight.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    y_true = train_df["target"].to_numpy()

    raw_weights = np.array([float(item[2]) for item in args.component], dtype=np.float64)
    weights = raw_weights / raw_weights.sum()

    ensemble_oof = None
    ensemble_test = None
    component_summaries = []
    for (name, exp_dir_raw, _raw_weight), weight in zip(args.component, weights):
        exp_dir = Path(exp_dir_raw)
        oof = np.load(exp_dir / "oof_probs.npy")
        test = np.load(exp_dir / "test_probs.npy")
        ensemble_oof = oof * weight if ensemble_oof is None else ensemble_oof + oof * weight
        ensemble_test = test * weight if ensemble_test is None else ensemble_test + test * weight

        summary_path = exp_dir / "summary.json"
        component_summary = {"name": name, "weight": float(weight), "oof_accuracy": float(accuracy_score(y_true, oof.argmax(axis=1)))}
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            component_summary.update(
                {
                    "model_name": payload.get("model_name", name),
                    "input_mode": payload.get("input_mode", "unknown"),
                    "fold_scores": payload.get("fold_scores", []),
                }
            )
        component_summaries.append(component_summary)

    ensemble_preds = ensemble_oof.argmax(axis=1)
    ensemble_acc = accuracy_score(y_true, ensemble_preds)
    cm = confusion_matrix(y_true, ensemble_preds)
    plot_confusion(cm, output_dir / "ensemble_confusion_matrix.png", "Final blended ensemble confusion matrix")

    np.save(output_dir / "ensemble_oof_probs.npy", ensemble_oof)
    np.save(output_dir / "ensemble_test_probs.npy", ensemble_test)

    submission = test_df.copy()
    submission["target"] = ensemble_test.argmax(axis=1).astype(int)
    submission_path = output_dir / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    save_json(
        {
            "device": "blended-from-saved-runs",
            "ensemble_oof_accuracy": float(ensemble_acc),
            "submission_path": str(submission_path),
            "blend_components": component_summaries,
        },
        output_dir / "run_summary.json",
    )

    print(json.dumps({"ensemble_oof_accuracy": ensemble_acc, "submission_path": str(submission_path)}, indent=2))


if __name__ == "__main__":
    main()

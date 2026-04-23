"""Enhanced engineered-features model.

Adds delay-Doppler-aware peak features on top of the previous feature set:
- top-K peak intensities + their (row, col) coordinates
- pairwise distances between top-K peaks
- separate delay (row) projection peaks vs Doppler (col) projection peaks
- moments of the intensity distribution
- multi-resolution stats from box-blur of the image
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def extract_features(image_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    H, W = arr.shape
    flat = arr.ravel()

    feats: list[float] = []
    feats.extend([flat.mean(), flat.std(), flat.max(), flat.min()])
    for q in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
        feats.append(float(np.quantile(flat, q)))

    # multi-threshold blob stats
    for thr in [0.97, 0.98, 0.985, 0.99, 0.995, 0.998]:
        t = float(np.quantile(flat, thr))
        mask = arr > t
        labeled, num = ndi.label(mask)
        objects = ndi.find_objects(labeled)
        areas = sorted(
            [(sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
             for sl in objects if sl is not None], reverse=True
        )
        coords = np.argwhere(mask)
        if len(coords):
            cy, cx = coords.mean(axis=0)
            sy, sx = coords.std(axis=0)
        else:
            cy = cx = sy = sx = 0.0
        feats.extend([float(mask.sum()), float(num), float(cy), float(cx), float(sy), float(sx)])
        feats.extend([float(v) for v in (areas[:5] + [0] * (5 - len(areas[:5])))])

    # local-max peak detection at several scales
    for size in (3, 5, 7):
        max_filt = ndi.maximum_filter(arr, size=size)
        peaks_mask = (arr == max_filt) & (arr > np.quantile(flat, 0.99))
        peak_coords = np.argwhere(peaks_mask)
        peak_vals = arr[peaks_mask]
        feats.append(float(peaks_mask.sum()))
        if len(peak_coords):
            order = np.argsort(-peak_vals)[:5]
            top_coords = peak_coords[order]
            top_vals = peak_vals[order]
            for i in range(5):
                if i < len(order):
                    feats.extend([float(top_coords[i, 0]), float(top_coords[i, 1]), float(top_vals[i])])
                else:
                    feats.extend([0.0, 0.0, 0.0])
            # pairwise distances among top peaks
            if len(order) >= 2:
                d = []
                for i in range(len(order)):
                    for j in range(i + 1, len(order)):
                        d.append(float(np.linalg.norm(top_coords[i] - top_coords[j])))
                feats.extend([float(np.mean(d)), float(np.max(d)), float(np.min(d))])
            else:
                feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([0.0] * (5 * 3 + 3))

    # row / col projections (delay-only and Doppler-only)
    row = arr.mean(axis=1)
    col = arr.mean(axis=0)
    row_max = arr.max(axis=1)
    col_max = arr.max(axis=0)
    feats.extend([row.mean(), row.std(), row.max(), float(row.argmax())])
    feats.extend([col.mean(), col.std(), col.max(), float(col.argmax())])

    # peaks in projections
    for proj in (row_max, col_max):
        diff_sign = np.sign(np.diff(proj))
        # 1d local maxima
        is_peak = np.zeros_like(proj, dtype=bool)
        is_peak[1:-1] = (proj[1:-1] > proj[:-2]) & (proj[1:-1] > proj[2:]) & (proj[1:-1] > 0.5 * proj.max())
        feats.append(float(is_peak.sum()))

    # multi-resolution: box blur stats
    for k in (3, 5):
        blur = ndi.uniform_filter(arr, size=k)
        feats.extend([float(blur.max()), float(np.quantile(blur.ravel(), 0.99))])

    # intensity moments
    yy, xx = np.mgrid[0:H, 0:W]
    norm = arr.sum() + 1e-9
    cy_w = float((yy * arr).sum() / norm)
    cx_w = float((xx * arr).sum() / norm)
    feats.extend([cy_w, cx_w])

    # full projection vectors
    return np.concatenate([
        np.asarray(feats, dtype=np.float32),
        row, col, row_max, col_max,
    ]).astype(np.float32)


def build_matrix(image_dir: Path, ids: list[int], cache: Path | None) -> np.ndarray:
    if cache and cache.exists():
        return np.load(cache)
    out = []
    for i, sid in enumerate(ids, 1):
        out.append(extract_features(image_dir / f"img_{sid + 1}.png"))
        if i % 2000 == 0 or i == len(ids):
            print(f"  features {i}/{len(ids)} ({image_dir.name})", flush=True)
    M = np.stack(out)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, M)
    return M


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 7])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)

    X_train = build_matrix(args.data_root / "train_images", train_df["id"].tolist(),
                           args.output_dir / "train_features.npy")
    X_test = build_matrix(args.data_root / "test_images", test_df["id"].tolist(),
                          args.output_dir / "test_features.npy")
    y = train_df["target"].to_numpy()

    n_train, n_test = len(train_df), len(test_df)
    seed_oof, seed_test, seed_summaries = [], [], []
    for seed in args.seeds:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        oof = np.zeros((n_train, 4), dtype=np.float32)
        tst = np.zeros((n_test, 4), dtype=np.float32)
        fold_scores = []
        for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
            for params in (
                dict(max_depth=8, learning_rate=0.05, max_iter=600, l2_regularization=0.0),
                dict(max_depth=6, learning_rate=0.03, max_iter=800, l2_regularization=0.1),
            ):
                clf = HistGradientBoostingClassifier(**params, random_state=seed + fold)
                clf.fit(X_train[tr], y[tr])
                oof[va] += clf.predict_proba(X_train[va]) / 2
                tst += clf.predict_proba(X_test) / (args.folds * 2)
            fold_acc = accuracy_score(y[va], oof[va].argmax(1))
            fold_scores.append(float(fold_acc))
            print(f"  seed={seed} fold={fold} val_acc={fold_acc:.4f}", flush=True)
        seed_oof.append(oof)
        seed_test.append(tst)
        seed_summaries.append({
            "seed": seed,
            "fold_scores": fold_scores,
            "oof_accuracy": float(accuracy_score(y, oof.argmax(1))),
        })

    oof_avg = np.mean(seed_oof, axis=0)
    test_avg = np.mean(seed_test, axis=0)
    final = float(accuracy_score(y, oof_avg.argmax(1)))
    cm = confusion_matrix(y, oof_avg.argmax(1))
    np.save(args.output_dir / "oof_probs.npy", oof_avg)
    np.save(args.output_dir / "test_probs.npy", test_avg)

    sub = test_df.copy()
    sub["target"] = test_avg.argmax(1).astype(int)
    sp = args.output_dir / "submission_features_v2.csv"
    sub.to_csv(sp, index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "seed_summaries": seed_summaries,
        "ensemble_oof_accuracy": final,
        "confusion_matrix": cm.tolist(),
        "submission_path": str(sp),
    }, indent=2))
    print(json.dumps({"final_oof": final, "submission": str(sp)}, indent=2))


if __name__ == "__main__":
    main()

"""Final blender WITHOUT stack (stack killed LB in v1).

Takes N test_probs.npy files + optional accuracy weights, computes:
- pure mean
- geometric mean
- power mean (k=2,3)
- best-confidence mean (weighted by mean max prob)

Emits one submission per variant + saves all probs for later manual mixing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--probs", type=Path, nargs="+", required=True,
                   help="Per-source test_probs.npy files")
    p.add_argument("--names", type=str, nargs="+", default=None,
                   help="Same length as --probs, labels for logging")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    names = args.names or [p.stem for p in args.probs]

    all_probs = []
    stats = {}
    for name, path in zip(names, args.probs):
        pr = np.load(path)
        all_probs.append(pr)
        mconf = float(pr.max(1).mean())
        stats[name] = {"mean_max": mconf, "path": str(path)}
        print(f"  {name}: mean_max={mconf:.4f}")

    P = np.stack(all_probs, axis=0)  # (S, N, 4)
    S = P.shape[0]

    variants = {}
    variants["mean"] = P.mean(0)
    # geometric mean (log-average)
    variants["geom"] = np.exp(np.log(P + 1e-9).mean(0))
    variants["geom"] = variants["geom"] / variants["geom"].sum(1, keepdims=True)
    # power mean (k=2,3)
    for k in (2, 3):
        pm = (P ** k).mean(0) ** (1 / k)
        pm = pm / pm.sum(1, keepdims=True)
        variants[f"pow{k}"] = pm
    # confidence-weighted
    conf_w = np.array([s["mean_max"] for s in stats.values()])
    conf_w = conf_w ** 4
    conf_w = conf_w / conf_w.sum()
    variants["conf_weighted"] = (P * conf_w[:, None, None]).sum(0)
    print(f"conf_weighted weights: {dict(zip(names, conf_w.tolist()))}")

    for tag, pr in variants.items():
        sub = test_df.copy()
        sub["target"] = pr.argmax(1).astype(int)
        path = args.output_dir / f"submission_{tag}.csv"
        sub.to_csv(path, index=False)
        np.save(args.output_dir / f"probs_{tag}.npy", pr)
        dist = np.bincount(pr.argmax(1), minlength=4).tolist()
        print(f"  [{tag}] saved {path}  dist={dist}")

    (args.output_dir / "blend_summary.json").write_text(json.dumps({
        "sources": stats,
        "n_sources": S,
    }, indent=2))
    print("done.")


if __name__ == "__main__":
    main()

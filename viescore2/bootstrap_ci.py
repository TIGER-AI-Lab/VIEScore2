#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals and paired model comparisons.

A pitfall: headline deltas were
reported as point estimates while the repo's own paired bootstrap put the RL
gain's 95% CI across zero. Any number we now publish gets an interval, and any
claimed improvement gets a paired test on the SAME resampled samples.

Metrics, all recomputed from per-sample records so resampling is honest:
  * micro cell-F1 (sum tp/fp/fn over the resampled set, not a mean of F1s),
  * IoU_p (mean grid IoU over problem samples only),
  * per-axis Spearman with tie-averaged ranks.

    python viescore2/bootstrap_ci.py \
        eval_results/viescore2_sft.json \
        eval_results/viescore2_rl.json --n 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import rank_avg  # noqa: E402


def load(path):
    rows = json.load(open(path))["results"]
    out = {}
    for r in rows:
        out[r.get("id")] = {
            # EvalMuse rows have no localization GT; older result files
            # recorded spurious counts for them
            # 2026-07-22) — mask them out here so old and new files agree.
            "tp": None if r.get("source") == "EvalMuse" else r.get("cell_tp"),
            "fp": None if r.get("source") == "EvalMuse" else r.get("cell_fp"),
            "fn": None if r.get("source") == "EvalMuse" else r.get("cell_fn"),
            "iou": None if r.get("source") == "EvalMuse" else r.get("grid_iou"),
            "gt_score": r.get("gt_score"), "pred_score": r.get("pred_score"),
            "gt_axes": r.get("gt_axes"), "pred_pq": r.get("pred_pq"),
            "pred_sc": r.get("pred_sc"), "source": r.get("source", ""),
        }
    return out


def micro_f1(recs):
    recs = [r for r in recs if r["tp"] is not None]
    tp = sum(r["tp"] for r in recs); fp = sum(r["fp"] for r in recs)
    fn = sum(r["fn"] for r in recs)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def iou_p(recs):
    v = [r["iou"] for r in recs
         if r["iou"] is not None and r["tp"] is not None and (r["tp"] + r["fn"]) > 0]
    return float(np.mean(v)) if v else float("nan")


def spearman(recs, axis):
    idx = 0 if axis == "pq" else 1
    pk = "pred_pq" if axis == "pq" else "pred_sc"
    pairs = [(r[pk], r["gt_axes"][idx]) for r in recs
             if r.get(pk) is not None and r.get("gt_axes")
             and r["gt_axes"][idx] is not None and r["source"] != "COCO-real"]
    if len(pairs) < 5:
        return float("nan")
    a = np.array([p for p, _ in pairs], float)
    b = np.array([g for _, g in pairs], float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_avg(a), rank_avg(b))[0, 1])


METRICS = {"F1": micro_f1, "IoU_p": iou_p,
           "pq_rho": lambda r: spearman(r, "pq"),
           "sc_rho": lambda r: spearman(r, "sc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result JSONs; first is the reference")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    models = {}
    for p in args.results:
        models[Path(p).stem] = load(p)
    names = list(models)
    common = set.intersection(*[set(m) for m in models.values()])
    ids = sorted(common)
    print(f"models: {names}\naligned samples: {len(ids)}\n", flush=True)

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(ids), size=(args.n, len(ids)))

    point, boot = {}, {}
    for name, m in models.items():
        recs = [m[i] for i in ids]
        point[name] = {k: f(recs) for k, f in METRICS.items()}
        boot[name] = {k: np.empty(args.n) for k in METRICS}
        for b in range(args.n):
            sub = [recs[j] for j in draws[b]]
            for k, f in METRICS.items():
                boot[name][k][b] = f(sub)

    out = {"n_samples": len(ids), "n_boot": args.n, "models": {}, "paired": {}}
    print("=== point estimates with 95% CI ===")
    for name in names:
        out["models"][name] = {}
        for k in METRICS:
            v = point[name][k]
            lo, hi = np.nanpercentile(boot[name][k], [2.5, 97.5])
            out["models"][name][k] = {"point": round(float(v), 4),
                                      "ci95": [round(float(lo), 4), round(float(hi), 4)]}
            print(f"  {name:<28} {k:<7} {v:.4f}  [{lo:.4f}, {hi:.4f}]")

    # paired comparisons against the first model
    ref = names[0]
    for name in names[1:]:
        key = f"{name} - {ref}"
        out["paired"][key] = {}
        print(f"\n=== paired: {key} ===")
        for k in METRICS:
            d = boot[name][k] - boot[ref][k]
            diff = point[name][k] - point[ref][k]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            ppos = float(np.mean(d > 0))
            sig = "significant" if lo > 0 or hi < 0 else "NOT significant (CI spans 0)"
            out["paired"][key][k] = {"diff": round(float(diff), 4),
                                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                     "p_positive": round(ppos, 3), "significant": bool(lo > 0 or hi < 0)}
            print(f"  {k:<7} {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(>0)={ppos:.3f}  {sig}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute a result file's localization summary excluding score-only rows.

Result files written before 2026-07-22 aggregated EvalMuse rows into the
localization micrometrics even though those rows have NO spatial ground
truth ("unlabeled" is not "clean") — their all-zero GT injected spurious
false positives. run_eval.py now nulls the cell fields on such
rows at eval time; this script applies the identical exclusion to already
written files, in place, so no on-disk summary carries the polluted
numbers. Per-sample rows are untouched (bootstrap_ci.py always masked
EvalMuse itself, so CIs never saw the pollution).

    python viescore2/resummarize.py eval_results/foo.json ...
"""
import json
import sys


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def fix(path):
    d = json.load(open(path))
    s = d["summary"]
    rows = [r for r in d["results"] if r.get("cell_tp") is not None
            and r.get("source") != "EvalMuse"]
    if s.get("n_localization_gt") == len(rows):
        print(f"{path}: already clean (n_localization_gt={len(rows)}), skipped")
        return
    old = {k: s.get(k) for k in ("cell_f1_micro", "avg_grid_iou_problem_only")}
    tp = sum(r["cell_tp"] for r in rows)
    fp = sum(r["cell_fp"] for r in rows)
    fn = sum(r["cell_fn"] for r in rows)
    p, rc, f1 = prf(tp, fp, fn)
    problem = [r for r in rows if r["cell_tp"] + r["cell_fn"] > 0]
    iou = [r["grid_iou"] for r in rows if r.get("grid_iou") is not None]
    s["n_localization_gt"] = len(rows)
    s["avg_grid_iou"] = round(sum(iou) / len(iou), 4) if iou else None
    s["avg_grid_iou_problem_only"] = round(
        sum(r["grid_iou"] for r in problem) / len(problem), 4) if problem else None
    s["cell_precision_micro"] = round(p, 4)
    s["cell_recall_micro"] = round(rc, 4)
    s["cell_f1_micro"] = round(f1, 4)
    if problem and all("cell_f1" in r for r in problem):
        s["cell_f1_macro_problem"] = round(
            sum(r.get("cell_f1", 0.0) for r in problem) / len(problem), 4)
    s["n_problem_gt"] = len(problem)
    s["clean_correct"] = sum(
        1 for r in rows if r["cell_tp"] + r["cell_fp"] + r["cell_fn"] == 0)
    s["resummarized"] = ("2026-07-23: localization metrics exclude score-only "
                         "EvalMuse rows (no spatial GT)")
    json.dump(d, open(path, "w"), indent=2)
    print(f"{path}: F1 {old['cell_f1_micro']} -> {s['cell_f1_micro']}, "
          f"IoU_p {old['avg_grid_iou_problem_only']} -> "
          f"{s['avg_grid_iou_problem_only']}  (n_loc={len(rows)})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        fix(p)

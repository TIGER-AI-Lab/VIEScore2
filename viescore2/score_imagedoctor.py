#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score ImageDoctor's stored outputs against our protocol (no GPU).

Producer for eval_results_external/imagedoctor_analysis.json:

1. RichHF-test localization: union of its two 16x16 sigmoid maps, binarized
   at the threshold that maximizes ITS micro-F1 on the first 100 benchmark
   rows (tuned in its favor), scored on the remaining 840 against the same
   GT grids our models use; our RL numbers recomputed on the identical 840.
2. eval_suite scoring: tie-aware Spearman per source and per ImagenWorld task
   for its overall and semantic-alignment axes.

    python viescore2/score_imagedoctor.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "eval_results_external"
OUT = EXT / "imagedoctor_analysis.json"
N_TUNE = 100


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def loc_richhf():
    idoc = json.load(open(EXT / "imagedoctor_richhf_test.json"))["results"]
    ours = {r["id"]: r for r in
            json.load(open(EXT / "viescore2_rl_richhf_test.json"))["results"]}
    rows = []
    for r in idoc:
        o = ours.get(r["id"])
        if not o or not r.get("heatmap_16"):
            continue
        a = np.array(r["heatmap_16"]["artifact"], np.float32)
        m = np.array(r["heatmap_16"]["misalignment"], np.float32)
        rows.append((np.maximum(a, m), np.array(o["gt_grid"], np.uint8), o))
    tune, rep = rows[:N_TUNE], rows[N_TUNE:]

    def micro(sel, t):
        tp = fp = fn = 0
        for hm, gt, _ in sel:
            pred = hm > t
            gb = gt.astype(bool)
            tp += int((gb & pred).sum()); fp += int((~gb & pred).sum())
            fn += int((gb & ~pred).sum())
        return prf(tp, fp, fn)

    best_t, best_f1 = None, -1.0
    for t in np.arange(0.01, 0.91, 0.01):
        f1 = micro(tune, t)[2]
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), f1

    p, r, f1 = micro(rep, best_t)
    ious = []
    for hm, gt, _ in rep:
        gb = gt.astype(bool)
        if not gb.any():
            continue
        pred = hm > best_t
        u = int((gb | pred).sum())
        ious.append(int((gb & pred).sum()) / u if u else 1.0)

    tp = sum(o["cell_tp"] for _, _, o in rep)
    fp = sum(o["cell_fp"] for _, _, o in rep)
    fn = sum(o["cell_fn"] for _, _, o in rep)
    op, orc, of1 = prf(tp, fp, fn)
    oious = [o["grid_iou"] for _, gt, o in rep if gt.astype(bool).any()]

    # paired bootstrap of ours - imagedoctor on the identical 840 rows
    per = []
    for hm, gt, o in rep:
        pred = hm > best_t
        gb = gt.astype(bool)
        itp = int((gb & pred).sum()); ifp = int((~gb & pred).sum())
        ifn = int((gb & ~pred).sum())
        iiou = None
        if gb.any():
            u = int((gb | pred).sum())
            iiou = (itp / u if u else 1.0)
        per.append((o["cell_tp"], o["cell_fp"], o["cell_fn"],
                    o["grid_iou"] if gb.any() else None, itp, ifp, ifn, iiou))
    rng = np.random.default_rng(0)
    d_f1, d_iou = [], []
    for _ in range(2000):
        k = rng.integers(0, len(per), len(per))
        sel = [per[i] for i in k]
        of = prf(sum(x[0] for x in sel), sum(x[1] for x in sel),
                 sum(x[2] for x in sel))[2]
        idf = prf(sum(x[4] for x in sel), sum(x[5] for x in sel),
                  sum(x[6] for x in sel))[2]
        d_f1.append(of - idf)
        oi = [x[3] for x in sel if x[3] is not None]
        ii = [x[7] for x in sel if x[7] is not None]
        d_iou.append(float(np.mean(oi)) - float(np.mean(ii)))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4),
                    round(float(np.percentile(v, 97.5)), 4)]
    return {
        "n_tune": len(tune), "n_report": len(rep),
        "threshold_tuned_for_imagedoctor": best_t,
        "tune_F1": round(best_f1, 4),
        "imagedoctor": {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
                        "IoU_p": round(float(np.mean(ious)), 4)},
        "viescore2_rl_same_840": {
            "P": round(op, 4), "R": round(orc, 4), "F1": round(of1, 4),
            "IoU_p": round(float(np.mean(oious)), 4)},
        "ours_minus_imagedoctor_paired": {
            "F1": round(of1 - f1, 4), "F1_CI": ci(d_f1),
            "IoU_p": round(float(np.mean(oious)) - float(np.mean(ious)), 4),
            "IoU_p_CI": ci(d_iou)},
    }


def scores():
    idoc = json.load(open(EXT / "imagedoctor.json"))["results"]
    ours = {r["id"]: r for r in json.load(open(
        ROOT / "eval_results/viescore2_rl.json"))["results"]
        if r.get("pred_score") is not None}
    def task_of(r):
        m = re.match(r"([A-Z]+)_", r["id"] or "")
        return m.group(1) if m else r["source"]
    groups = {}
    for r in idoc:
        if r.get("gt") is None or not r.get("scores") or r["id"] not in ours:
            continue
        groups.setdefault(task_of(r), []).append(r)
        groups.setdefault("ALL_scored", []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        gt = [r["gt"] for r in rs]
        if len(rs) < 10 or len(set(gt)) < 3:
            continue
        out[g] = {"n": len(rs)}
        for axis in ("overall", "semantic_alignment"):
            pred = [r["scores"][axis] for r in rs]
            out[g][axis] = round(float(spearmanr(gt, pred).statistic), 4)
        op = [ours[r["id"]]["pred_score"] for r in rs]
        og = [ours[r["id"]]["gt_score"] for r in rs]
        out[g]["viescore2_rl"] = round(float(spearmanr(og, op).statistic), 4)
    return out


def main():
    res = {"localization_richhf_test": loc_richhf(),
           "scores_eval_suite_per_group": scores()}
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired input-context intervention (tab:planned-context).

Same frozen policy, same 250 conditioned rows (TIE/SRIG/SRIE/MRIG/MRIE),
same prompt TEXT: chi_K sees the conditioning images, chi_0 does not. The
paired difference therefore isolates access to I_i alone. Per-task rho
(tie-averaged Spearman of predicted vs human overall score) and micro cell-F1,
with paired bootstrap 95% CIs (2,000 resamples, seed 42) over rows.

  python viescore2/analyze_chi0.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHI_K = ROOT / "eval_results/viescore2_rl.json"
CHI_0 = ROOT / "eval_results/viescore2_rl_chi0_conditioned250.json"
OUT = ROOT / "eval_results/chi0_intervention_analysis.json"
TASKS = ["TIE", "SRIG", "SRIE", "MRIG", "MRIE"]


def rank_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = rank_avg(a), rank_avg(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    k_all = {r["id"]: r for r in json.load(open(CHI_K))["results"]}
    z_rows = json.load(open(CHI_0))["results"]

    per_task = {t: [] for t in TASKS}
    for z in z_rows:
        k = k_all.get(z["id"])
        if k is None:
            continue
        task = z["source"].replace("ImagenWorld-", "")
        if task not in per_task:
            continue
        gt = z.get("gt_score")
        per_task[task].append({
            "gt": None if gt in (None, "") else float(gt),
            "pk": None if k.get("pred_score") in (None, "") else float(k["pred_score"]),
            "pz": None if z.get("pred_score") in (None, "") else float(z["pred_score"]),
            "k": (k["cell_tp"], k["cell_fp"], k["cell_fn"]),
            "z": (z["cell_tp"], z["cell_fp"], z["cell_fn"]),
        })

    rng = np.random.default_rng(42)

    def block(rows):
        sc = [r for r in rows if r["gt"] is not None
              and r["pk"] is not None and r["pz"] is not None]
        rk = spearman([r["pk"] for r in sc], [r["gt"] for r in sc]) if sc else float("nan")
        rz = spearman([r["pz"] for r in sc], [r["gt"] for r in sc]) if sc else float("nan")
        fk = f1(*[sum(r["k"][i] for r in rows) for i in range(3)])
        fz = f1(*[sum(r["z"][i] for r in rows) for i in range(3)])
        d_rho, d_f1 = [], []
        for _ in range(2000):
            if sc:
                idx = rng.integers(0, len(sc), len(sc))
                s = [sc[i] for i in idx]
                d_rho.append(spearman([r["pk"] for r in s], [r["gt"] for r in s])
                             - spearman([r["pz"] for r in s], [r["gt"] for r in s]))
            idx2 = rng.integers(0, len(rows), len(rows))
            g = [rows[i] for i in idx2]
            d_f1.append(f1(*[sum(r["k"][i] for r in g) for i in range(3)])
                        - f1(*[sum(r["z"][i] for r in g) for i in range(3)]))
        ci = lambda v: ([round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)] if v else None)
        return {"n": len(rows), "n_scored": len(sc),
                "rho_chiK": round(rk, 4), "rho_chi0": round(rz, 4),
                "d_rho": round(rk - rz, 4), "d_rho_CI": ci(d_rho),
                "F1_chiK": round(fk, 4), "F1_chi0": round(fz, 4),
                "d_F1": round(fk - fz, 4), "d_F1_CI": ci(d_f1)}

    out = {t: block(per_task[t]) for t in TASKS if per_task[t]}
    out["E_Kgt0"] = block([r for t in TASKS for r in per_task[t]])
    out["_protocol"] = ("identical policy/decoding/prompt text on the same 250 rows; "
                        "chi_0 withholds conditioning images only "
                        "(run_eval --no-cond-images). Paired bootstrap "
                        "2,000 resamples, seed 42.")
    json.dump(out, open(OUT, "w"), indent=2)
    hdr = f"{'task':<8}{'n':>4}{'rhoK':>8}{'rho0':>8}{'drho':>9}{'F1K':>8}{'F10':>8}{'dF1':>9}"
    print(hdr)
    for t in TASKS + ["E_Kgt0"]:
        if t not in out:
            continue
        b = out[t]
        print(f"{t:<8}{b['n']:>4}{b['rho_chiK']:>8.3f}{b['rho_chi0']:>8.3f}"
              f"{b['d_rho']:>9.3f}{b['F1_chiK']:>8.3f}{b['F1_chi0']:>8.3f}{b['d_F1']:>9.3f}")
    print(f"\nAggregate d_rho CI {out['E_Kgt0']['d_rho_CI']}, "
          f"d_F1 CI {out['E_Kgt0']['d_F1_CI']} -> {OUT}")


if __name__ == "__main__":
    main()

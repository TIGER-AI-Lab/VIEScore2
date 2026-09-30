#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Seven-evaluator rows for a new spatial benchmark (HAD / SynthScars).

Basis: rows whose GT grid is nonzero (same convention as the published
AbHuman-1,407 block, where it reproduced the tex rows cell-for-cell).
Metrics: micro cell P/R/F1 over the basis + problem-mean grid IoU.

Row sources:
  run_eval results (zeroshot / sft / rl / segformer): per-row cell counts.
  RAHF / ImageDoctor heatmap dumps: union=max(artifact,misalign) at 16x16,
    RichHF-frozen thresholds (0.06 / 0.03) — fully zero-shot.
  PAL4VST specialist grid dumps: native 0.5 mask threshold.

  python viescore2/analyze_external.py --bench had
  python viescore2/analyze_external.py --bench synthscars
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viescore2"))
from build_sft_data import gt_visual_to_grid  # noqa: E402

EXT = ROOT / "eval_results_external"
THRESH = {"rahf": 0.06, "imagedoctor": 0.03}
FILES = {  # bench -> row name -> (kind, filename)
    "had": {
        "zeroshot": ("runeval", "viescore2_zeroshot_had.json"),
        "sft": ("runeval", "viescore2_sft_had.json"),
        "rl": ("runeval", "viescore2_rl_had.json"),
        "segformer": ("runeval", "segformer_had.json"),
        "rahf": ("heatmap", "rahf_had.json"),
        "imagedoctor": ("heatmap", "imagedoctor_had.json"),
        "specialist": ("grid", "specialist_had.json"),
    },
    "synthscars": {
        "zeroshot": ("runeval", "viescore2_zeroshot_synthscars.json"),
        "sft": ("runeval", "viescore2_sft_synthscars.json"),
        "rl": ("runeval", "viescore2_rl_synthscars.json"),
        "segformer": ("runeval", "segformer_synthscars.json"),
        "rahf": ("heatmap", "rahf_synthscars.json"),
        "imagedoctor": ("heatmap", "imagedoctor_synthscars.json"),
        "specialist": ("grid", "specialist_synthscars.json"),
    },
    "sdg30k": {
        "zeroshot": ("runeval", "viescore2_zeroshot_sdg30k.json"),
        "sft": ("runeval", "viescore2_sft_sdg30k.json"),
        "rl": ("runeval", "viescore2_rl_sdg30k.json"),
        "segformer": ("runeval", "segformer_sdg30k.json"),
        "rahf": ("heatmap", "rahf_sdg30k.json"),
        "imagedoctor": ("heatmap", "imagedoctor_sdg30k.json"),
        "specialist": ("grid", "specialist_sdg30k.json"),
    },
}
EVAL_DATA = {"had": "eval_samples/had_eval.jsonl",
             "synthscars": "eval_samples/synthscars_eval.jsonl",
             "sdg30k": "eval_samples/sdg30k_eval.jsonl"}


def prf_iou(cells):
    tp = sum(c[0] for c in cells); fp = sum(c[1] for c in cells)
    fn = sum(c[2] for c in cells)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ious = [c[3] for c in cells]
    return {"P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
            "IoU_p": round(float(np.mean(ious)), 3) if ious else None,
            "n": len(cells)}


def grid_cells(pred, gt):
    pred = pred.astype(bool); g = gt.astype(bool)
    tp = int((pred & g).sum()); fp = int((pred & ~g).sum())
    fn = int((~pred & g).sum()); un = int((pred | g).sum())
    return (tp, fp, fn, tp / un if un else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(FILES))
    args = ap.parse_args()

    gt_by_id, problem = {}, set()
    for line in open(ROOT / EVAL_DATA[args.bench]):
        s = json.loads(line)
        g = gt_visual_to_grid(s["response"]["visual_reason"])
        gt_by_id[s["meta"]["id"]] = g
        if g.sum() > 0:
            problem.add(s["meta"]["id"])
    print(f"{args.bench}: basis = {len(problem)} nonzero-GT of {len(gt_by_id)}")

    out = {"basis_n": len(problem), "protocol": (
        "micro cell P/R/F1 + problem-mean grid IoU on the nonzero-GT basis; "
        "RAHF/ImageDoctor zero-shot with RichHF-frozen thresholds "
        "(0.06/0.03), union=max(artifact,misalign); specialist native 0.5")}
    for name, (kind, fn) in FILES[args.bench].items():
        p = EXT / fn
        if not p.exists():
            out[name] = "pending"
            print(f"{name}: pending ({fn})")
            continue
        res = json.load(open(p))["results"]
        cells = []
        if kind == "runeval":
            for r in res:
                if r["id"] in problem and "cell_tp" in r:
                    gsum = int(np.asarray(r["gt_grid"], np.uint8).sum()) \
                        if "gt_grid" in r else 1
                    cells.append((r["cell_tp"], r["cell_fp"], r["cell_fn"],
                                  r.get("grid_iou", 0.0)))
        elif kind == "heatmap":
            thr = THRESH[name]
            for r in res:
                if r["id"] not in problem:
                    continue
                hm = r.get("heatmap_16") or {}
                a = hm.get("artifact"); m = hm.get("misalignment")
                a = np.asarray(a, np.float32) if a is not None else np.zeros((16, 16), np.float32)
                m = np.asarray(m, np.float32) if m is not None else np.zeros((16, 16), np.float32)
                cells.append(grid_cells((np.maximum(a, m) > thr).astype(np.uint8),
                                        gt_by_id[r["id"]]))
        else:  # grid dumps (specialist)
            for r in res:
                if r["id"] in problem:
                    cells.append(grid_cells(np.asarray(r["grid16"], np.uint8),
                                            gt_by_id[r["id"]]))
        out[name] = prf_iou(cells)
        print(name, out[name])

    json.dump(out, open(EXT / f"extern_{args.bench}_analysis.json", "w"), indent=1)
    print("wrote", EXT / f"extern_{args.bench}_analysis.json")


if __name__ == "__main__":
    main()

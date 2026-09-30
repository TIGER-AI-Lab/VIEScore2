#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""New tab:spatialcompare rows: API evaluators (GPT-5.6-terra, Gemini-3-Flash,
Gemini-3.1-Pro), HADM, LEGION, SDG — on each benchmark's PUBLISHED basis.

Bases (identical to the published rows):
  richhf     rows after the 100-row threshold-tuning slice (RichHF-840);
             micro P/R/F1 over all basis rows, IoU_p over nonzero-GT rows
  abhuman / had / synthscars / sdg30k   nonzero-GT basis (extern convention)
  pal4vst    all 1,405 rows micro; IoU_p over the 898 nonzero-GT rows

Prediction sources:
  runeval   run_eval outputs (API rows): per-row cell counts;
            parse failures (grid_parsed=False) already scored as empty grids
  grid      16x16 grid dumps (LEGION); missing outputs scored empty
  hadm      raw L+G detections; score >= 0.5 (release demo default), union,
            pixel xyxy -> [x,y,w,h]/1000 -> the SAME bbox rasterizer as GT
  sdg       native-contract defect dumps; Qwen [x0,y0,x1,y1]/1000 boxes ->
            same rasterizer; defects=None (nothing parseable) scored empty

  python viescore2/analyze_evaluators.py --bench richhf
  python viescore2/analyze_evaluators.py --bench all
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
N_TUNE = 100
HADM_THR = 0.5

# bench -> (reference runeval file for GT/basis, basis mode, api suffix)
BENCH = {
    "richhf":     ("viescore2_rl_richhf_test.json", "tune_slice", "richhf_test"),
    "abhuman":    ("viescore2_rl_abhuman.json",     "nonzero",    "abhuman_eval"),
    "had":        ("viescore2_rl_had.json",         "nonzero",    "had_eval"),
    "synthscars": ("viescore2_rl_synthscars.json",  "nonzero",    "synthscars_eval"),
    "pal4vst":    ("viescore2_rl_pal4vst.json",     "all",        "pal4vst_test"),
    "sdg30k":     (None,                                "nonzero",    "sdg30k_eval"),
}
EVAL_JSONL = {"sdg30k": "eval_samples/sdg30k_eval.jsonl"}

ROWS = {  # row name -> (kind, filename pattern over the api suffix)
    "gpt56terra":   ("runeval", "gpt56terra_{b}.json"),
    "gpt56sol":     ("runeval", "gpt56sol_{b}.json"),
    "gemini3flash": ("runeval", "gemini3flash_{b}.json"),
    "gemini31pro":  ("runeval", "gemini31pro_{b}.json"),
    "opus55":       ("runeval", "opus55_{b}.json"),
    "legion":       ("grid",    "legion_{b}.json"),
    "hadm":         ("hadm",    "hadm_raw_{short}.json"),
    "sdg":          ("sdg",     "sdg_raw_{b}.json"),
}
# contract applicability: HADM is a human-artifact detector -> human suites only
HADM_BENCHES = {"had", "abhuman"}


def boxes_to_grid(xywh_1000):
    return gt_visual_to_grid({"type": "bbox", "data": xywh_1000}) \
        if xywh_1000 else np.zeros((16, 16), np.uint8)


def load_ref(bench):
    """-> (ordered ids, gt grids by id, basis id set)"""
    ref_file, mode, _ = BENCH[bench]
    gt, order = {}, []
    if ref_file is not None:
        for r in json.load(open(EXT / ref_file))["results"]:
            order.append(r["id"])
            gt[r["id"]] = np.asarray(r["gt_grid"], np.uint8)
    else:
        for line in open(ROOT / EVAL_JSONL[bench]):
            s = json.loads(line)
            order.append(s["meta"]["id"])
            gt[s["meta"]["id"]] = gt_visual_to_grid(s["response"]["visual_reason"])
    if mode == "tune_slice":
        basis = set(order[N_TUNE:])
    elif mode == "nonzero":
        basis = {i for i in order if gt[i].sum() > 0}
    else:
        basis = set(order)
    return order, gt, basis


def cell_stats(pred, gt):
    pred = pred.astype(bool); g = gt.astype(bool)
    tp = int((pred & g).sum()); fp = int((pred & ~g).sum())
    fn = int((~pred & g).sum()); un = int((pred | g).sum())
    iou = (tp / un if un else 1.0) if g.sum() else None
    return tp, fp, fn, iou


def prf_iou(cells):
    tp = sum(c[0] for c in cells); fp = sum(c[1] for c in cells)
    fn = sum(c[2] for c in cells)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ious = [c[3] for c in cells if c[3] is not None]
    return {"P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
            "IoU_p": round(float(np.mean(ious)), 3) if ious else None,
            "n": len(cells)}


def analyze(bench):
    order, gt, basis = load_ref(bench)
    _, mode, suffix = BENCH[bench]
    short = {"abhuman_eval": "abhuman", "had_eval": "had"}.get(suffix, suffix)
    out = {"bench": bench, "basis_n": len(basis), "basis_mode": mode}
    print(f"── {bench}: basis {len(basis)} ({mode}) of {len(order)}")

    for name, (kind, pat) in ROWS.items():
        if name == "hadm" and bench not in HADM_BENCHES:
            continue
        p = EXT / pat.format(b=suffix, short=short)
        if not p.exists():
            out[name] = "pending"
            print(f"  {name}: pending ({p.name})")
            continue
        res = json.load(open(p))["results"]
        cells, notes = [], {}
        if kind == "runeval":
            nofail = sum(0 if r.get("grid_parsed", True) else 1
                         for r in res if r["id"] in basis)
            for r in res:
                if r["id"] in basis:
                    g = np.asarray(r["gt_grid"], np.uint8)
                    cells.append((r["cell_tp"], r["cell_fp"], r["cell_fn"],
                                  r["grid_iou"] if g.sum() else None))
            notes["grid_parse_failures"] = nofail
        elif kind == "grid":
            pred = {r["id"]: np.asarray(r["grid16"], np.uint8) for r in res}
            miss = 0
            for i in basis:
                g = pred.get(i)
                if g is None:
                    g, miss = np.zeros((16, 16), np.uint8), miss + 1
                cells.append(cell_stats(g, gt[i]))
            notes["missing_outputs"] = miss
        elif kind == "hadm":
            by_id = {r["id"]: r for r in res}
            miss = 0
            for i in basis:
                r = by_id.get(i)
                if r is None:
                    cells.append(cell_stats(np.zeros((16, 16), np.uint8), gt[i]))
                    miss += 1
                    continue
                bx = []
                for head in ("local", "global"):
                    h = r[head]
                    W, H = h["wh"]
                    for b, s in zip(h["boxes"], h["scores"]):
                        if s >= HADM_THR:
                            x0, y0, x1, y1 = b
                            bx.append([x0 / W * 1000, y0 / H * 1000,
                                       (x1 - x0) / W * 1000, (y1 - y0) / H * 1000])
                cells.append(cell_stats(boxes_to_grid(bx), gt[i]))
            notes["missing_outputs"] = miss
            notes["score_thr"] = HADM_THR
        elif kind == "sdg":
            by_id = {r["id"]: r for r in res}
            nofail = trunc = miss = 0
            for i in basis:
                r = by_id.get(i)
                if r is None:
                    cells.append(cell_stats(np.zeros((16, 16), np.uint8), gt[i]))
                    miss += 1
                    continue
                ds = r["defects"]
                if ds is None:
                    nofail += 1
                    ds = []
                if r.get("truncated"):
                    trunc += 1
                # degenerate boxes (x1<=x0 or y1<=y0) contribute IoU 0 in the
                # official eval; equivalently they rasterize to nothing — drop
                bx = [[d["box_2d"][0], d["box_2d"][1],
                       d["box_2d"][2] - d["box_2d"][0],
                       d["box_2d"][3] - d["box_2d"][1]] for d in ds
                      if d["box_2d"][2] > d["box_2d"][0]
                      and d["box_2d"][3] > d["box_2d"][1]]
                cells.append(cell_stats(boxes_to_grid(bx), gt[i]))
            notes.update(parse_failures=nofail, truncated=trunc,
                         missing_outputs=miss)
        out[name] = {**prf_iou(cells), **notes}
        print(f"  {name}: {out[name]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="all",
                    choices=list(BENCH) + ["all"])
    args = ap.parse_args()
    benches = list(BENCH) if args.bench == "all" else [args.bench]
    out = {b: analyze(b) for b in benches}
    dst = EXT / "table4_new_rows.json"
    if dst.exists():
        prev = json.load(open(dst))
        prev.update(out)
        out = prev
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()

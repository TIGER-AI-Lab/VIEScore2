#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Symmetry fill for tab:spatialcompare — the three external-method cells.

Computes, on each benchmark's PUBLISHED basis (verified below by
reproducing the existing \\score-RL row from its own result file):

  specialist_richhf   PAL4VST Swin-L/UperNet on RichHF-840 (native 0.5
                      mask threshold, no text input — structurally blind to
                      misalignment GT; disclosed).
  rahf_pal4vst        RAHF heatmaps on PAL4VST-1,405, RichHF-frozen 0.06.
  imagedoctor_pal4vst ImageDoctor heatmaps on PAL4VST-1,405, frozen 0.03.

Bases: RichHF-840 = rows after the 100-row threshold-tuning slice
(same convention as the RAHF/ImageDoctor analyses);
PAL4VST-1,405 = all rows, micro P/R/F1 + problem-mean grid IoU (898
nonzero-GT rows), exactly the summary fields the published row uses.

  python viescore2/analyze_specialists.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "eval_results_external"
THRESH = {"rahf": 0.06, "imagedoctor": 0.03}
PUBLISHED = {
    "richhf840_ours": (0.423, 0.518, 0.466, 0.299),
    "pal1405_ours": (0.354, 0.578, 0.439, 0.335),
}
N_TUNE = 100  # RichHF tuning slice size (rows BEFORE it are the tune set)


def prf_iou(cells):
    """cells: [(tp, fp, fn, gt_union_iou_or_None)]; IoU averaged over rows
    with nonzero GT only (grid problem-mean)."""
    tp = sum(c[0] for c in cells); fp = sum(c[1] for c in cells)
    fn = sum(c[2] for c in cells)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ious = [c[3] for c in cells if c[3] is not None]
    return (round(p, 3), round(r, 3), round(f1, 3),
            round(float(np.mean(ious)), 3) if ious else None)


def cells_from_own_rows(rows):
    out = []
    for r in rows:
        gt = np.asarray(r["gt_grid"], np.uint8)
        out.append((r["cell_tp"], r["cell_fp"], r["cell_fn"],
                    r["grid_iou"] if gt.sum() > 0 else None))
    return out


def cells_from_grids(pred_by_id, rows):
    out = []
    for r in rows:
        pred = pred_by_id.get(r["id"])
        if pred is None:
            continue
        gt = np.asarray(r["gt_grid"], np.uint8).astype(bool)
        pred = pred.astype(bool)
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum()); un = int((pred | gt).sum())
        out.append((tp, fp, fn, (tp / un if un else 1.0) if gt.sum() else None))
    return out


def heatmap_grids(path, thr):
    res = json.load(open(path))["results"]
    out = {}
    for r in res:
        hm = r.get("heatmap_16") or {}
        a = hm.get("artifact"); m = hm.get("misalignment")
        a = np.asarray(a, np.float32) if a is not None else np.zeros((16, 16), np.float32)
        m = np.asarray(m, np.float32) if m is not None else np.zeros((16, 16), np.float32)
        out[r["id"]] = (np.maximum(a, m) > thr).astype(np.uint8)
    return out


def main():
    out = {}

    # ── RichHF-840 basis ────────────────────────────────────────────────────
    rh = json.load(open(EXT / "viescore2_rl_richhf_test.json"))["results"]
    rh840 = rh[N_TUNE:]
    ours = prf_iou(cells_from_own_rows(rh840))
    ok_rh = ours == PUBLISHED["richhf840_ours"]
    print(f"verify RichHF-840 ours: computed {ours} published "
          f"{PUBLISHED['richhf840_ours']} -> {'OK' if ok_rh else 'MISMATCH'}")

    sp_path = EXT / "specialist_richhf.json"
    if sp_path.exists():
        pred = {r["id"]: np.asarray(r["grid16"], np.uint8)
                for r in json.load(open(sp_path))["results"]}
        cells = cells_from_grids(pred, rh840)
        out["specialist_richhf"] = dict(
            zip(("P", "R", "F1", "IoU_p"), prf_iou(cells)),
            n=len(cells), basis="RichHF-840", basis_verified=ok_rh,
            note="native 0.5 mask threshold; no text input — cannot see "
                 "misalignment GT by construction")
        print("specialist_richhf:", out["specialist_richhf"])
    else:
        print("specialist_richhf: pending")

    # ── PAL4VST-1,405 basis ─────────────────────────────────────────────────
    pal = json.load(open(EXT / "viescore2_rl_pal4vst.json"))["results"]
    ours_pal = prf_iou(cells_from_own_rows(pal))
    # per-row grid_iou is stored at 4dp, so the recomputed problem-mean can
    # drift half a thousandth vs the summary's unrounded path — allow 0.001
    ok_pal = all(abs(a - b) <= 0.001 + 1e-9
                 for a, b in zip(ours_pal, PUBLISHED["pal1405_ours"]))
    print(f"verify PAL-1405 ours: computed {ours_pal} published "
          f"{PUBLISHED['pal1405_ours']} -> {'OK' if ok_pal else 'MISMATCH'}")

    for name in ("rahf", "imagedoctor"):
        p = EXT / f"{name}_pal4vst.json"
        if not p.exists():
            print(f"{name}_pal4vst: pending")
            continue
        pred = heatmap_grids(p, THRESH[name])
        cells = cells_from_grids(pred, pal)
        out[f"{name}_pal4vst"] = dict(
            zip(("P", "R", "F1", "IoU_p"), prf_iou(cells)),
            n=len(cells), basis="PAL4VST-1405", basis_verified=ok_pal,
            threshold=THRESH[name],
            note="zero-shot, RichHF-frozen threshold; PAL4VST images carry "
                 "no prompt — identical placeholder text as our rows")
        print(f"{name}_pal4vst:", out[f"{name}_pal4vst"])

    json.dump(out, open(EXT / "spatial_fill_analysis.json", "w"), indent=1)
    print("wrote", EXT / "spatial_fill_analysis.json")


if __name__ == "__main__":
    main()

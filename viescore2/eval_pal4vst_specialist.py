#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PAL4VST unified specialist on OUR PAL4VST test split — with per-sample rows.

Producer for eval_results_external/pal4vst_specialist.json (the
earlier run kept only a summary, so no paired CI against our models was
possible). Scores the released Swin-L/UperNet torchscript at 512^2 pixel
resolution AND rasterized to the 16x16 grid; each sample row carries id,
cell counts and grid IoU keyed identically to our result files.

    python viescore2/eval_pal4vst_specialist.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "$DATA_ROOT/pal4vst_repo")
from utils import prepare_input  # noqa: E402

DEVICE = 0
TS = ("$DATA_ROOT/pal4vst_repo/deployment/pal4vst/"
      "swin-large_upernet_unified_512x512/end2end.pt")
EVAL = "eval_samples/pal4vst_test.jsonl"
OUT = "eval_results_external/pal4vst_specialist.json"
ROOT = Path(__file__).resolve().parent.parent


def to_grid(mask, gs=16, thr=0.5):
    small = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (gs, gs), Image.LANCZOS), np.float32) / 255.0
    return (small > thr).astype(np.uint8)


def main():
    model = torch.jit.load(TS).to(DEVICE)
    model.eval()
    samples = [json.loads(l) for l in open(ROOT / EVAL) if l.strip()]

    rows = []
    pix_i = pix_u = 0
    iou_pix = []
    for k, s in enumerate(samples):
        path = s["image"] if s["image"].startswith("/") else str(ROOT / s["image"])
        try:
            img = Image.open(path).convert("RGB").resize((512, 512))
        except Exception:
            continue
        with torch.no_grad():
            pal = model(prepare_input(np.array(img), DEVICE)).cpu().data.numpy()[0][0]
        pred = (pal > 0.5).astype(np.uint8)

        vr = s["response"].get("visual_reason") or {}
        data = vr.get("data")
        gt512 = np.zeros((512, 512), np.uint8)
        if isinstance(data, str):
            gp = data if Path(data).exists() else str(ROOT / data)
            try:
                garr = np.array(Image.open(gp).convert("L").resize((512, 512)),
                                np.float32)
                gt512 = (garr > (127 if garr.max() > 1 else 0.5)).astype(np.uint8)
            except Exception:
                pass

        inter = int((pred & gt512).sum()); union = int((pred | gt512).sum())
        pix_i += inter; pix_u += union
        if gt512.any():
            iou_pix.append(inter / union if union else 1.0)

        pg, gg = to_grid(pred), to_grid(gt512)
        tp = int((pg & gg).sum()); fp = int((pg & ~gg).sum()); fn = int((~pg & gg).sum())
        gi = tp; gu = int((pg | gg).sum())
        rows.append({
            "id": (s.get("meta") or {}).get("id"),
            "source": (s.get("meta") or {}).get("source", "PAL4VST-test"),
            "image": s["image"],
            "cell_tp": tp, "cell_fp": fp, "cell_fn": fn,
            "grid_iou": round(gi / gu, 4) if gu else 1.0,
            "pixel_iou": round(inter / union, 4) if union else 1.0,
            "gt_any": bool(gt512.any()),
        })
        if (k + 1) % 300 == 0:
            print(f"{k+1}/{len(samples)}", flush=True)

    tp = sum(r["cell_tp"] for r in rows)
    fp = sum(r["cell_fp"] for r in rows)
    fn = sum(r["cell_fn"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    prob = [r for r in rows if r["cell_tp"] + r["cell_fn"] > 0]
    out = {
        "summary": {
            "model": "PAL4VST specialist (Swin-L UperNet, released ckpt)",
            "n": len(rows),
            "pixel_IoU_micro": round(pix_i / pix_u, 4) if pix_u else None,
            "pixel_IoU_problem_mean": round(float(np.mean(iou_pix)), 4),
            "grid16_P": round(p, 4), "grid16_R": round(rc, 4),
            "grid16_F1": round(f1, 4),
            "grid16_IoU_problem_mean": round(float(np.mean(
                [r["grid_iou"] for r in prob])), 4),
            "n_problem": len(prob),
        },
        "results": rows,
    }
    json.dump(out, open(ROOT / OUT, "w"), indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PAL4VST unified specialist, zero-shot on the AbHuman localization slice.

Same released Swin-L/UperNet torchscript as eval_pal4vst_specialist.py, run
on eval_samples/abhuman_eval.jsonl (bbox GT). Grid-16 GT is rasterized
through the SAME gt_visual_to_grid path run_eval uses for our
model, so cell metrics are directly comparable; pixel IoU uses the boxes
drawn at 512^2.

    python viescore2/eval_specialist_abhuman.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viescore2"))
sys.path.insert(0, "$DATA_ROOT/pal4vst_repo")
from utils import prepare_input  # noqa: E402
from build_sft_data import gt_visual_to_grid  # noqa: E402

DEVICE = 0
TS = ("$DATA_ROOT/pal4vst_repo/deployment/pal4vst/"
      "swin-large_upernet_unified_512x512/end2end.pt")
EVAL = "eval_samples/abhuman_eval.jsonl"
OUT = "eval_results_external/specialist_abhuman.json"


def to_grid(mask, gs=16, thr=0.5):
    small = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (gs, gs), Image.LANCZOS), np.float32) / 255.0
    return (small > thr).astype(np.uint8)


def boxes_to_mask512(boxes):
    canvas = Image.new("L", (512, 512), 0)
    d = ImageDraw.Draw(canvas)
    for x, y, w, h in boxes:
        d.rectangle([x / 1000 * 512, y / 1000 * 512,
                     (x + w) / 1000 * 512, (y + h) / 1000 * 512], fill=255)
    return (np.array(canvas) > 127).astype(np.uint8)


def main():
    model = torch.jit.load(TS).to(DEVICE)
    model.eval()
    samples = [json.loads(l) for l in open(ROOT / EVAL) if l.strip()]

    rows, iou_pix = [], []
    pix_i = pix_u = 0
    for k, s in enumerate(samples):
        img = Image.open(s["image"]).convert("RGB").resize((512, 512))
        with torch.no_grad():
            pal = model(prepare_input(np.array(img), DEVICE)).cpu().data.numpy()[0][0]
        pred = (pal > 0.5).astype(np.uint8)

        vr = s["response"]["visual_reason"]
        gt512 = boxes_to_mask512(vr["data"])
        inter = int((pred & gt512).sum()); union = int((pred | gt512).sum())
        pix_i += inter; pix_u += union
        if gt512.any():
            iou_pix.append(inter / union if union else 1.0)

        pg = to_grid(pred)
        gg = gt_visual_to_grid(vr)
        tp = int((pg & gg).sum()); fp = int((pg & ~gg).sum()); fn = int((~pg & gg).sum())
        gu = int((pg | gg).sum())
        rows.append({
            "id": s["meta"]["id"], "source": "AbHuman-val", "image": s["image"],
            "cell_tp": tp, "cell_fp": fp, "cell_fn": fn,
            "grid_iou": round(tp / gu, 4) if gu else 1.0,
            "pixel_iou": round(inter / union, 4) if union else 1.0,
        })
        if (k + 1) % 200 == 0:
            print(f"{k+1}/{len(samples)}", flush=True)

    tp = sum(r["cell_tp"] for r in rows)
    fp = sum(r["cell_fp"] for r in rows)
    fn = sum(r["cell_fn"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "summary": {
            "model": "PAL4VST specialist (Swin-L UperNet, released ckpt), zero-shot",
            "n": len(rows),
            "pixel_IoU_micro": round(pix_i / pix_u, 4) if pix_u else None,
            "pixel_IoU_problem_mean": round(float(np.mean(iou_pix)), 4),
            "grid16_P": round(p, 4), "grid16_R": round(rc, 4),
            "grid16_F1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
            "grid16_IoU_problem_mean": round(float(np.mean(
                [r["grid_iou"] for r in rows])), 4),
        },
        "results": rows,
    }
    json.dump(out, open(ROOT / OUT, "w"), indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()

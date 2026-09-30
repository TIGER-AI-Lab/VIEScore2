#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PAL4VST unified specialist, zero-shot on RichHF-test (tab:spatialcompare
symmetry fill). Same released Swin-L/UperNet torchscript and native 0.5 mask
threshold as eval_specialist_abhuman.py; dumps the 16x16 grid per row —
analyze_specialists.py scores it on the RichHF-840 basis against the
gt_grids embedded in our result file. The specialist takes no text input,
so it cannot recover RichHF's misalignment GT by construction (disclosed).

If $DATA_ROOT/pal4vst_repo is missing (volatile box), restore with:
    git clone --depth 1 https://github.com/owenzlz/PAL4VST $DATA_ROOT/pal4vst_repo
    python -m gdown 1bGjEKquWa4cZJ6v52NoRfViwKK15U7vd -O $DATA_ROOT/pal4vst_repo/\
deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt

    python viescore2/eval_specialist_richhf.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "$DATA_ROOT/pal4vst_repo")
from utils import prepare_input  # noqa: E402

DEVICE = 0
TS = ("$DATA_ROOT/pal4vst_repo/deployment/pal4vst/"
      "swin-large_upernet_unified_512x512/end2end.pt")
import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("--input", default=str(ROOT / "eval_samples/richhf_test.jsonl"))
_ap.add_argument("--output", default=str(
    ROOT / "eval_results_external/specialist_richhf.json"))
_args = _ap.parse_args()
EVAL = Path(_args.input)
OUT = Path(_args.output)


def to_grid(mask, gs=16, thr=0.5):
    small = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (gs, gs), Image.LANCZOS), np.float32) / 255.0
    return (small > thr).astype(np.uint8)


def main():
    model = torch.jit.load(TS).to(DEVICE)
    model.eval()
    samples = [json.loads(l) for l in open(EVAL) if l.strip()]
    rows = []
    for k, s in enumerate(samples):
        img = Image.open(ROOT / s["image"]).convert("RGB").resize((512, 512))
        with torch.no_grad():
            pal = model(prepare_input(np.array(img), DEVICE)).cpu().data.numpy()[0][0]
        rows.append({"id": s["meta"]["id"],
                     "grid16": to_grid((pal > 0.5).astype(np.uint8)).tolist()})
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(samples)}", flush=True)
    json.dump({"summary": {"model": "PAL4VST specialist (released ckpt), "
                                    "zero-shot on RichHF-test",
                           "n": len(rows), "mask_threshold": 0.5},
               "results": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

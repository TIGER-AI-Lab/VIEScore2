#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HADM (arXiv 2411.13842) box dumps for our spatial benchmarks.

Runs the released HADM-L (local body-part artifacts) and HADM-G (global)
EVA-02-L ViTDet detectors on one of our eval jsonls and dumps per-row raw
detections (xyxy boxes in pixel space, scores, classes) for both heads.
Loading mirrors tools/lazyconfig_train_net.py --inference exactly
(instantiate -> EMA build -> DetectionCheckpointer with EMA state -> apply).
Grid conversion and thresholds live in the analyzer, not here.

Run inside the dedicated env:
  source $DATA_ROOT/hadm_venv/bin/activate
  PYTHONPATH=$DATA_ROOT/hadm_repo python viescore2/eval_hadm_dump.py \
      --input eval_samples/had_eval.jsonl \
      --output eval_results_external/hadm_raw_had.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("$DATA_ROOT/hadm_repo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

CONFIGS = {
    "local": REPO / "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py",
    "global": REPO / "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py",
}
CKPTS = {
    "local": REPO / "pretrained_models/HADM-L_0249999.pth",
    "global": REPO / "pretrained_models/HADM-G_0249999.pth",
}


def build(kind):
    import torch  # noqa: F401
    from detectron2.config import LazyConfig, instantiate
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.engine.defaults import create_ddp_model
    import detectron2.modeling.ema as ema

    cfg = LazyConfig.load(str(CONFIGS[kind]))
    cfg.train.init_checkpoint = str(CKPTS[kind])
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(
        cfg.train.init_checkpoint)
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        ema.apply_model_ema(model)
    from detectron2.engine.defaults import DefaultInferencer  # hadm fork
    return DefaultInferencer(cfg, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent

    rows = []
    for line in open(args.input):
        s = json.loads(line)
        p = Path(s["image"])
        rows.append({"id": s["meta"]["id"],
                     "path": str(p if p.is_absolute() else root / p)})
    print(f"{len(rows)} rows from {args.input}", flush=True)

    out = {r["id"]: {} for r in rows}
    for kind in ("local", "global"):
        print(f"loading HADM-{kind[0].upper()} ...", flush=True)
        inferencer = build(kind)
        for k, r in enumerate(rows):
            im = Image.open(r["path"]).convert("RGB")
            arr = np.asarray(im)[:, :, ::-1]  # BGR, matching read_image(format="BGR")
            inst = inferencer(arr)["instances"].to("cpu")
            out[r["id"]][kind] = {
                "wh": [im.width, im.height],
                "boxes": [[round(v, 2) for v in b] for b in inst.pred_boxes.tensor.tolist()],
                "scores": [round(v, 4) for v in inst.scores.tolist()],
                "classes": inst.pred_classes.tolist(),
            }
            if (k + 1) % 100 == 0:
                print(f"  {kind} {k+1}/{len(rows)}", flush=True)
        del inferencer
        import torch
        torch.cuda.empty_cache()

    json.dump({"summary": {"model": "HADM-L+G (released 0249999 ckpts, EVA-02-L "
                                    "ViTDet), raw detections; thresholds applied "
                                    "at analysis",
                           "n": len(rows), "input": args.input},
               "results": [{"id": i, **out[i]} for i in out]},
              open(args.output, "w"))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dev-split evaluation for protocol/hyperparameter selection (P2/P0-6).

Evaluates a checkpoint on the FROZEN dev split
(manifests/dev_eval.jsonl: 580 val-partition rows, never trained on,
disjoint from eval_suite by the leakage gate). Each row carries its own
training-time user prompt, so evaluation is prompt-matched by construction.

This is where hyperparameter sweeps are selected;
eval_suite remains frozen for the single pre-registered configuration.

Usage:
  python viescore2/dev_eval.py --checkpoint <ckpt> --output <json>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import (  # noqa: E402
    _load_model_and_processor, resize_to_train_res, _resize_to,
    parse_grid_from_response, SYSTEM_PROMPT, MAX_PIXELS_REF,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="manifests/dev_eval.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from PIL import Image

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    model, processor = _load_model_and_processor(args.checkpoint)

    per = {}   # source -> [tp, fp, fn]
    clean_stats = {}  # source -> [n_clean, n_flagged]
    ious = []
    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        imgs = []
        for j, p in enumerate(r["image_paths"]):
            im = Image.open(p).convert("RGB")
            im = resize_to_train_res(im) if j == len(r["image_paths"]) - 1 \
                else _resize_to(im, MAX_PIXELS_REF)
            imgs.append(im)
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": r["user_text"]})
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=imgs, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        resp = processor.batch_decode(
            [out[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0]

        gt = np.asarray(r["gt_grid"], np.uint8).astype(bool)
        pred = parse_grid_from_response(resp)
        pred_b = pred.astype(bool) if pred is not None else np.zeros_like(gt)
        tp = int((gt & pred_b).sum()); fp = int((~gt & pred_b).sum())
        fn = int((gt & ~pred_b).sum())
        d = per.setdefault(r["source"], [0, 0, 0])
        d[0] += tp; d[1] += fp; d[2] += fn
        if gt.any():
            u = int((gt | pred_b).sum())
            ious.append(tp / u if u else 1.0)
        if r["clean"]:
            c = clean_stats.setdefault(r["source"], [0, 0])
            c[0] += 1; c[1] += int(pred_b.any())
        results.append({"uid": r["uid"], "source": r["source"],
                        "clean": r["clean"], "tp": tp, "fp": fp, "fn": fn,
                        "parsed": pred is not None})
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    def prf(t, f, n):
        p = t / (t + f) if t + f else 0.0
        rc = t / (t + n) if t + n else 0.0
        return p, rc, 2 * p * rc / (p + rc) if p + rc else 0.0

    TP = sum(v[0] for v in per.values())
    FP = sum(v[1] for v in per.values())
    FN = sum(v[2] for v in per.values())
    P, R, F1 = prf(TP, FP, FN)
    summary = {
        "checkpoint": args.checkpoint, "manifest": args.manifest,
        "n": len(results),
        "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4),
        "iou_p": round(float(np.mean(ious)), 4) if ious else None,
        "clean_false_positive_rate": {
            k: {"n_clean": v[0], "flagged": v[1]}
            for k, v in sorted(clean_stats.items())},
        "per_source_f1": {k: round(prf(*v)[2], 4) for k, v in sorted(per.items())},
        "parse_rate": round(100.0 * sum(x["parsed"] for x in results) / len(results), 1),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results},
              open(args.output, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

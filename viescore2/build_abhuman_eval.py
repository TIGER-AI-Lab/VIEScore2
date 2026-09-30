#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AbHuman (HumanRefiner) val -> eval jsonl: localization OOD study.

AbHuman (Fang et al., ECCV 2024, MIT): 56K synthesized human images with
147K human-anatomy anomaly bboxes in 18 classes. The val split here is
11,266 images (1024^2, YOLO-format normalized boxes), ALL problem images
(no clean negatives). Images, prompts, annotators, and the anomaly
taxonomy are disjoint from every training source -- a zero-shot
localization transfer test in a domain (human anatomy) none of our
sources labels, and one where scalar T2I scorers have no output at all.

Boxes are converted to the pool's bbox convention (x,y,w,h on a
1000x1000 canvas) so run_eval rasterizes GT through the exact
same gt_visual_to_grid path as ImagenWorld bbox annotations. Rows are
grid-only (no score request), prompt-matched to the PAL4VST-style
no-prompt template; the parenthetical uses the Stable-Diffusion variant
already present in eval_suite.

Caveat for analysis: only anatomy anomalies are annotated -- flags on
unannotated non-anatomy artifacts count as false positives, so recall
and IoU_p are the more meaningful numbers.

Usage:
  python viescore2/build_abhuman_eval.py \
      [--val-dir data/abhuman/val] [--n 2000] [--seed 42] \
      [--output eval_samples/abhuman_eval.jsonl]
"""
import argparse
import json
import random
from pathlib import Path

INSTRUCTION = ("(no text prompt available; image synthesized by a "
               "Stable Diffusion text-to-image model)")


def yolo_to_bbox1000(line):
    _cls, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    return [round((cx - w / 2) * 1000, 2), round((cy - h / 2) * 1000, 2),
            round(w * 1000, 2), round(h * 1000, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="data/abhuman/val")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="eval_samples/abhuman_eval.jsonl")
    args = ap.parse_args()

    val = Path(args.val_dir)
    labels = sorted((val / "labels").glob("*.txt"))
    rng = random.Random(args.seed)
    rng.shuffle(labels)

    out, n_boxes = [], 0
    for lf in labels:
        if len(out) >= args.n:
            break
        img = val / "images" / (lf.stem + ".jpg")
        if not img.exists():
            continue
        boxes = [yolo_to_bbox1000(l) for l in lf.read_text().splitlines()
                 if l.strip()]
        if not boxes:
            continue
        n_boxes += len(boxes)
        out.append({
            "image": str(img),
            "instruction": INSTRUCTION,
            "response": {"score": None, "text_reason": "",
                         "visual_reason": {"type": "bbox", "data": boxes}},
            "meta": {"source": "AbHuman-val", "id": f"abhuman_{lf.stem}",
                     "split": "external_ood",
                     "orig": {"n_boxes": len(boxes)}},
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{len(out)} rows ({n_boxes} boxes, "
          f"{n_boxes/len(out):.2f}/img) -> {args.output}")


if __name__ == "__main__":
    main()

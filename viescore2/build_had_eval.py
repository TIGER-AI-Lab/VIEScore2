#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HAD (Human Artifact Dataset, arXiv 2411.13842) val_ALL → our eval jsonl.

Stratified seeded slice: 350 rows per generator (dalle2/dalle3/mj/sdxl) =
1,400 total, keeping each image's REAL generation prompt as the
instruction (info embedded per-annotation json). GT bboxes [x1,y1,x2,y2]
in pixel space are converted to the abhuman convention:
visual_reason {type: bbox, data: [[x,y,w,h]]} in 0-1000 units.

  python viescore2/build_had_eval.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAD = Path("$DATA_ROOT/had/human_artifact_dataset")
OUT = ROOT / "eval_samples/had_eval.jsonl"
PER_GEN = 350
SEED = 42


def main():
    by_gen = defaultdict(list)
    for f in sorted((HAD / "annotations/val_ALL").glob("*.json")):
        by_gen[f.name.split("_")[0]].append(f)
    rng = random.Random(SEED)
    rows = []
    for gen, files in sorted(by_gen.items()):
        rng.shuffle(files)
        picked = files[:PER_GEN]
        print(f"{gen}: {len(files)} available, {len(picked)} picked")
        for f in picked:
            d = json.load(open(f))
            im = d["image"]
            W, H = im["width"], im["height"]
            img_path = HAD / "images/val_ALL" / im["file_name"]
            if not img_path.exists():
                continue
            boxes = []
            for a in d.get("annotation", []):
                x1, y1, x2, y2 = a["bbox"]
                boxes.append([round(x1 * 1000 / W, 2), round(y1 * 1000 / H, 2),
                              round((x2 - x1) * 1000 / W, 2),
                              round((y2 - y1) * 1000 / H, 2)])
            rows.append({
                "image": str(img_path),
                "instruction": im.get("prompt") or
                               "(no text prompt available)",
                "response": {"score": None, "text_reason": "",
                             "visual_reason": {"type": "bbox", "data": boxes}},
                "meta": {"source": "HAD-val", "id": f"had_{f.stem}",
                         "split": "external_ood",
                         "orig": {"generator": gen, "n_boxes": len(boxes),
                                  "levels": [a.get("level") for a in
                                             d.get("annotation", [])]}},
            })
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    nz = sum(1 for r in rows if r["response"]["visual_reason"]["data"])
    print(f"wrote {len(rows)} rows ({nz} with nonzero GT) -> {OUT}")


if __name__ == "__main__":
    main()

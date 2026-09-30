#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SynthScars test (LEGION, arXiv 2503.15264) → our eval jsonl.

1,000 fully synthetic test images with expert polygon annotations of
artifact regions (pixel coordinates in each image's own space). Polygons
are rasterised to a union mask PNG per image; rows carry
visual_reason {type: mask, data: <png>} so gt_visual_to_grid handles them
exactly like PAL4VST masks. The annotation captions CONTAIN artifact
explanations, so the instruction is a placeholder — never the caption
(GT leak).

  python viescore2/build_synthscars_eval.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SS = Path("$DATA_ROOT/synthscars/SynthScars/test")
MASKS = Path("$DATA_ROOT/synthscars/masks_test")
OUT = ROOT / "eval_samples/synthscars_eval.jsonl"
PLACEHOLDER = ("(no text prompt available; fully synthetic image from the "
               "SynthScars benchmark)")


def main():
    MASKS.mkdir(parents=True, exist_ok=True)
    items = json.load(open(SS / "annotations/test.json"))
    rows, n_nz = [], 0
    for item in items:
        for key, v in item.items():
            img_path = SS / "images" / v["img_file_name"]
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                W, H = im.size
            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)
            n_poly = 0
            for ref in v.get("refs", []):
                for poly in (ref.get("segmentation") or []):
                    pts = list(zip(poly[0::2], poly[1::2]))
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=255)
                        n_poly += 1
            mask_path = MASKS / (Path(v["img_file_name"]).stem + ".png")
            mask.save(mask_path)
            if n_poly:
                n_nz += 1
            rows.append({
                "image": str(img_path),
                "instruction": PLACEHOLDER,
                "response": {"score": None, "text_reason": "",
                             "visual_reason": {"type": "mask",
                                               "data": str(mask_path)}},
                "meta": {"source": "SynthScars-test",
                         "id": f"synthscars_{key}_{Path(v['img_file_name']).stem[:12]}",
                         "split": "external_ood",
                         "orig": {"n_polygons": n_poly, "wh": [W, H]}},
            })
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({n_nz} with nonzero GT) -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SDG-30K test split (arXiv 2606.06113) → our eval jsonl (sixth benchmark).

All 1,154 official test rows. GT boxes come in the dataset's Gemini
convention [y0,x0,y1,x1] normalized to 0-1000 and are converted to our
bbox convention [x,y,w,h] in 0-1000 (the AbHuman/HAD path). The union of
artifact and misalignment boxes forms the flat localization target;
per-channel boxes are preserved in meta.orig. Captions are the real T2I
prompts (prompt-disjoint from their train split by construction).

  python viescore2/build_sdg30k_eval.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDG = Path("$DATA_ROOT/sdg/sdg30k")
OUT = ROOT / "eval_samples/sdg30k_eval.jsonl"


def conv(bs):
    out = []
    for b in bs or []:
        y0, x0, y1, x1 = [float(v) for v in b["box_2d"][:4]]
        out.append([round(x0, 2), round(y0, 2),
                    round(x1 - x0, 2), round(y1 - y0, 2)])
    return out


def main():
    rows, miss = [], 0
    for i, line in enumerate(open(SDG / "annotations/test.jsonl")):
        s = json.loads(line)
        img = SDG / s["filepath"]
        if not img.exists():
            miss += 1
            continue
        art = conv(s.get("artifact_bboxes"))
        mis = conv(s.get("misalignment_bboxes"))
        rows.append({
            "image": str(img),
            "instruction": s["caption"],
            "response": {"score": None, "text_reason": "",
                         "visual_reason": {"type": "bbox", "data": art + mis}},
            "meta": {"source": "SDG30K-test",
                     # filepath stems repeat across generators -> id needs both
                     "id": f"sdg30k_{s['generator']}_{Path(s['filepath']).stem}",
                     "split": "external_ood",
                     "orig": {"artifact_boxes": art, "misalign_boxes": mis,
                              "n_art": len(art), "n_mis": len(mis)}},
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nz = sum(1 for r in rows if r["response"]["visual_reason"]["data"])
    print(f"wrote {len(rows)} rows ({nz} nonzero GT, {miss} missing images) -> {OUT}")


if __name__ == "__main__":
    main()

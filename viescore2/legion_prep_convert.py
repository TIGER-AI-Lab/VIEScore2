#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prep/convert glue around LEGION's official infer.py.

prep:    build a flat symlink dir of a benchmark's images (row-indexed
         names, original extensions) + an id-mapping json, so the official
         script can consume the benchmark unchanged.
convert: map infer.py outputs (<out>/<filename>/mask.png, union of predicted
         artifact segments at native size) back to row ids and rasterize to
         16x16 grids (cell = mask mean > 0.5, the PAL-specialist convention).

  python viescore2/legion_prep_convert.py prep --input eval_samples/X.jsonl \
      --workdir $DATA_ROOT/legion/runs/X
  python viescore2/legion_prep_convert.py convert --workdir ... \
      --output eval_results_external/legion_X.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def prep(args):
    wd = Path(args.workdir)
    (wd / "images").mkdir(parents=True, exist_ok=True)
    mapping = {}
    n = skipped = 0
    for i, line in enumerate(open(args.input)):
        s = json.loads(line)
        p = Path(s["image"])
        if not p.is_absolute():
            p = ROOT / p
        name = f"r{i:05d}{p.suffix.lower()}"
        dst = wd / "images" / name
        mapping[name] = s["meta"]["id"]
        # resume: an image whose mask already exists is pruned from the infer
        # set (mapping keeps it, so convert still picks its output up)
        if args.skip_done and (wd / "out" / name / "mask.png").exists():
            if dst.is_symlink():
                dst.unlink()
            skipped += 1
            continue
        if not dst.exists():
            os.symlink(p.resolve(), dst)
        n += 1
    json.dump(mapping, open(wd / "mapping.json", "w"))
    print(f"prepped {n} images ({skipped} already done, pruned) -> {wd}/images")


def convert(args):
    wd = Path(args.workdir)
    mapping = json.load(open(wd / "mapping.json"))
    rows, missing = [], 0
    for name, rid in mapping.items():
        mp = wd / "out" / name / "mask.png"
        if not mp.exists():
            missing += 1
            continue
        m = np.array(Image.open(mp).convert("L"), np.float32) / 255.0
        H, W = m.shape
        grid = np.zeros((16, 16), np.uint8)
        for r in range(16):
            for c in range(16):
                cell = m[r * H // 16:(r + 1) * H // 16,
                         c * W // 16:(c + 1) * W // 16]
                grid[r, c] = 1 if cell.mean() > 0.5 else 0
        rows.append({"id": rid, "grid16": grid.tolist()})
    json.dump({"summary": {"model": "LEGION (khr0516/legion_LE intermediate "
                                    "release; GLaMM+SAM), native artifact-"
                                    "analysis instruction, union mask > 0.5 "
                                    "per cell", "n": len(rows),
                           "missing_outputs": missing},
               "results": rows}, open(args.output, "w"))
    print(f"wrote {args.output}: {len(rows)} rows, {missing} missing")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--input", required=True)
    a.add_argument("--workdir", required=True)
    a.add_argument("--skip-done", action="store_true",
                   help="prune images that already have out/<name>/mask.png")
    b = sub.add_parser("convert")
    b.add_argument("--workdir", required=True)
    b.add_argument("--output", required=True)
    args = ap.parse_args()
    (prep if args.cmd == "prep" else convert)(args)


if __name__ == "__main__":
    main()

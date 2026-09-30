#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert PAL4VST (ICCV'23, github.com/owenzlz/PAL4VST) into chat
samples and (optionally) merge them with an existing train set.

PAL4VST = ~14K generated images across ~15 synthesis tasks with human
per-pixel artifact labels (labels/<split>/<stem>.png, pixel values {0,1};
an all-zero label = artifact-free image → a natural REAL-AIGC clean negative).

Differences vs the main builder (build_sft_data.py):
  • no human quality score → target has NO "score:" line and the user prompt
    does NOT ask for one (with_score=False) — never ask for what the label
    cannot supervise;
  • no text prompt → instruction is a task-aware description derived from the
    filename prefix;
  • label pixels are {0,1}, NOT {0,255}: value range is auto-normalised before
    the LANCZOS→0.5-threshold cell rule (same rule as the main pipeline).

Usage:
    python viescore2/build_pal4vst_eval.py \\
        --pal4vst-root data/pal4vst/unified \\
        --output-dir $DATA_ROOT/viescore2_data_full \\
        --empty-target-frac 0.1 --seed 42
    # then merge with the main training data:
    python viescore2/build_pal4vst_eval.py --merge \\
        --base-dir $DATA_ROOT/viescore2_data \\
        --output-dir $DATA_ROOT/viescore2_data_full
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_data import (  # noqa: E402
    GRID_SIZE,
    MAX_PIXELS,
    SYSTEM_PROMPT,
    build_user_prompt,
    grid_to_target,
    resize_and_save,
)

log = logging.getLogger("pal4vst")

# filename prefix → human-readable generator description (for the instruction)
TASK_DESC = {
    "sd": "a Stable Diffusion text-to-image model",
    "dalle": "a DALL-E text-to-image model",
    "ldm_sample": "a latent diffusion model",
    "inpaint": "an image inpainting model",
    "mask2image": "a mask-to-image translation model",
    "edge2image": "an edge-to-image translation model",
    "cvton": "a virtual try-on model",
    "stylegan": "a StyleGAN generator",
    "proggan": "a ProgressiveGAN generator",
    "anyresgan": "a super-resolution model",
    "realesrgan": "a super-resolution model",
    "portrait_shadow": "a portrait shadow-removal model",
    "afhqcat": "a StyleGAN animal-face generator",
    "ffhq": "a StyleGAN face generator",
}


def task_desc_for(stem: str) -> str:
    s = stem.lower()
    for prefix, desc in TASK_DESC.items():
        if s.startswith(prefix):
            return desc
    return "a generative image model"


def label_to_grid(label_path: Path, grid_size: int = GRID_SIZE) -> Optional[np.ndarray]:
    """{0,1}-pixel artifact label → 16×16 cell grid (LANCZOS + 0.5 rule)."""
    from PIL import Image

    try:
        arr = np.array(Image.open(label_path).convert("L"), dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    m = arr.max()
    if m > 1.0:            # {0,255}-style masks, just in case
        arr = arr / 255.0
    # else already in [0,1] ({0,1} labels) — the /255 bug this guards against
    # would silently zero every grid.
    small = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (grid_size, grid_size), Image.LANCZOS
    )
    return (np.array(small, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)


def make_pal4vst_sample(image_ref: str, instruction: str, grid: np.ndarray) -> dict:
    """Single-image, no-score chat sample (prompt does not ask for a score)."""
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "path": image_ref},
                {"type": "text",
                 "text": build_user_prompt(instruction, 0, with_score=False)},
            ]},
            {"role": "assistant",
             "content": [{"type": "text", "text": grid_to_target(grid)}]},
        ]
    }


def build(args) -> None:
    root = Path(args.pal4vst_root)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images_pal4vst"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    problem, clean = [], []
    skipped = 0
    for split in ("train", "val"):        # test split held out on purpose
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            lbl_path = lbl_dir / (img_path.stem + ".png")
            if not lbl_path.exists():
                skipped += 1
                continue
            grid = label_to_grid(lbl_path)
            if grid is None:
                skipped += 1
                continue
            dst = images_dir / f"pal4vst_{split}_{img_path.name}"
            if not resize_and_save(img_path, dst, max_pixels=MAX_PIXELS):
                skipped += 1
                continue
            instruction = (
                f"(no text prompt available; image synthesized by {task_desc_for(img_path.stem)})"
            )
            sample = make_pal4vst_sample(str(dst.resolve()), instruction, grid)
            (problem if int(grid.sum()) > 0 else clean).append(sample)

    # keep clean images at ~empty_frac of the final mix (same rule as main builder)
    n_clean_keep = int(len(problem) * args.empty_target_frac / (1 - args.empty_target_frac))
    rng.shuffle(clean)
    kept_clean = clean[:n_clean_keep]
    samples = problem + kept_clean
    rng.shuffle(samples)

    out = out_dir / "pal4vst.json"
    out.write_text(json.dumps(samples, ensure_ascii=False))
    log.info(f"PAL4VST: {len(problem)} problem + {len(kept_clean)}/{len(clean)} clean "
             f"→ {len(samples)} samples (skipped {skipped}) → {out}")


def merge(args) -> None:
    """base train/val + pal4vst.json → merged train/val in output-dir."""
    out_dir = Path(args.output_dir)
    base = Path(args.base_dir)
    rng = random.Random(args.seed)

    pal = json.loads((out_dir / "pal4vst.json").read_text())
    n_val = max(1, int(len(pal) * args.val_ratio))
    pal_val, pal_train = pal[:n_val], pal[n_val:]

    for split, extra in (("train", pal_train), ("val", pal_val)):
        base_samples = json.loads((base / f"{split}.json").read_text())
        merged = base_samples + extra
        rng.shuffle(merged)
        (out_dir / f"{split}.json").write_text(json.dumps(merged, ensure_ascii=False))
        log.info(f"{split}: {len(base_samples)} base + {len(extra)} pal4vst "
                 f"= {len(merged)} → {out_dir / (split + '.json')}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pal4vst-root", default="data/pal4vst/unified")
    p.add_argument("--output-dir", default="$DATA_ROOT/viescore2_data_full")
    p.add_argument("--base-dir", default="$DATA_ROOT/viescore2_data",
                   help="(--merge) existing training data to merge with")
    p.add_argument("--empty-target-frac", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge", action="store_true",
                   help="merge output-dir/pal4vst.json with base-dir train/val")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.merge:
        merge(args)
    else:
        build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

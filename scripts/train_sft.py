#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full SFT Fine-Tuning Script for Qwen3-VL (Axolotl Backend)

Trains ALL model weights for Qwen3-VL using Axolotl with DeepSpeed ZeRO-3
for multimodal image quality evaluation with visual grounding.

Prerequisites:
    pip install axolotl[flash-attn,deepspeed]
    pip install pillow numpy pyyaml

Usage:
    # Convert SFT data to Axolotl format
    python scripts/train_sft.py prepare \
        --input sft_samples/sft_unified.jsonl \
        --output axolotl_qwen3_train_data

    # Launch full SFT
    python scripts/train_sft.py train \
        --data-dir axolotl_qwen3_train_data

    # Test the fine-tuned model
    python scripts/train_sft.py test \
        --checkpoint checkpoints/qwen3_8B_full \
        --image path/to/image.png \
        --instruction "Evaluate this image"

References:
    - https://docs.axolotl.ai/docs/multimodal.html
    - https://github.com/axolotl-ai-cloud/axolotl
    - https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Axolotl chat_template for Qwen3-VL models
AXOLOTL_CHAT_TEMPLATE = "qwen2_vl"


def _default_checkpoint_dir(model_name: str) -> Path:
    """Derive a checkpoint directory from the model name.

    E.g. "Qwen/Qwen3-VL-8B-Instruct" -> "checkpoints/qwen3_8B_full"
    """
    base = model_name.split("/")[-1]          # "Qwen3-VL-8B-Instruct"
    base = base.replace("-Instruct", "")       # "Qwen3-VL-8B"
    base = base.replace("-VL-", "_")           # "Qwen3_8B"
    base = base.replace("-", "_")              # "Qwen3_8B"
    base = re.sub(r"(\d+)B", lambda m: m.group(1) + "B", base)
    parts = base.split("_")
    parts[0] = parts[0].lower()
    base = "_".join(parts)
    return Path(f"checkpoints/{base}_full")


VISUAL_FORMATS = ("svg", "polygon")

SYSTEM_PROMPT_SVG = """You are an expert image quality evaluator. Your task is to assess generated images based on:
1. Instruction compliance: How well the image follows the given text prompt
2. Image quality: Technical quality including artifacts, coherence, and aesthetics

For each evaluation, provide:
- A quality score from 0 to 10 (integer)
- A detailed text explanation of any issues found
- Problem regions as an SVG with <path> elements using M (moveto), L (lineto), Z (closepath) commands, coordinates in 0-1000 scale (viewBox="0 0 1000 1000")

Be specific and objective in your assessments."""

SYSTEM_PROMPT_POLYGON = """You are an expert image quality evaluator. Your task is to assess generated images based on:
1. Instruction compliance: How well the image follows the given text prompt
2. Image quality: Technical quality including artifacts, coherence, and aesthetics

For each evaluation, provide:
- A quality score from 0 to 10 (integer)
- A detailed text explanation of any issues found
- Problem regions as polygons with vertex coordinates [[x1, y1], [x2, y2], ...] (normalized 0-1000 scale)

Be specific and objective in your assessments."""


def get_system_prompt(visual_format: str = "svg") -> str:
    """Return the system prompt for the given visual format."""
    if visual_format == "svg":
        return SYSTEM_PROMPT_SVG
    return SYSTEM_PROMPT_POLYGON


# ---------------------------------------------------------------------------
# Eval-set filtering
# ---------------------------------------------------------------------------

def load_eval_image_set(
    eval_set_path: Path,
    data_root: Path = Path("data"),
) -> set:
    """Load eval sample image paths into a set of resolved absolute paths."""
    eval_images: set = set()
    eval_set_path = Path(eval_set_path)
    if not eval_set_path.exists():
        logger.warning(
            f"Eval set not found at {eval_set_path}, no eval filtering applied"
        )
        return eval_images

    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                img = sample.get("image", "")
                if img:
                    p = Path(img)
                    if p.is_absolute():
                        eval_images.add(str(p.resolve()))
                    else:
                        if p.exists():
                            eval_images.add(str(p.resolve()))
                        eval_images.add(str((data_root / img).resolve()))

    logger.info(
        f"Loaded {len(eval_images)} eval image paths to exclude from training"
    )
    return eval_images


# ---------------------------------------------------------------------------
# Utility functions (polygon extraction, heatmap rendering, prompt formatting)
# ---------------------------------------------------------------------------

def extract_polygons_from_mask(
    mask_path: str,
    threshold: float = 0.5,
    max_polygons: int = 5,
    max_vertices: int = 15,
) -> List[List[List[int]]]:
    """
    Extract simplified polygon contours from a binary mask image.

    Uses OpenCV findContours + Douglas-Peucker simplification.
    Falls back to a rectangular polygon (4 vertices) via scipy/numpy
    if OpenCV is unavailable.

    Args:
        mask_path: Path to mask image (white = problem areas)
        threshold: Threshold for binary mask (0-1)
        max_polygons: Maximum number of polygons to return
        max_vertices: Maximum vertices per polygon

    Returns:
        List of polygons, each polygon is a list of [x, y] points
        normalized to 0-1000 scale.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return []

    try:
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask) / 255.0
        height, width = mask_array.shape
        binary = (mask_array > threshold).astype(np.uint8)

        if binary.sum() == 0:
            return []

        min_area = width * height * 0.001

        # Preferred: OpenCV contour extraction + Douglas-Peucker simplification
        try:
            import cv2

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            polygons = []
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue

                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                attempts = 0
                while len(approx) > max_vertices and attempts < 5:
                    epsilon *= 1.5
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    attempts += 1

                if len(approx) < 3:
                    continue

                poly = [
                    [int(pt[0][0] / width * 1000), int(pt[0][1] / height * 1000)]
                    for pt in approx
                ]
                polygons.append(poly)

                if len(polygons) >= max_polygons:
                    break

            return polygons
        except ImportError:
            pass

        # Fallback: scipy/numpy — extract bounding boxes as rectangular polygons
        try:
            from scipy import ndimage

            labeled, num_features = ndimage.label(binary)
            polygons = []
            for comp_id in range(1, num_features + 1):
                ys, xs = np.where(labeled == comp_id)
                if len(ys) < min_area:
                    continue
                min_x = int(xs.min() / width * 1000)
                min_y = int(ys.min() / height * 1000)
                max_x = int((xs.max() + 1) / width * 1000)
                max_y = int((ys.max() + 1) / height * 1000)
                polygons.append([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ])
                if len(polygons) >= max_polygons:
                    break
            return polygons
        except ImportError:
            pass

        # Last resort: numpy-only row/col projection → rectangular polygons
        row_any = np.any(binary, axis=1)
        row_diff = np.diff(row_any.astype(np.int8))
        band_starts = np.where(row_diff == 1)[0] + 1
        band_ends = np.where(row_diff == -1)[0] + 1
        if row_any[0]:
            band_starts = np.concatenate([[0], band_starts])
        if row_any[-1]:
            band_ends = np.concatenate([band_ends, [height]])

        polygons = []
        for y_start, y_end in zip(band_starts, band_ends):
            band_slice = binary[y_start:y_end, :]
            col_mask = np.any(band_slice, axis=0)
            xs = np.where(col_mask)[0]
            if len(xs) == 0:
                continue
            min_x, max_x = int(xs[0]), int(xs[-1]) + 1
            if (max_x - min_x) * (y_end - y_start) < min_area:
                continue
            x1 = int(min_x / width * 1000)
            y1 = int(y_start / height * 1000)
            x2 = int(max_x / width * 1000)
            y2 = int(y_end / height * 1000)
            polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            if len(polygons) >= max_polygons:
                break

        return polygons

    except Exception as e:
        logger.warning(f"Failed to extract polygons from {mask_path}: {e}")
        return []


def render_heatmap_from_polygons(
    polygons: List[List[List[int]]],
    output_path: str,
    image_size: tuple = (512, 512),
    base_image_path: Optional[str] = None,
) -> str:
    """Render polygons as a heatmap overlay."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np
    except ImportError:
        raise ImportError("PIL not installed")

    width, height = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)

    for poly in polygons:
        pixel_pts = [
            (int(pt[0] / 1000 * width), int(pt[1] / 1000 * height))
            for pt in poly
        ]
        poly_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(poly_img)
        if len(pixel_pts) >= 3:
            draw.polygon(pixel_pts, fill=255)
        poly_array = np.array(poly_img, dtype=np.float32) / 255.0
        heatmap += poly_array

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = (heatmap * 255).astype(np.uint8)
    rgba[:, :, 3] = (heatmap * 180).astype(np.uint8)

    heatmap_img = Image.fromarray(rgba, mode="RGBA")
    heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=10))

    if base_image_path and Path(base_image_path).exists():
        base = Image.open(base_image_path).convert("RGBA")
        base = base.resize(image_size)
        result = Image.alpha_composite(base, heatmap_img)
    else:
        background = Image.new("RGBA", image_size, (128, 128, 128, 255))
        result = Image.alpha_composite(background, heatmap_img)

    result.save(output_path)
    return output_path


def polygons_to_svg(polygons: List[List[List[int]]]) -> str:
    """Convert polygon coordinate lists to a minimal SVG string."""
    if not polygons:
        return '<svg viewBox="0 0 1000 1000"></svg>'
    paths = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = [[max(0, min(1000, x)), max(0, min(1000, y))] for x, y in poly]
        d = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            d += f" L {pt[0]} {pt[1]}"
        d += " Z"
        paths.append(f'  <path d="{d}"/>')
    inner = "\n".join(paths)
    if inner:
        return f'<svg viewBox="0 0 1000 1000">\n{inner}\n</svg>'
    return '<svg viewBox="0 0 1000 1000"></svg>'


def svg_to_polygons(svg_str: str) -> List[List[List[int]]]:
    """Parse SVG path elements back to polygon coordinate lists."""
    if not svg_str or "<path" not in svg_str:
        return []
    polygons = []
    for match in re.finditer(r'd="([^"]*)"', svg_str):
        d = match.group(1)
        points = []
        for cmd_match in re.finditer(r'[ML]\s*(\d+)\s+(\d+)', d):
            x, y = int(cmd_match.group(1)), int(cmd_match.group(2))
            points.append([x, y])
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def parse_svg_from_response(response_text: str) -> str:
    """Extract SVG markup from model response text."""
    m = re.search(r'```svg\s*\n(.*?)```', response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'(<svg[^>]*>.*?</svg>)', response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def parse_polygons_from_response(response_text: str) -> List[List[List[int]]]:
    """Parse polygon point lists from model response text.

    Supports SVG format (preferred) and legacy polygon JSON format.
    """
    svg_str = parse_svg_from_response(response_text)
    if svg_str:
        polygons = svg_to_polygons(svg_str)
        if polygons:
            return polygons

    patterns = [
        r"\*\*Problem Regions \(polygon\):\*\*\s*(\[\[\[[\d,\s\[\]]+\]\]\])",
        r"polygon.*?:\s*(\[\[\[[\d,\s\[\]]+\]\]\])",
        r"regions.*?:\s*(\[\[\[[\d,\s\[\]]+\]\]\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                import ast
                polygons = ast.literal_eval(match.group(1))
                if isinstance(polygons, list) and all(
                    isinstance(p, list) and len(p) >= 3 and all(
                        isinstance(pt, list) and len(pt) == 2 for pt in p
                    )
                    for p in polygons
                ):
                    return polygons
            except Exception:
                continue
    return []


def build_model_response(
    response: Dict[str, Any],
    polygons: List[List[List[int]]],
    visual_format: str = "svg",
) -> str:
    """Build the expected model response for training."""
    raw_score = response.get("score", 0.0)
    score = int(round(raw_score * 10))
    score = max(0, min(10, score))
    text_reason = response.get("text_reason", "")

    if score < 7:
        summary = "This image has notable issues that should be addressed."
    elif score < 9:
        summary = "This image meets quality standards with minor or no issues."
    else:
        summary = (
            "This is a high-quality image that effectively follows the instruction."
        )

    if visual_format == "svg":
        svg_str = polygons_to_svg(polygons)
        if polygons:
            grounding_desc = (
                "Regions with issues are marked in the SVG below "
                "(coordinates in 0-1000 scale, viewBox=\"0 0 1000 1000\")."
            )
        else:
            grounding_desc = "No specific problem regions identified."

        regions_block = f"""**Problem Regions (SVG):**
```svg
{svg_str}
```"""
    else:
        # polygon format
        polygon_str = json.dumps(polygons) if polygons else "[]"
        if polygons:
            grounding_desc = (
                "Regions with issues are located at the following coordinates "
                "(normalized 0-1000 scale, format: [[x1, y1], [x2, y2], ...])."
            )
        else:
            grounding_desc = "No specific problem regions identified."

        regions_block = f"**Problem Regions (polygon):** {polygon_str}"

    response_text = f"""## Evaluation Results

**Quality Score:** {score}/10

**Assessment:**
{text_reason}

**Visual Grounding:**
{grounding_desc}

{regions_block}

**Summary:**
{summary}"""

    return response_text


def build_user_prompt(instruction: str) -> str:
    """Build the user prompt for evaluation."""
    return f"""Evaluate the quality of this generated image.

Original instruction/prompt: "{instruction}"

Analyze the image for:
1. How well it follows the instruction
2. Visual quality and any artifacts
3. Content coherence and aesthetics

Provide your evaluation with a score, explanation, and identification of problem areas."""


# ---------------------------------------------------------------------------
# prepare command — convert SFT JSONL to Axolotl chat_template format
# ---------------------------------------------------------------------------

def prepare_data(
    input_path: Path,
    output_dir: Path,
    data_root: Path = Path("data"),
    max_samples: Optional[int] = None,
    validation_split: float = 0.1,
    copy_images: bool = True,
    eval_set: Optional[Path] = None,
    visual_format: str = "svg",
) -> Dict[str, Any]:
    """
    Convert SFT JSONL to Axolotl chat_template format.

    Produces:
      - output_dir/train.json   (JSON array of chat_template messages)
      - output_dir/val.json     (JSON array of chat_template messages)
      - output_dir/images/      (copied images)
    """
    logger.info(f"Streaming SFT data from {input_path}")

    eval_image_paths: set = set()
    if eval_set is not None:
        eval_image_paths = load_eval_image_set(eval_set, data_root)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if copy_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

    converted = []
    skipped = 0
    eval_skipped = 0
    polygon_found_count = 0
    polygon_empty_count = 0
    mask_not_found_count = 0
    contradictory_skipped = 0
    # Threshold: samples with score below this AND empty polygons are
    # dropped to avoid teaching "low score + no grounding". Set low so
    # that only the most extreme cases are filtered — low-score samples
    # without grounding still teach the model score calibration, and
    # dropping too aggressively skews the score distribution to only
    # high scores.
    EMPTY_REGION_SCORE_FLOOR = 0.15
    i = -1

    with open(input_path, "r", encoding="utf-8") as _f_in:
      for _raw_line in _f_in:
        if not _raw_line.strip():
            continue
        i += 1
        if max_samples and i >= max_samples:
            break

        sample = json.loads(_raw_line)
        image_path = sample.get("image", "")
        instruction = sample.get("instruction", "")
        response = sample.get("response", {})

        # Resolve image path
        if image_path:
            img_path = Path(image_path)
            if not img_path.is_absolute():
                if not img_path.exists():
                    img_path = data_root / image_path

            if not img_path.exists():
                logger.debug(f"Skipping sample {i}: image not found at {img_path}")
                skipped += 1
                continue

            if eval_image_paths and str(img_path.resolve()) in eval_image_paths:
                logger.debug(f"Skipping eval sample {i}: {image_path}")
                eval_skipped += 1
                continue
        else:
            logger.debug(f"Skipping sample {i}: no image path")
            skipped += 1
            continue

        if copy_images:
            new_img_name = f"{i:06d}_{img_path.name}"
            new_img_path = images_dir / new_img_name
            if not new_img_path.exists():
                # Resize large images to cap visual tokens for Qwen3-VL
                # (patch=14, merge=2 → tokens ≈ ceil(W/28)*ceil(H/28))
                MAX_PIXELS = 512 * 512  # ~342 visual tokens max
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    w, h = img.size
                    if w * h > MAX_PIXELS:
                        scale = (MAX_PIXELS / (w * h)) ** 0.5
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                        img.save(new_img_path)
                    else:
                        shutil.copy2(img_path, new_img_path)
                except Exception:
                    shutil.copy2(img_path, new_img_path)
            # Use absolute path for Axolotl
            image_ref = str(new_img_path.resolve())
        else:
            image_ref = str(img_path.absolute())

        # Extract polygons from mask or heatmap
        polygons = []
        visual_reason = response.get("visual_reason", {})
        visual_data = visual_reason.get("data", "")
        visual_type = visual_reason.get("type", "")

        if visual_data and isinstance(visual_data, str) and visual_type == "svg":
            # SVG string — parse polygons directly from SVG path elements
            polygons = svg_to_polygons(visual_data)
            if polygons:
                polygon_found_count += 1
            else:
                polygon_empty_count += 1
        elif visual_data and isinstance(visual_data, str):
            # File path to mask/heatmap image
            mask_path = Path(visual_data)
            if not mask_path.is_absolute():
                if not mask_path.exists():
                    mask_path = data_root / visual_data
            if mask_path.exists():
                polygons = extract_polygons_from_mask(str(mask_path))
                if polygons:
                    polygon_found_count += 1
                else:
                    polygon_empty_count += 1
            else:
                mask_not_found_count += 1
        elif visual_data and isinstance(visual_data, list) and visual_type in ("heatmap", "mask"):
            # Heatmap/mask stored as nested list (from numpy array)
            try:
                import numpy as np
                from PIL import Image
                import tempfile
                heatmap = np.array(visual_data, dtype=np.float32)
                if heatmap.ndim == 2 and heatmap.max() > 0:
                    # Normalize to 0-255 and save as temp mask
                    mask_arr = (heatmap / heatmap.max() * 255).astype(np.uint8)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        Image.fromarray(mask_arr, mode="L").save(tmp.name)
                        # Soft heatmaps need a lower threshold than binary
                        # masks — RichHF artifact/misalignment maps peak
                        # locally and the 0.5 default loses most regions.
                        polygons = extract_polygons_from_mask(
                            tmp.name, threshold=0.25,
                        )
                        os.unlink(tmp.name)
                    if polygons:
                        polygon_found_count += 1
                    else:
                        polygon_empty_count += 1
            except Exception as e:
                logger.debug(f"Failed to convert heatmap to polygons: {e}")

        # Drop contradictory samples: a low score (problems exist) paired with
        # empty polygons (no regions to point at) teaches the model to predict
        # "bad image + no grounding", which it then over-applies to every
        # input. High-score samples with empty polygons are kept because
        # "good image + no problems" is a legitimate supervision signal.
        sample_score = float(response.get("score", 0.0))
        if not polygons and sample_score < EMPTY_REGION_SCORE_FLOOR:
            contradictory_skipped += 1
            continue

        # Build Axolotl chat_template sample
        user_prompt = build_user_prompt(instruction)
        assistant_response = build_model_response(response, polygons, visual_format)

        system_prompt = get_system_prompt(visual_format)
        axolotl_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": image_ref},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_response},
                    ],
                },
            ],
        }

        converted.append(axolotl_sample)

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1} samples, {len(converted)} converted")

    logger.info(
        f"Converted {len(converted)} samples, skipped {skipped} "
        f"(+{eval_skipped} excluded as eval, "
        f"+{contradictory_skipped} dropped as low-score with empty grounding)"
    )
    logger.info(
        f"Polygon stats: {polygon_found_count} with polygons, "
        f"{polygon_empty_count} empty (mask found but no regions), "
        f"{mask_not_found_count} mask not found"
    )

    # Split train / validation
    split_idx = int(len(converted) * (1 - validation_split))
    train_samples = converted[:split_idx]
    val_samples = converted[split_idx:]

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    logger.info(f"Train: {len(train_samples)} samples -> {train_path}")
    logger.info(f"Val:   {len(val_samples)} samples -> {val_path}")

    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# train command — generate Axolotl config + DeepSpeed ZeRO-3, then train
# ---------------------------------------------------------------------------

DEEPSPEED_ZERO3_CONFIG = {
    "bf16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 3,
        # Keep optimizer states on-GPU: CPU offload makes the save-time 16-bit
        # weight gather segfault in this container. We instead cut training
        # memory via lower image resolution (build_sft_data MAX_PIXELS) so
        # the normal GPU-side save works. Re-enable offload only if GPU OOMs.
        "offload_optimizer": {
            "device": "none",
        },
        "offload_param": {
            "device": "none",
        },
        "overlap_comm": False,
        "contiguous_gradients": True,
        "sub_group_size": 1e8,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


def generate_train_config(
    data_dir: Path,
    model_name: str = DEFAULT_MODEL,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-5,
    cutoff_len: int = 2048,
    report_to: str = "wandb",
    lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_target: str = "all",
) -> Path:
    """Generate an Axolotl training YAML config and DeepSpeed JSON."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    data_dir = Path(data_dir).resolve()
    if output_dir is None:
        output_dir = _default_checkpoint_dir(model_name)
        if lora:
            output_dir = Path(str(output_dir).replace("_full", "_lora"))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write DeepSpeed config (only for full finetuning)
    ds_config_path = output_dir / "ds_zero3_config.json"
    if not lora:
        with open(ds_config_path, "w") as f:
            json.dump(DEEPSPEED_ZERO3_CONFIG, f, indent=2)
        logger.info(f"DeepSpeed ZeRO-3 config saved to {ds_config_path}")

    # Derive wandb run name
    short = model_name.split("/")[-1].replace("-Instruct", "").replace("-VL-", "_").replace("-", "_").lower()
    short = re.sub(r"(\d)b", lambda m: m.group(1) + "B", short)
    run_name = f"{short}_{'lora' if lora else 'full'}"

    # Build datasets list
    datasets = [
        {
            "path": str(data_dir / "train.json"),
            "ds_type": "json",
            "type": "chat_template",
            "split": "train",
        },
    ]

    val_path = data_dir / "val.json"
    val_set_size = 0.0
    if val_path.exists():
        # Use val_set_size to carve a validation split from the training data.
        # Axolotl splits the first dataset by this ratio when > 0. Kept small
        # (~1%) because eval runs generation and is slow; we just want the loss
        # trend, not a precise metric (use run_eval.py for that).
        val_set_size = 0.01

    config = {
        # model
        "base_model": model_name,
        "processor_type": "AutoProcessor",
        # multimodal requirements
        "skip_prepare_dataset": True,
        "remove_unused_columns": False,
        "sample_packing": False,
        # chat template
        "chat_template": AXOLOTL_CHAT_TEMPLATE,
        # datasets
        "datasets": datasets,
        "val_set_size": val_set_size,
        # output
        "output_dir": str(output_dir),
        # sequence
        "sequence_len": cutoff_len,
        "pad_to_sequence_len": False,
        # training
        "seed": 42,
        "num_epochs": epochs,
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": float(learning_rate),
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        # precision
        "bf16": True,
        "tf32": True,
        # memory
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "flash_attention": False,
        "sdp_attention": True,
        "dataloader_num_workers": 0,
        # logging & saving
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "evals_per_epoch": 1,
        "saves_per_epoch": 1,
    }

    if lora:
        # LoRA-specific config
        config["adapter"] = "lora"
        config["lora_r"] = lora_rank
        config["lora_alpha"] = lora_alpha
        config["lora_target_linear"] = True if lora_target == "all" else False
        if lora_target != "all":
            config["lora_target_modules"] = lora_target.split(",")
        config["lora_dropout"] = 0.05
    else:
        # Full finetuning uses DeepSpeed ZeRO-3
        config["deepspeed"] = str(ds_config_path)

    if report_to == "wandb":
        config["wandb_project"] = "viescore2"
        config["wandb_name"] = run_name
    elif report_to == "tensorboard":
        config["wandb_project"] = ""

    config_path = output_dir / "axolotl_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Axolotl training config saved to {config_path}")
    return config_path


def run_train(
    data_dir: Path,
    model_name: str = DEFAULT_MODEL,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-5,
    cutoff_len: int = 2048,
    report_to: str = "wandb",
    lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_target: str = "all",
) -> str:
    """Generate training config and invoke Axolotl training."""
    data_dir = Path(data_dir)

    # Validate data directory
    if not (data_dir / "train.json").exists():
        raise FileNotFoundError(
            f"train.json not found in {data_dir}. "
            "Run the 'prepare' command first."
        )

    config_path = generate_train_config(
        data_dir=data_dir,
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        report_to=report_to,
        lora=lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
    )

    if output_dir is None:
        output_dir = _default_checkpoint_dir(model_name)
        if lora:
            output_dir = Path(str(output_dir).replace("_full", "_lora"))
    output_dir = Path(output_dir).resolve()

    env = os.environ.copy()
    env["AXOLOTL_DO_NOT_TRACK"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if report_to == "wandb":
        env.setdefault("WANDB_PROJECT", "viescore2")

    cmd = [
        "accelerate", "launch",
        "-m", "axolotl.cli.train",
        str(config_path),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    logger.info(f"Training complete. Checkpoints saved to: {output_dir}")
    return str(output_dir)


# ---------------------------------------------------------------------------
# Inference helpers — load full checkpoint with transformers
# ---------------------------------------------------------------------------

def _load_model_and_processor(checkpoint_path: str):
    """Load a Qwen3-VL model and processor from checkpoint.

    Supports both full-finetuned checkpoints and LoRA adapter checkpoints.
    LoRA adapters are detected by the presence of adapter_config.json.
    """
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        logger.warning(
            "Qwen3VLForConditionalGeneration not found in transformers; "
            "falling back to AutoModel"
        )
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    # If checkpoint_path is an Axolotl output dir (contains checkpoint-* subdirs
    # but no model weights at top level), resolve to the latest checkpoint.
    cp = Path(checkpoint_path)
    processor_path = checkpoint_path  # may differ from model path
    if not (cp / "model.safetensors").exists() and not (cp / "pytorch_model.bin").exists():
        subdirs = sorted(cp.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if subdirs:
            processor_path = checkpoint_path  # top-level has processor configs
            checkpoint_path = str(subdirs[-1])
            logger.info(f"Resolved to latest checkpoint: {checkpoint_path}")

    adapter_config = Path(checkpoint_path) / "adapter_config.json"
    if adapter_config.exists():
        # LoRA checkpoint: load base model + merge adapter
        with open(adapter_config) as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
        logger.info(f"Loading LoRA adapter from {checkpoint_path} (base: {base_model})")

        model = model_cls.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged into base model")

        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
    else:
        # Full checkpoint
        logger.info(f"Loading model from {checkpoint_path} ({model_cls.__name__})")
        model = model_cls.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )

    return model, processor


def _generate_response(
    model,
    processor,
    image_path: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 2048,
) -> str:
    """Run inference on a single image+prompt."""
    import torch
    from PIL import Image

    if system_prompt is None:
        system_prompt = get_system_prompt("svg")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True,
    )[0]

    return response_text


def _parse_score(text: str) -> Optional[float]:
    """Extract quality score (0-10) from model response text."""
    for pat in [
        r"\*\*Quality Score:\*\*\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"Quality Score:\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"[Ss]core:\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
    ]:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# test command
# ---------------------------------------------------------------------------

def run_test(
    checkpoint_path: str,
    image_path: str,
    instruction: str,
    render_heatmap: bool = False,
    heatmap_output: Optional[str] = None,
) -> Dict[str, Any]:
    """Test the fine-tuned model on a single image."""
    model, processor = _load_model_and_processor(checkpoint_path)

    user_prompt = build_user_prompt(instruction)
    response_text = _generate_response(model, processor, image_path, user_prompt)

    result = {
        "response": response_text,
        "polygons": [],
        "heatmap_path": None,
    }

    polygons = parse_polygons_from_response(response_text)
    result["polygons"] = polygons

    if render_heatmap and polygons:
        try:
            from PIL import Image

            img = Image.open(image_path)
            image_size = img.size

            if not heatmap_output:
                img_p = Path(image_path)
                heatmap_output = str(
                    img_p.parent / f"{img_p.stem}_heatmap.png"
                )

            heatmap_path = render_heatmap_from_polygons(
                polygons=polygons,
                output_path=heatmap_output,
                image_size=image_size,
                base_image_path=image_path,
            )
            result["heatmap_path"] = heatmap_path
            logger.info(f"Heatmap saved to: {heatmap_path}")
        except Exception as e:
            logger.warning(f"Failed to render heatmap: {e}")

    return result


# ---------------------------------------------------------------------------
# batch-test command
# ---------------------------------------------------------------------------

def run_batch_test(
    checkpoint_path: str,
    eval_data_path: str,
    output_path: str,
    model_label: str = "",
    pics_dir: str = "",
) -> Dict[str, Any]:
    """Load model once, evaluate all samples in an eval JSONL, save results."""
    import time as _time

    model, processor = _load_model_and_processor(checkpoint_path)
    label = model_label or Path(checkpoint_path).name

    # Load eval samples
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} eval samples from {eval_data_path}")

    results = []
    total_time = 0.0

    for i, sample in enumerate(samples):
        image_path = sample.get("image", "")
        instruction = sample.get("instruction", "")
        gt_score = sample.get("response", {}).get("score", None)
        gt_text = sample.get("response", {}).get("text_reason", "")
        sample_id = sample.get("meta", {}).get("id", f"sample_{i}")
        source = sample.get("meta", {}).get("source", "")
        pic_id = f"pic_{i+1:03d}"

        if not Path(image_path).exists():
            logger.warning(f"[{i+1}/{len(samples)}] SKIP {sample_id}: image not found")
            results.append({"id": sample_id, "source": source, "pic_id": pic_id, "error": "image not found"})
            continue

        user_prompt = build_user_prompt(instruction)

        t0 = _time.time()
        try:
            response_text = _generate_response(
                model, processor, image_path, user_prompt,
            )
        except Exception as e:
            logger.error(f"[{i+1}/{len(samples)}] ERROR {sample_id}: {e}")
            results.append({"id": sample_id, "source": source, "pic_id": pic_id, "error": str(e)})
            continue
        elapsed = _time.time() - t0
        total_time += elapsed

        pred_score = _parse_score(response_text)
        pred_polygons = parse_polygons_from_response(response_text)

        # Render heatmap overlay images
        gt_visual_data = (
            sample.get("response", {})
            .get("visual_reason", {})
            .get("data", "")
        )

        if pics_dir:
            pic_folder = Path(pics_dir) / pic_id
            pic_folder.mkdir(parents=True, exist_ok=True)

            # Always overwrite original.png so pics stay in sync with the
            # current eval.jsonl ordering (eval sets get rebuilt over time).
            orig_dest = pic_folder / "original.png"
            shutil.copy2(image_path, orig_dest)

            # GT visual overlay (mask → red overlay on original). Always
            # re-render so it tracks the current sample at this pic_id.
            gt_dest = pic_folder / "gt.png"
            if gt_visual_data:
                try:
                    from PIL import Image as _Image
                    import numpy as _np
                    orig_img = _Image.open(image_path).convert("RGBA")
                    if isinstance(gt_visual_data, list):
                        # data is a 2D mask array (e.g. 640x640 floats)
                        mask_arr = (_np.array(gt_visual_data) * 255).clip(0, 255).astype(_np.uint8)
                    elif isinstance(gt_visual_data, str) and "<path" in gt_visual_data:
                        # SVG string with path elements — rasterize to mask
                        polys = svg_to_polygons(gt_visual_data)
                        w, h = orig_img.size
                        from PIL import ImageDraw as _ImageDraw
                        mask_pil_tmp = _Image.new("L", (w, h), 0)
                        drw = _ImageDraw.Draw(mask_pil_tmp)
                        for poly in polys:
                            if len(poly) >= 3:
                                pts = [(int(x / 1000 * w), int(y / 1000 * h)) for x, y in poly]
                                drw.polygon(pts, fill=255)
                        mask_arr = _np.array(mask_pil_tmp)
                    elif isinstance(gt_visual_data, str) and Path(gt_visual_data).exists():
                        mask_img = _Image.open(gt_visual_data).convert("L")
                        mask_img = mask_img.resize(orig_img.size, _Image.NEAREST)
                        mask_arr = _np.array(mask_img)
                    else:
                        raise ValueError(f"unsupported gt_visual_data type: {type(gt_visual_data)}")
                    mask_pil = _Image.fromarray(mask_arr, "L").resize(orig_img.size, _Image.NEAREST)
                    mask_arr = _np.array(mask_pil)
                    overlay = _np.zeros((*orig_img.size[::-1], 4), dtype=_np.uint8)
                    overlay[mask_arr > 128] = [255, 0, 0, 140]  # red, semi-transparent
                    overlay_img = _Image.fromarray(overlay, "RGBA")
                    combined = _Image.alpha_composite(orig_img, overlay_img).convert("RGB")
                    combined.save(str(gt_dest))
                except Exception as e:
                    logger.warning(f"GT overlay render failed for {sample_id}: {e}")
                    shutil.copy2(image_path, gt_dest)
            else:
                shutil.copy2(image_path, gt_dest)

            heatmap_dest = str(pic_folder / f"{label}.png")
            if pred_polygons:
                try:
                    from PIL import Image as _Image
                    img = _Image.open(image_path)
                    render_heatmap_from_polygons(
                        polygons=pred_polygons,
                        output_path=heatmap_dest,
                        image_size=img.size,
                        base_image_path=image_path,
                    )
                except Exception as e:
                    logger.warning(f"Heatmap render failed for {sample_id}: {e}")
                    shutil.copy2(image_path, heatmap_dest)
            else:
                shutil.copy2(image_path, heatmap_dest)

            with open(str(pic_folder / f"{label}.txt"), "w", encoding="utf-8") as tf:
                tf.write(response_text)

        result = {
            "id": sample_id,
            "source": source,
            "pic_id": pic_id,
            "instruction": instruction,
            "image": image_path,
            "gt_score": gt_score,
            "gt_text_reason": gt_text,
            "gt_visual": gt_visual_data,
            "pred_response": response_text,
            "pred_score": pred_score,
            "pred_polygons": pred_polygons,
            "inference_time_s": round(elapsed, 2),
        }
        results.append(result)

        score_str = f"{pred_score:.1f}/10" if pred_score is not None else "N/A"
        gt_str = f"{gt_score:.3f}" if gt_score is not None else "N/A"
        logger.info(
            f"[{i+1}/{len(samples)}] {sample_id}  "
            f"pred={score_str}  gt={gt_str}  "
            f"polygons={len(pred_polygons)}  time={elapsed:.1f}s"
        )

    # Summary
    scored = [r for r in results if r.get("pred_score") is not None and r.get("gt_score") is not None]
    errors = [r for r in results if "error" in r]

    summary = {
        "model": label,
        "checkpoint": checkpoint_path,
        "total_samples": len(samples),
        "scored_samples": len(scored),
        "errors": len(errors),
        "avg_inference_time_s": round(total_time / max(len(results) - len(errors), 1), 2),
    }
    if scored:
        pred_scores = [r["pred_score"] for r in scored]
        gt_scores = [r["gt_score"] * 10 for r in scored]
        abs_errors = [abs(p - g) for p, g in zip(pred_scores, gt_scores)]
        summary["avg_pred_score"] = round(sum(pred_scores) / len(pred_scores), 2)
        summary["avg_gt_score"] = round(sum(gt_scores) / len(gt_scores), 2)
        summary["mae"] = round(sum(abs_errors) / len(abs_errors), 3)

    output = {"summary": summary, "results": results}
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Full SFT Fine-Tuning for Image Quality Evaluation (Axolotl + DeepSpeed ZeRO-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # -- prepare --
    p_prepare = subparsers.add_parser(
        "prepare", help="Convert SFT data to Axolotl format",
    )
    p_prepare.add_argument(
        "--input", type=Path, required=True, help="Input SFT JSONL file",
    )
    p_prepare.add_argument(
        "--output", type=Path, default=Path("axolotl_qwen3_train_data"),
        help="Output directory",
    )
    p_prepare.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root dir for resolving relative image paths",
    )
    p_prepare.add_argument(
        "--max-samples", type=int, help="Maximum samples to convert",
    )
    p_prepare.add_argument(
        "--validation-split", type=float, default=0.1,
        help="Validation split ratio",
    )
    p_prepare.add_argument(
        "--no-copy-images", action="store_true",
        help="Don't copy images to output dir",
    )
    p_prepare.add_argument(
        "--eval-set", type=Path, default=Path("eval_samples/eval.jsonl"),
        help="Eval JSONL whose images are excluded from training "
             "(default: eval_samples/eval.jsonl)",
    )
    p_prepare.add_argument(
        "--no-eval-filter", action="store_true",
        help="Don't exclude eval images from training data",
    )
    p_prepare.add_argument(
        "--format", choices=["svg", "polygon"], default="svg",
        help="Visual grounding format: svg or polygon (default: svg)",
    )

    # -- train --
    p_train = subparsers.add_parser(
        "train", help="Launch full SFT via Axolotl",
    )
    p_train.add_argument(
        "--data-dir", type=Path, required=True,
        help="Prepared data directory (from prepare command)",
    )
    p_train.add_argument(
        "--model", default=DEFAULT_MODEL, help="Base model name",
    )
    p_train.add_argument(
        "--output-dir", type=Path, default=None,
        help="Checkpoint output directory (default: checkpoints/<model>_full)",
    )
    p_train.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p_train.add_argument(
        "--batch-size", type=int, default=1, help="Per-device batch size",
    )
    p_train.add_argument(
        "--gradient-accumulation", type=int, default=16,
        help="Gradient accumulation steps",
    )
    p_train.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    p_train.add_argument(
        "--cutoff-len", type=int, default=2048, help="Max sequence length",
    )
    p_train.add_argument(
        "--report-to", choices=["wandb", "tensorboard", "none"],
        default="wandb", help="Logging backend (default: wandb)",
    )
    p_train.add_argument(
        "--lora", action="store_true",
        help="Use LoRA instead of full finetuning",
    )
    p_train.add_argument(
        "--lora-r", type=int, default=32, help="LoRA rank (default: 32)",
    )
    p_train.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)",
    )
    p_train.add_argument(
        "--lora-target", default="all",
        help="LoRA target modules: 'all' for all linear, or comma-separated names",
    )

    # -- test --
    p_test = subparsers.add_parser("test", help="Test fine-tuned model")
    p_test.add_argument(
        "--checkpoint", required=True, help="Checkpoint directory path",
    )
    p_test.add_argument("--image", required=True, help="Test image path")
    p_test.add_argument(
        "--instruction", required=True, help="Original instruction for the image",
    )
    p_test.add_argument(
        "--render-heatmap", action="store_true",
        help="Render polygon heatmap overlay",
    )
    p_test.add_argument("--heatmap-output", help="Heatmap output path")

    # -- batch-test --
    p_batch = subparsers.add_parser(
        "batch-test", help="Evaluate model on all samples in an eval JSONL",
    )
    p_batch.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p_batch.add_argument(
        "--eval-data", type=Path, required=True, help="Eval JSONL file",
    )
    p_batch.add_argument(
        "--output", type=Path, required=True, help="Output JSON path",
    )
    p_batch.add_argument("--label", default="", help="Model label for results")
    p_batch.add_argument(
        "--pics-dir", default="", help="Directory for per-sample heatmap images",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "prepare":
            result = prepare_data(
                input_path=args.input,
                output_dir=args.output,
                data_root=args.data_root,
                max_samples=args.max_samples,
                validation_split=args.validation_split,
                copy_images=not args.no_copy_images,
                eval_set=None if args.no_eval_filter else args.eval_set,
                visual_format=args.format,
            )
            logger.info("Data preparation complete:")
            logger.info(f"  Train: {result['train_samples']} samples")
            logger.info(f"  Val:   {result['val_samples']} samples")
            logger.info(f"  Dir:   {result['output_dir']}")
            logger.info(
                f"\nNext step: python scripts/train_sft.py train "
                f"--data-dir {result['output_dir']}"
            )

        elif args.command == "train":
            checkpoint = run_train(
                data_dir=args.data_dir,
                model_name=args.model,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.learning_rate,
                cutoff_len=args.cutoff_len,
                report_to=args.report_to,
                lora=args.lora,
                lora_rank=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_target=args.lora_target,
            )
            logger.info(f"Training complete! Checkpoints: {checkpoint}")

        elif args.command == "test":
            result = run_test(
                checkpoint_path=args.checkpoint,
                image_path=args.image,
                instruction=args.instruction,
                render_heatmap=args.render_heatmap,
                heatmap_output=args.heatmap_output,
            )

            print("\n" + "=" * 50)
            print("Model Response:")
            print("=" * 50)
            print(result["response"])

            if result["polygons"]:
                print("\n" + "-" * 50)
                print(
                    f"Extracted Problem Regions ({len(result['polygons'])} polygons):"
                )
                svg_str = polygons_to_svg(result["polygons"])
                print(f"\nSVG:\n{svg_str}")
                for i, poly in enumerate(result["polygons"]):
                    pts_str = ", ".join(f"({pt[0]}, {pt[1]})" for pt in poly)
                    print(
                        f"  {i + 1}. [{pts_str}] "
                        f"({len(poly)} vertices, 0-1000 scale)"
                    )

            if result["heatmap_path"]:
                print(f"\nHeatmap saved to: {result['heatmap_path']}")

        elif args.command == "batch-test":
            run_batch_test(
                checkpoint_path=args.checkpoint,
                eval_data_path=str(args.eval_data),
                output_path=str(args.output),
                model_label=args.label,
                pics_dir=args.pics_dir,
            )

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch-JSON evaluation runner for VIEScore2.

For each eval sample:
  1. Show the model the FULL image + generation prompt
  2. Ask it for a single 16×16 JSON array (1 = problematic, 0 = clean)
  3. Parse the array into a 16×16 binary grid
  4. Compute GridIoU against the GT annotation grid

Localization is emitted in one forward pass as a sparse text grid, then
parsed and scored deterministically against the ground-truth grid.

Usage:
    python viescore2/run_eval.py \\
        --checkpoint Qwen/Qwen3-VL-8B-Instruct \\
        --eval-data  eval_samples/eval.jsonl \\
        --output     eval_results/qwen3_8B_base.json \\
        --label      qwen3_8B_base
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

GRID_SIZE = 16

# Must match build_sft_data.MAX_PIXELS so eval images are downscaled the
# same way training images were. Without this the model sees a far larger
# visual-token grid at eval than it ever saw in training (train images were
# capped at 384x384), shifting it off-distribution and worsening the collapse.
MAX_PIXELS = 384 * 384


def _resize_to(image, max_pixels: int):
    """Downscale ``image`` to at most ``max_pixels`` (LANCZOS), else return as-is."""
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        from PIL import Image as _Image
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), _Image.LANCZOS
        )
    return image


def resize_to_train_res(image):
    """Downscale the generated image to MAX_PIXELS exactly as the builder did."""
    return _resize_to(image, MAX_PIXELS)


# Share the EXACT prompt text, multi-image layout and conditioning-image
# derivation with the SFT builder so eval is never off-distribution from
# training. (viescore2 is on sys.path when this file is run.)
from build_sft_data import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    derive_input_images,
    heatmap_png_to_grids,
    extract_axis_scores,
    MAX_PIXELS_REF,
)


# ── GT annotation → binary grid ───────────────────────────────────────────────

def gt_visual_to_grid(
    visual_reason: Dict[str, Any],
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    vtype = visual_reason.get("type", "")
    data = visual_reason.get("data")
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    if not data:
        return grid

    if vtype in ("mask", "heatmap"):
        if isinstance(data, str):
            try:
                img = Image.open(data).convert("L")
                arr = np.array(img, dtype=np.float32)
                # Value-range adaptive: {0,255} masks (RichHF/ImagenWorld) get
                # normalised; {0,1}-pixel labels (PAL4VST) are already in [0,1]
                # — blindly dividing by 255 would silently zero every grid.
                if arr.max() > 1.0:
                    arr = arr / 255.0
            except Exception:
                return grid
        else:
            arr = np.array(data, dtype=np.float32)
        if arr.ndim != 2:
            return grid
        threshold = 0.5 if vtype == "mask" else 0.25
        small = Image.fromarray((arr * 255).astype(np.uint8)).resize(
            (grid_size, grid_size), Image.LANCZOS
        )
        return (np.array(small, dtype=np.float32) / 255.0 > threshold).astype(np.uint8)

    elif vtype == "bbox":
        raw = [data] if isinstance(data[0], (int, float)) else data
        canvas = Image.new("L", (1000, 1000), 0)
        draw = ImageDraw.Draw(canvas)
        for b in raw:
            x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            draw.rectangle([x, y, x + w, y + h], fill=255)
        return (np.array(canvas.resize((grid_size, grid_size), Image.LANCZOS)) > 127).astype(np.uint8)

    elif vtype == "polygon":
        if not data or not isinstance(data[0], (list, tuple)):
            return grid
        canvas = Image.new("L", (1000, 1000), 0)
        draw = ImageDraw.Draw(canvas)
        pts = [(float(p[0]), float(p[1])) for p in data]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
        return (np.array(canvas.resize((grid_size, grid_size), Image.LANCZOS)) > 127).astype(np.uint8)

    elif vtype == "svg":
        svg_str = data if isinstance(data, str) else ""
        coords = []
        for m in re.finditer(r'[MLCml]\s*([\d.]+)\s*[,\s]\s*([\d.]+)', svg_str):
            coords.append((float(m.group(1)), float(m.group(2))))
        if not coords:
            return grid
        canvas = Image.new("L", (1000, 1000), 0)
        draw = ImageDraw.Draw(canvas)
        if len(coords) >= 3:
            draw.polygon(coords, fill=255)
        else:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            draw.rectangle([min(xs), min(ys), max(xs), max(ys)], fill=255)
        return (np.array(canvas.resize((grid_size, grid_size), Image.LANCZOS)) > 127).astype(np.uint8)

    return grid


# ── Output parsing ────────────────────────────────────────────────────────────

def parse_grid_from_response(
    text: str, grid_size: int = GRID_SIZE
) -> Optional[np.ndarray]:
    """Parse the SPARSE per-row enumeration back into a 16×16 binary grid.

    The target format is one line per affected row, ``"r<row>: <col>,<col>"``
    with 1-based indices, and the literal ``none`` for a fully clean image.
    The parser is tolerant of ``r5:`` / ``row 5:`` / ``5:`` and arbitrary
    spacing. Indices outside 1..16 are ignored.

    A recognised ``none`` (or any response that yields zero coordinates while
    explicitly signalling "none") parses to an all-zero grid — a SUCCESSFUL
    parse, not a failure. Returns None only when the text matches neither
    shape, so genuine generation garbage is still counted as a parse failure.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    found_any = False

    # Dual-channel (inspector) responses mark severe cells with "!" and group
    # rows under "artifact:" / "misalign:" headers; hierarchical (M1)
    # responses append "[1,3]" subcell brackets. Stripping marks and brackets
    # lets the row regex see every column, and the headers (no digits before
    # the colon) never match it — so this parse is the CHANNEL-UNION 16×16
    # grid, directly comparable with flat responses.
    text = re.sub(r'\[[^\]]*\]', '', text).replace("!", "")

    # "r5: 8,9"  /  "row 5: 8, 9"  /  "5: 8 9 10"  (colon required to avoid
    # latching onto stray numbers in any prose the model emits).
    for m in re.finditer(r'(?:r|row)?\s*(\d{1,2})\s*:\s*([0-9][0-9,\s]*)', text, re.IGNORECASE):
        row = int(m.group(1))
        if not (1 <= row <= grid_size):
            continue
        for cs in re.findall(r'\d{1,2}', m.group(2)):
            col = int(cs)
            if 1 <= col <= grid_size:
                grid[row - 1, col - 1] = 1
                found_any = True

    if found_any:
        return grid
    if re.search(r'\bnone\b', text, re.IGNORECASE):
        return grid  # explicit clean image → all-zero, parsed successfully
    return None


def parse_channels_from_response(
    text: str, grid_size: int = GRID_SIZE
) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse an inspector dual-channel response into per-channel grids.

    Returns ``{channel: {"problem": grid, "major": grid}}`` for each
    ``artifact:`` / ``misalign:`` section found (empty dict for flat
    responses). ``major`` marks the columns carrying a ``!`` severity mark.
    """
    headers = list(re.finditer(r'^[ \t]*(artifact|misalign)\s*:[ \t]*$',
                               text, re.IGNORECASE | re.MULTILINE))
    channels: Dict[str, Dict[str, np.ndarray]] = {}
    for i, h in enumerate(headers):
        name = h.group(1).lower()
        section = text[h.end(): headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        problem = np.zeros((grid_size, grid_size), dtype=np.uint8)
        major = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for m in re.finditer(r'(?:r|row)?\s*(\d{1,2})\s*:\s*([0-9][0-9,!\s]*)',
                             section, re.IGNORECASE):
            row = int(m.group(1))
            if not (1 <= row <= grid_size):
                continue
            for cs in re.finditer(r'(\d{1,2})(!?)', m.group(2)):
                col = int(cs.group(1))
                if 1 <= col <= grid_size:
                    problem[row - 1, col - 1] = 1
                    if cs.group(2):
                        major[row - 1, col - 1] = 1
        channels[name] = {"problem": problem, "major": major}
    return channels


def parse_hier_from_response(text: str, grid_size: int = GRID_SIZE) -> Optional[np.ndarray]:
    """Parse a hierarchical (16→32) response into a 32×32 binary grid.

    Bare column = whole cell (all four subcells); ``8[1,3]`` = only the
    listed quarters (1=TL, 2=TR, 3=BL, 4=BR). Channel headers and ``!``
    marks are tolerated. Returns None when nothing parses (mirrors
    :func:`parse_grid_from_response` semantics)."""
    g32 = np.zeros((grid_size * 2, grid_size * 2), dtype=np.uint8)
    found = False
    for m in re.finditer(
        r'(?:r|row)?\s*(\d{1,2})\s*:\s*((?:\d{1,2}!?(?:\[[\d,\s]*\])?[,\s]*)+)',
        text, re.IGNORECASE,
    ):
        row = int(m.group(1))
        if not (1 <= row <= grid_size):
            continue
        for cm in re.finditer(r'(\d{1,2})!?(\[([\d,\s]*)\])?', m.group(2)):
            col = int(cm.group(1))
            if not (1 <= col <= grid_size):
                continue
            found = True
            subs = ([int(s) for s in re.findall(r'[1-4]', cm.group(3))]
                    if cm.group(2) else [1, 2, 3, 4])
            for s in subs:
                dr, dc = divmod(s - 1, 2)
                g32[(row - 1) * 2 + dr, (col - 1) * 2 + dc] = 1
    if found:
        return g32
    if re.search(r'\bnone\b', text, re.IGNORECASE):
        return g32
    return None


def parse_axis_scores(text: str) -> tuple:
    """Extract the dual-axis ``pq: N/10`` / ``sc: N/10`` lines → (pq, sc) on
    the 0-10 scale, None per missing axis. Flat/score-line responses give
    (None, None), keeping every earlier protocol untouched."""
    out = []
    for axis in ("pq", "sc"):
        m = re.search(rf'\b{axis}\s*:\s*(\d{{1,2}}(?:\.\d+)?)\s*/\s*10', text, re.IGNORECASE)
        v = float(m.group(1)) if m else None
        out.append(v if v is not None and 0.0 <= v <= 10.0 else None)
    return tuple(out)


def parse_score_from_response(text: str) -> Optional[float]:
    """Extract the ``score: <0-10>/10`` line (score-head targets) → 0-10.

    Returns None when absent (pre-v3 models never emit it), keeping this eval
    fully backward-compatible: grid metrics are unaffected and the score
    correlation simply reports n=0.
    """
    m = re.search(r'score\s*:\s*(\d{1,2}(?:\.\d+)?)\s*/\s*10', text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    return val if 0.0 <= val <= 10.0 else None


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_grid_iou(gt_grid: np.ndarray, pred_grid: Optional[np.ndarray]) -> float:
    if pred_grid is None:
        return 0.0
    gt_b = gt_grid.astype(bool)
    pred_b = pred_grid.astype(bool)
    if not gt_b.any() and not pred_b.any():
        return 1.0
    if not gt_b.any() or not pred_b.any():
        return 0.0
    inter = int((gt_b & pred_b).sum())
    union = int((gt_b | pred_b).sum())
    return inter / union if union > 0 else 1.0


def compute_cell_confusion(
    gt_grid: np.ndarray, pred_grid: Optional[np.ndarray]
) -> Dict[str, int]:
    """Cell-level TP/FP/FN/TN for the 16×16 problem grids.

    Unlike IoU, these counts are additive across samples, so a *micro-averaged*
    precision/recall/F1 over the whole eval set is not inflated by the
    empty-GT / empty-pred "both blank → 1.0" freebie that dominates avg IoU.
    """
    gt_b = gt_grid.astype(bool)
    if pred_grid is None:
        return {"tp": 0, "fp": 0, "fn": int(gt_b.sum()), "tn": int((~gt_b).sum())}
    pred_b = pred_grid.astype(bool)
    return {
        "tp": int((gt_b & pred_b).sum()),
        "fp": int((~gt_b & pred_b).sum()),
        "fn": int((gt_b & ~pred_b).sum()),
        "tn": int((~gt_b & ~pred_b).sum()),
    }


def rank_avg(x):
    """Average ranks with tie handling (scipy-style). argsort(argsort) is
    biased under heavy ties — integer 0-10 scores are ~95% ties ()."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    vals = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def prf_from_counts(tp: int, fp: int, fn: int):
    """Precision / recall / F1 from cell counts.

    Both-empty (no GT and no predicted positives) → all 1.0; otherwise an
    undefined denominator (no positives on one side) → 0.0 for that term.
    """
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ── Model loading ─────────────────────────────────────────────────────────────

def _predict_grid_api(
    client,
    api_model: str,
    image,
    instruction: str,
    input_images: Optional[List] = None,
    max_new_tokens: int = 1024,
    channels: bool = False,
    with_score: bool = True,
    score_last: bool = False,
    axes: bool = False,
    hier: bool = False,
    score_only: bool = False,
) -> str:
    """API-backed twin of ``_predict_grid`` (): byte-identical
    SYSTEM_PROMPT and user prompt text, same image order (conditioning images
    before the generated image, at the same resized resolutions), sent to an
    OpenAI-compatible chat endpoint. Retries transient errors with backoff."""
    import base64
    import io
    import time as _time

    def to_data_url(im) -> str:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    input_images = input_images or []
    if score_only:
        prompt_text = (SCORE_ONLY_PROMPT_PREFIX + instruction
                       + SCORE_ONLY_PROMPT_SUFFIX)
    else:
        prompt_text = build_user_prompt(
            instruction, len(input_images), channels=channels,
            with_score=with_score, score_last=score_last, axes=axes,
            hier=hier,
        )
    user_content = [{"type": "image_url", "image_url": {"url": to_data_url(im)}}
                    for im in list(input_images) + [image]]
    user_content.append({"type": "text", "text": prompt_text})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    last_err = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=api_model,
                messages=messages,
                max_completion_tokens=max_new_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # rate limits / transient 5xx
            last_err = e
            if "insufficient_quota" in str(e):
                raise ApiCreditsExhausted(str(e))
            _time.sleep(5 * (2 ** attempt))
    raise RuntimeError(f"API predict failed after retries: {last_err}")


class ApiCreditsExhausted(RuntimeError):
    """Prepaid API credits ran out — fatal for the whole run, not the row."""


def _predict_grid_gemini(
    client,
    api_model: str,
    image,
    instruction: str,
    input_images: Optional[List] = None,
    max_new_tokens: int = 1024,
    channels: bool = False,
    with_score: bool = True,
    score_last: bool = False,
    axes: bool = False,
    hier: bool = False,
    score_only: bool = False,
) -> str:
    """Gemini twin of ``_predict_grid_api``: byte-identical SYSTEM_PROMPT and
    user prompt text, same image order, temperature 0. Retries with backoff."""
    import io
    import time as _time
    from google.genai import types as _gt

    def to_part(im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return _gt.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    input_images = input_images or []
    if score_only:
        prompt_text = (SCORE_ONLY_PROMPT_PREFIX + instruction
                       + SCORE_ONLY_PROMPT_SUFFIX)
    else:
        prompt_text = build_user_prompt(
            instruction, len(input_images), channels=channels,
            with_score=with_score, score_last=score_last, axes=axes,
            hier=hier,
        )
    parts = [to_part(im) for im in list(input_images) + [image]]
    parts.append(_gt.Part.from_text(text=prompt_text))
    cfg = _gt.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=max_new_tokens,
        temperature=0.0,
    )
    last_err = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=api_model, contents=parts, config=cfg)
            return (resp.text or "").strip()
        except Exception as e:  # rate limits / transient 5xx
            last_err = e
            # Depleted prepaid credits never recover by waiting: abort the
            # run instead of burning rows as errors (429 rate limits, by
            # contrast, do back off and retry).
            if "credits are depleted" in str(e):
                raise ApiCreditsExhausted(str(e))
            _time.sleep(5 * (2 ** attempt))
    raise RuntimeError(f"Gemini predict failed after retries: {last_err}")


def _predict_grid_anthropic(
    client,
    api_model: str,
    image,
    instruction: str,
    input_images: Optional[List] = None,
    max_new_tokens: int = 1024,
    channels: bool = False,
    with_score: bool = True,
    score_last: bool = False,
    axes: bool = False,
    hier: bool = False,
    score_only: bool = False,
) -> str:
    """Anthropic twin of ``_predict_grid_api``: byte-identical SYSTEM_PROMPT
    and user prompt text, same image order. Claude Opus 5.5 takes no
    temperature parameter and thinking cannot be disabled (adaptive by
    default; thinking tokens count toward max_tokens) — run with a generous
    cap and vendor-default reasoning; both disclosed in the fairness audit."""
    import base64 as _b64
    import io
    import time as _time
    import anthropic as _anthropic

    def to_block(im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": _b64.standard_b64encode(buf.getvalue()).decode()}}

    input_images = input_images or []
    if score_only:
        prompt_text = (SCORE_ONLY_PROMPT_PREFIX + instruction
                       + SCORE_ONLY_PROMPT_SUFFIX)
    else:
        prompt_text = build_user_prompt(
            instruction, len(input_images), channels=channels,
            with_score=with_score, score_last=score_last, axes=axes,
            hier=hier,
        )
    content = [to_block(im) for im in list(input_images) + [image]]
    content.append({"type": "text", "text": prompt_text})
    last_err = None
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=api_model,
                max_tokens=max_new_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            return "\n".join(b.text for b in resp.content
                             if b.type == "text").strip()
        except _anthropic.RateLimitError as e:
            last_err = e
            _time.sleep(15 * (attempt + 1))
        except Exception as e:
            last_err = e
            if "credit balance is too low" in str(e):
                raise ApiCreditsExhausted(str(e))
            _time.sleep(5 * (2 ** attempt))
    raise RuntimeError(f"Anthropic predict failed after retries: {last_err}")


def _load_model_and_processor(checkpoint_path: str):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    cp = Path(checkpoint_path)
    processor_path = checkpoint_path
    if not (cp / "model.safetensors").exists() and not (cp / "pytorch_model.bin").exists():
        subdirs = sorted(cp.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if subdirs:
            processor_path = checkpoint_path
            checkpoint_path = str(subdirs[-1])
            print(f"Resolved to latest checkpoint: {checkpoint_path}")

    adapter_config = Path(checkpoint_path) / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
        print(f"Loading LoRA adapter from {checkpoint_path} (base: {base_model})")
        model = model_cls.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        print(f"Loading model from {checkpoint_path}")
        model = model_cls.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    return model, processor


# ── Single full-image inference ───────────────────────────────────────────────

# Byte-identical to the EvalMuse score-only booster's training prompt (built
# by the legacy EM builder, not build_user_prompt): no grid mention at all.
# Used when a source protocol sets score_only=true, so eval prompts match what
# that source's training rows actually asked for (P0-1 prompt matching).
SCORE_ONLY_PROMPT_PREFIX = 'Generation prompt: "'
SCORE_ONLY_PROMPT_SUFFIX = (
    '"\nGive the overall quality score of the GENERATED image as '
    '"score: <0-10>/10".'
)


def _predict_grid(
    model,
    processor,
    image,
    instruction: str,
    input_images: Optional[List] = None,
    max_new_tokens: int = 1024,
    channels: bool = False,
    with_score: bool = True,
    score_last: bool = False,
    axes: bool = False,
    hier: bool = False,
    score_only: bool = False,
) -> str:
    """Run one forward pass; return the raw response text.

    ``input_images`` are the conditioning images (original-to-edit / references)
    shown BEFORE the generated ``image``, mirroring the SFT layout. When empty
    this is the original single-image path. ``channels=True`` asks for the
    dual-channel (artifact/misalign + severity) output format;
    ``with_score=False`` drops the score request (for score-free ablations —
    same prompt-matching principle as ``channels``). ``score_only=True``
    replaces the whole prompt with the EvalMuse booster's verbatim score-only
    template (no grid request).
    """
    import torch

    input_images = input_images or []
    all_images = list(input_images) + [image]
    user_content = [{"type": "image", "image": im} for im in input_images]
    user_content.append({"type": "image", "image": image})
    if score_only:
        prompt_text = (SCORE_ONLY_PROMPT_PREFIX + instruction
                       + SCORE_ONLY_PROMPT_SUFFIX)
    else:
        prompt_text = build_user_prompt(
            instruction, len(input_images), channels=channels,
            with_score=with_score, score_last=score_last, axes=axes,
            hier=hier,
        )
    user_content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=all_images, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return response


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_eval(
    checkpoint: str,
    eval_data_path: Path,
    output_path: Path,
    model_label: str,
    max_new_tokens: int = 1024,
    limit: int = 0,
    channel_sources: Optional[set] = None,
    with_score: bool = True,
    score_last: bool = False,
    dual_axis: bool = False,
    hier: bool = False,
    verbalize_out: bool = False,
    source_protocols: Optional[Dict[str, Dict[str, Any]]] = None,
    no_cond_images: bool = False,
    api_backend: Optional[str] = None,
    api_model: Optional[str] = None,
) -> Dict[str, Any]:
    """``channel_sources``: sources whose samples are prompted with the
    dual-channel format regardless of GT sidecars. Inspector-class models see
    EVERY RichHF training sample with the channel prompt, so flat-prompting
    that source at eval is out-of-distribution and collapses its grids
    (measured: F1 0.186 flat vs 0.475 channel on the same checkpoint). The
    union parse keeps metrics comparable either way.

    ``source_protocols``: per-source prompt schema overrides (P0-1). Maps
    ``meta.source`` → {"with_score": bool, "axes": bool, "channels": bool,
    "score_only": bool}. A listed source is prompted EXACTLY as its training
    rows were (e.g. PAL4VST grid-only, EvalMuse bare score) instead of
    inheriting the global flags; unlisted sources keep the global behavior.
    Under the old single global --dual-axis, PAL4VST-test rows were forced to
    emit pq/sc lines they were never trained to produce there (measured cost:
    subset F1 0.238 vs 0.398 on the full test under a closer prompt)."""
    channel_sources = channel_sources or set()
    from PIL import Image

    samples = []
    with open(eval_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if limit and limit > 0:
        samples = samples[:limit]
        print(f"Loaded {len(samples)} eval samples (limited to first {limit}) "
              f"from {eval_data_path}")
    else:
        print(f"Loaded {len(samples)} eval samples from {eval_data_path}")

    if api_backend == "openai":
        import os as _os
        from openai import OpenAI
        _key = _os.environ.get("OPENAI_API_KEY")
        if not _key and Path(".env").exists():
            for _l in Path(".env").read_text().splitlines():
                if _l.startswith("OPENAI_API_KEY="):
                    _key = _l.split("=", 1)[1].strip()
        if not _key:
            raise SystemExit("OPENAI_API_KEY not set (env or .env)")
        api_client = OpenAI(api_key=_key)
        model, processor = None, None
        print(f"API backend: openai / {api_model}")
    elif api_backend == "gemini":
        import os as _os
        from google import genai as _genai
        _key = _os.environ.get("GEMINI_API_KEY")
        if not _key and Path(".env").exists():
            for _l in Path(".env").read_text().splitlines():
                if _l.startswith("GEMINI_API_KEY="):
                    _key = _l.split("=", 1)[1].strip()
        if not _key:
            raise SystemExit("GEMINI_API_KEY not set (env or .env)")
        api_client = _genai.Client(api_key=_key)
        model, processor = None, None
        print(f"API backend: gemini / {api_model}")
    elif api_backend == "anthropic":
        import os as _os
        import anthropic as _anthropic
        _key = _os.environ.get("ANTHROPIC_API_KEY")
        if not _key and Path(".env").exists():
            for _l in Path(".env").read_text().splitlines():
                if _l.startswith("ANTHROPIC_API_KEY="):
                    _key = _l.split("=", 1)[1].strip()
        if not _key:
            raise SystemExit("ANTHROPIC_API_KEY not set (env or .env)")
        api_client = _anthropic.Anthropic(api_key=_key)
        model, processor = None, None
        print(f"API backend: anthropic / {api_model}")
    else:
        api_client = None
        model, processor = _load_model_and_processor(checkpoint)
    print("Model loaded.\n")

    results = []
    total_time = 0.0

    # API runs: resume from a .partial dump (written every 100 rows and on
    # fatal abort) so a credit/quota outage doesn't burn completed calls.
    partial_path = Path(str(output_path) + ".partial")
    resumed_ids: set = set()
    if api_client is not None and partial_path.exists():
        try:
            for r in json.load(open(partial_path)):
                if "error" not in r:  # errored rows get retried
                    results.append(r)
                    resumed_ids.add(r["id"])
            print(f"Resuming {len(resumed_ids)} rows from {partial_path}")
        except Exception as e:
            print(f"Ignoring unreadable partial {partial_path}: {e}")
    consecutive_api_errors = 0

    def _dump_partial():
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    for i, sample in enumerate(samples):
        image_path = sample.get("image", "")
        instruction = sample.get("instruction", "")
        gt_response = sample.get("response", {})
        gt_score_raw = gt_response.get("score")
        gt_visual = gt_response.get("visual_reason", {})
        sample_id = sample.get("meta", {}).get("id", f"sample_{i}")
        source = sample.get("meta", {}).get("source", "")
        if sample_id in resumed_ids:
            continue

        # Channel-annotated eval sets (RichHF sidecars in meta.orig) get the
        # dual-channel prompt and per-channel scoring; the GT union grid keeps
        # the main metrics comparable with flat eval sets. the earlier benchmark records
        # carry no sidecars, so its protocol is byte-identical to before.
        orig_meta = sample.get("meta", {}).get("orig", {}) or {}
        gt_channels: Dict[str, Any] = {}
        if orig_meta.get("artifact_map_png") or orig_meta.get("misalign_map_png"):
            for name in ("artifact", "misalign"):
                problem, major = heatmap_png_to_grids(
                    orig_meta.get(f"{name}_map_png", "")
                )
                gt_channels[name] = {"problem": problem, "major": major}

        if gt_channels:
            gt_grid = np.clip(
                gt_channels["artifact"]["problem"] + gt_channels["misalign"]["problem"],
                0, 1,
            ).astype(np.uint8)
        elif gt_visual:
            gt_grid = gt_visual_to_grid(gt_visual)
        else:
            gt_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        if not Path(image_path).exists():
            print(f"  [{i+1}/{len(samples)}] SKIP {sample_id}: image not found")
            results.append({
                "id": sample_id, "source": source, "image": image_path,
                "gt_score": gt_score_raw, "gt_grid": gt_grid.tolist(),
                "error": f"image not found: {image_path}",
            })
            continue

        t0 = time.time()
        try:
            image = resize_to_train_res(Image.open(image_path).convert("RGB"))
            # Recover conditioning images (editing/reference tasks) exactly as the
            # builder did, so the model sees the same multi-image layout it trained
            # on. Empty for TIG/COCO/RichHF → single-image path.
            input_paths, refined = derive_input_images(Path(image_path))
            if refined:
                instruction = refined
            if no_cond_images:
                # χ0 intervention (tab:planned-context): withhold the
                # conditioning images while keeping the TEXT identical to the
                # χK run — isolates access to the input images only.
                input_paths = []
            input_images = [
                _resize_to(Image.open(p).convert("RGB"), MAX_PIXELS_REF)
                for p in input_paths
            ]
            # Per-source protocol override (P0-1): prompt each row exactly as
            # its source's TRAINING rows were prompted; fall back to the
            # global flags for sources without an entry.
            proto = (source_protocols or {}).get(source)
            if proto is not None:
                row_channels = bool(proto.get("channels", False)) or bool(gt_channels)
                row_with_score = bool(proto.get("with_score", True))
                row_axes = bool(proto.get("axes", False))
                row_score_only = bool(proto.get("score_only", False))
            else:
                row_channels = bool(gt_channels) or source in channel_sources
                row_with_score = with_score
                row_axes = dual_axis
                row_score_only = False
            _predict_kwargs = dict(
                input_images=input_images, max_new_tokens=max_new_tokens,
                channels=row_channels,
                with_score=row_with_score, score_last=score_last,
                axes=row_axes, score_only=row_score_only,
                # PAL4VST-style flat no-score sources keep 16-only targets in
                # hier training data — prompt-matched here too.
                hier=hier and gt_score_raw is not None,
            )
            if api_client is not None:
                _api_fn = {"gemini": _predict_grid_gemini,
                           "anthropic": _predict_grid_anthropic}.get(
                               api_backend, _predict_grid_api)
                response_text = _api_fn(
                    api_client, api_model, image, instruction, **_predict_kwargs)
            else:
                response_text = _predict_grid(
                    model, processor, image, instruction, **_predict_kwargs)
        except ApiCreditsExhausted as e:
            _dump_partial()
            print(f"\nFATAL: API credits exhausted at row {i+1}/{len(samples)}; "
                  f"{len(results)} rows saved to {partial_path}. "
                  f"Top up and rerun to resume.\n{e}")
            sys.exit(3)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ERROR {sample_id}: {e}")
            results.append({
                "id": sample_id, "source": source, "image": image_path,
                "gt_score": gt_score_raw, "gt_grid": gt_grid.tolist(), "error": str(e),
            })
            if api_client is not None:
                consecutive_api_errors += 1
                if consecutive_api_errors >= 5:
                    _dump_partial()
                    print(f"\nFATAL: {consecutive_api_errors} consecutive API "
                          f"errors; {len(results)} rows saved to {partial_path}. "
                          "Fix the cause and rerun to resume.")
                    sys.exit(3)
            continue
        elapsed = time.time() - t0
        total_time += elapsed
        consecutive_api_errors = 0
        if api_client is not None and (i + 1) % 25 == 0:
            _dump_partial()

        pred_grid = parse_grid_from_response(response_text)
        grid_parsed = pred_grid is not None
        pred_score = parse_score_from_response(response_text)
        pred_pq, pred_sc = parse_axis_scores(response_text)
        if pred_score is None and pred_pq is not None and pred_sc is not None:
            # VIEScore-style overall from the two axes.
            pred_score = float(np.sqrt(pred_pq * pred_sc))
        conf32 = None
        if hier:
            gt32 = (gt_visual_to_grid(gt_visual, grid_size=GRID_SIZE * 2)
                    if gt_visual else np.zeros((GRID_SIZE * 2, GRID_SIZE * 2), np.uint8))
            pred32 = parse_hier_from_response(response_text)
            conf32 = compute_cell_confusion(gt32, pred32)
        pred_channels = parse_channels_from_response(response_text)
        channel_conf: Dict[str, Dict[str, int]] = {}
        coverage_conf: Dict[str, int] = {}
        if gt_channels:
            zero = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            for name in ("artifact", "misalign"):
                pc = pred_channels.get(name, {}).get("problem", zero)
                channel_conf[name] = compute_cell_confusion(
                    gt_channels[name]["problem"], pc
                )
            # Coverage marks ("!"), i.e. cells >=90% inside the defect region.
            # Trained and RL-weighted but never scored on a held-out set before
            # ( M10); union over channels, same micro P/R/F1 treatment.
            gt_cov = np.clip(gt_channels["artifact"]["major"]
                             + gt_channels["misalign"]["major"], 0, 1).astype(np.uint8)
            pred_cov = np.clip(
                pred_channels.get("artifact", {}).get("major", zero)
                + pred_channels.get("misalign", {}).get("major", zero), 0, 1
            ).astype(np.uint8)
            coverage_conf = compute_cell_confusion(gt_cov, pred_cov)
        # Per-patch failure count for the summary schema.
        # A JSON grid either fully parses (0 failed) or not (all failed).
        parse_failed = 0 if grid_parsed else GRID_SIZE * GRID_SIZE

        # EvalMuse provides scores but NO localization annotation: an all-zero
        # GT there means "unlabeled", not "clean", so predicted cells must not
        # count as false positives (found post-eval_suite: 200 EM rows injected
        # 5712 FPs / 0 TPs into every model's localization metrics). COCO-real
        # keeps its grid metrics — clean-by-construction is a real label.
        has_loc_gt = bool(gt_channels) or bool(gt_visual) or source == "COCO-real"
        if has_loc_gt:
            grid_iou = compute_grid_iou(gt_grid, pred_grid)
            conf = compute_cell_confusion(gt_grid, pred_grid)
            precision, recall, f1 = prf_from_counts(conf["tp"], conf["fp"], conf["fn"])
        else:
            grid_iou = None
            conf = {"tp": None, "fp": None, "fn": None, "tn": None}
            precision = recall = f1 = None

        gt_cells = int(gt_grid.sum())
        pred_cells = int(pred_grid.sum()) if grid_parsed else -1
        iou_str = f"{grid_iou:.3f}" if grid_iou is not None else "n/a"
        f1_str = f"{f1:.3f}" if f1 is not None else "n/a"
        print(
            f"  [{i+1}/{len(samples)}] {sample_id}  "
            f"gt_cells={gt_cells}  pred_cells={pred_cells}  "
            f"iou={iou_str}  f1={f1_str}  parsed={grid_parsed}  t={elapsed:.1f}s"
        )

        results.append({
            "id": sample_id,
            "source": source,
            "instruction": instruction,
            "image": image_path,
            "gt_score": gt_score_raw,
            "pred_score": pred_score,
            "pred_pq": pred_pq,
            "pred_sc": pred_sc,
            "gt_axes": list(extract_axis_scores(sample)),
            "gt_grid": gt_grid.tolist(),
            "pred_grid": pred_grid.tolist() if grid_parsed else None,
            "raw_response": response_text,
            "grid_iou": round(grid_iou, 4) if grid_iou is not None else None,
            "cell_tp": conf["tp"],
            "cell_fp": conf["fp"],
            "cell_fn": conf["fn"],
            "cell_tn": conf["tn"],
            "cell_precision": round(precision, 4) if precision is not None else None,
            "cell_recall": round(recall, 4) if recall is not None else None,
            "cell_f1": round(f1, 4) if f1 is not None else None,
            "parse_failed_patches": parse_failed,
            "grid_parsed": grid_parsed,
            "pred_has_channels": bool(pred_channels),
            "channel_confusion": channel_conf or None,
            "coverage_confusion": coverage_conf or None,
            "cell32_confusion": conf32,
            "inference_time_s": round(elapsed, 2),
        })
        if verbalize_out:
            from verbalize import verbalize  # deferred: verbalize imports us
            results[-1]["explanation"] = verbalize(response_text)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    source_counts: Dict[str, int] = {}
    for r in results:
        src = r.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    summary: Dict[str, Any] = {
        "model": model_label,
        "checkpoint": checkpoint,
        "total_samples": len(samples),
        "scored_samples": len(valid),
        "errors": len(errors),
        "avg_inference_time_s": round(
            total_time / max(len(results) - len(errors), 1), 2
        ),
        "per_source_counts": source_counts,
    }

    if valid:
        # rows with localization GT (EvalMuse rows carry None cell fields)
        loc_rows = [r for r in valid if r.get("cell_tp") is not None]
        summary["n_localization_gt"] = len(loc_rows)
        iou_vals = [r["grid_iou"] for r in loc_rows if r.get("grid_iou") is not None]
        if iou_vals:
            summary["avg_grid_iou"] = round(sum(iou_vals) / len(iou_vals), 4)

        # NOTE: avg_grid_iou above is INFLATED by clean images where GT and pred
        # are both blank (scored 1.0). Below are the honest localization metrics:
        # problem-only IoU and micro-averaged cell precision/recall/F1, plus the
        # count of clean-correct freebies so the inflation is auditable.
        problem = [
            r for r in loc_rows
            if r.get("gt_grid") and any(any(row) for row in r["gt_grid"])
        ]
        if problem:
            summary["avg_grid_iou_problem_only"] = round(
                sum(r["grid_iou"] for r in problem) / len(problem), 4
            )

        tp = sum(r["cell_tp"] for r in loc_rows)
        fp = sum(r["cell_fp"] for r in loc_rows)
        fn = sum(r["cell_fn"] for r in loc_rows)
        mp, mr, mf = prf_from_counts(tp, fp, fn)
        summary["cell_precision_micro"] = round(mp, 4)
        summary["cell_recall_micro"] = round(mr, 4)
        summary["cell_f1_micro"] = round(mf, 4)
        if problem:
            summary["cell_f1_macro_problem"] = round(
                sum(r.get("cell_f1", 0.0) for r in problem) / len(problem), 4
            )
        summary["n_problem_gt"] = len(problem)
        summary["clean_correct"] = sum(
            1 for r in loc_rows
            if (r["cell_tp"] + r["cell_fp"] + r["cell_fn"]) == 0
        )

        # patch_parse_rate here = fraction of samples whose whole grid parsed.
        total_patches = len(valid) * GRID_SIZE * GRID_SIZE
        failed_patches = sum(r.get("parse_failed_patches", 0) for r in valid)
        summary["patch_parse_rate"] = round(
            (total_patches - failed_patches) / total_patches * 100, 1
        )
        summary["grid_parse_rate"] = round(
            sum(1 for r in valid if r.get("grid_parsed")) / len(valid) * 100, 1
        )

        # 32×32 micro P/R/F1 over samples evaluated with the hierarchical
        # prompt (subcell refinement).
        h32 = [r for r in valid if r.get("cell32_confusion")]
        if h32:
            t = sum(r["cell32_confusion"]["tp"] for r in h32)
            f = sum(r["cell32_confusion"]["fp"] for r in h32)
            n = sum(r["cell32_confusion"]["fn"] for r in h32)
            p32, r32, f32 = prf_from_counts(t, f, n)
            summary["cell32_samples"] = len(h32)
            summary["cell32_precision_micro"] = round(p32, 4)
            summary["cell32_recall_micro"] = round(r32, 4)
            summary["cell32_f1_micro"] = round(f32, 4)

        # Per-channel micro P/R/F1, aggregated over the samples whose GT
        # carries channel sidecars (0 samples on the earlier benchmark → keys absent).
        chan_samples = [r for r in valid if r.get("channel_confusion")]
        if chan_samples:
            summary["channel_samples"] = len(chan_samples)
            for name in ("artifact", "misalign"):
                ctp = sum(r["channel_confusion"][name]["tp"] for r in chan_samples)
                cfp = sum(r["channel_confusion"][name]["fp"] for r in chan_samples)
                cfn = sum(r["channel_confusion"][name]["fn"] for r in chan_samples)
                cp, cr, cf = prf_from_counts(ctp, cfp, cfn)
                summary[f"{name}_precision_micro"] = round(cp, 4)
                summary[f"{name}_recall_micro"] = round(cr, 4)
                summary[f"{name}_f1_micro"] = round(cf, 4)

            cov = [r for r in valid if r.get("coverage_confusion")]
            if cov:
                vtp = sum(r["coverage_confusion"]["tp"] for r in cov)
                vfp = sum(r["coverage_confusion"]["fp"] for r in cov)
                vfn = sum(r["coverage_confusion"]["fn"] for r in cov)
                vp, vr, vf = prf_from_counts(vtp, vfp, vfn)
                summary["coverage_samples"] = len(cov)
                summary["coverage_precision_micro"] = round(vp, 4)
                summary["coverage_recall_micro"] = round(vr, 4)
                summary["coverage_f1_micro"] = round(vf, 4)

        scored = [r for r in valid if r.get("gt_score") is not None]
        if scored:
            gt_sc = [r["gt_score"] * 10 for r in scored]
            summary["avg_gt_score"] = round(sum(gt_sc) / len(gt_sc), 2)

        # Score-head correlation with the human GT score. COCO-real is
        # excluded (all GT scores are 1.0 → zero variance). n=0 for pre-v3
        # models, which never emit a score line.
        sc = [r for r in valid
              if r.get("pred_score") is not None and r.get("gt_score") is not None
              and r.get("source") != "COCO-real"]
        summary["score_pred_n"] = len(sc)
        if len(sc) >= 3:
            pred = np.array([r["pred_score"] for r in sc], dtype=float)
            gt = np.array([r["gt_score"] * 10 for r in sc], dtype=float)
            if pred.std() > 0 and gt.std() > 0:
                summary["score_pearson"] = round(float(np.corrcoef(pred, gt)[0, 1]), 4)
                pr = rank_avg(pred)
                gr = rank_avg(gt)
                summary["score_spearman"] = round(float(np.corrcoef(pr, gr)[0, 1]), 4)

        # Per-axis correlations (dual-axis models): pred pq/sc vs the
        # source-native GT axes, non-COCO (constant GT there).
        for axis, pred_key in (("pq", "pred_pq"), ("sc", "pred_sc")):
            idx = 0 if axis == "pq" else 1
            ax = [r for r in valid
                  if r.get(pred_key) is not None
                  and r.get("gt_axes") and r["gt_axes"][idx] is not None
                  and r.get("source") != "COCO-real"]
            if len(ax) >= 3:
                pred = np.array([r[pred_key] for r in ax], dtype=float)
                gt = np.array([r["gt_axes"][idx] * 10 for r in ax], dtype=float)
                if pred.std() > 0 and gt.std() > 0:
                    pr = rank_avg(pred)
                    gr = rank_avg(gt)
                    summary[f"{axis}_n"] = len(ax)
                    summary[f"{axis}_pearson"] = round(float(np.corrcoef(pred, gt)[0, 1]), 4)
                    summary[f"{axis}_spearman"] = round(float(np.corrcoef(pr, gr)[0, 1]), 4)

    # Serialize the full eval protocol so every result JSON is traceable to
    # the exact flags that produced it (: untraceable protocols).
    summary["eval_protocol"] = {
        "checkpoint": str(checkpoint),
        "eval_data": str(eval_data_path),
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "channel_sources": sorted(channel_sources) if channel_sources else [],
        "with_score": with_score,
        "score_last": score_last,
        "dual_axis": dual_axis,
        "hier": hier,
        "source_protocols": source_protocols or {},
        "spearman": "tie_averaged_ranks",
    }
    output_data: Dict[str, Any] = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    partial_path.unlink(missing_ok=True)

    print(f"\nResults saved to {output_path}")
    print(f"Summary:\n{json.dumps(summary, indent=2)}")
    return output_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VIEScore2 eval: full image → 16×16 JSON grid"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-data", type=Path, default=Path("eval_samples/eval.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max tokens to generate for the JSON grid (default 1024)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N samples (0 = all). Useful "
                             "for fast HPO trials.")
    parser.add_argument("--channel-sources", default=None,
                        help="Comma-separated meta.source values to prompt with "
                             "the dual-channel format (e.g. \"RichHF-18K\"). Use "
                             "for inspector-class models whose training saw those "
                             "sources only with channel prompts.")
    parser.add_argument("--api-backend", choices=["openai", "gemini", "anthropic"], default=None,
                        help="Evaluate a closed model through an API instead "
                             "of a local checkpoint (). Prompts, "
                             "image order/resolutions, parser and metrics are "
                             "identical to the local path.")
    parser.add_argument("--api-model", default="gpt-5.6-terra",
                        help="API model id (with --api-backend).")
    parser.add_argument("--no-cond-images", action="store_true",
                        help="Withhold conditioning images (χ0 intervention, "
                             "tab:planned-context); prompt text unchanged.")
    parser.add_argument("--no-score-prompt", action="store_true",
                        help="Drop the score request from prompts (for models "
                             "trained WITHOUT the score line, e.g. the noscore "
                             "ablation — prompt-matching principle).")
    parser.add_argument("--score-last", action="store_true",
                        help="Ask for cells first and the score line last (for "
                             "the order-ablation model trained that way).")
    parser.add_argument("--dual-axis", action="store_true",
                        help="Ask for pq:/sc: dual-axis scores (for models "
                             "trained with --dual-axis data); overall score = "
                             "sqrt(pq*sc), per-axis correlations added to the "
                             "summary.")
    parser.add_argument("--verbalize", action="store_true",
                        help="Attach the deterministic faithful explanation "
                             "(verbalize.py) to each per-sample record.")
    parser.add_argument("--hier", action="store_true",
                        help="Ask for 2x2 subcell refinement (for models "
                             "trained with --hierarchical data); adds 32x32 "
                             "micro metrics. No-score samples stay flat "
                             "(prompt-matched, like training).")
    parser.add_argument("--protocol-config", type=Path, default=None,
                        help="JSON mapping meta.source → per-source prompt "
                             "schema {with_score, axes, channels, score_only}; "
                             "listed sources are prompted as their TRAINING "
                             "rows were, overriding the global flags "
                             "(configs/eval_protocol.json). Keys starting "
                             "with '_' are comments.")
    args = parser.parse_args()

    source_protocols = None
    if args.protocol_config:
        raw = json.loads(Path(args.protocol_config).read_text(encoding="utf-8"))
        source_protocols = {k: v for k, v in raw.items() if not k.startswith("_")}
        print(f"Per-source protocols from {args.protocol_config}: "
              f"{sorted(source_protocols)}")

    label = args.label or (
        args.checkpoint.replace("/", "_")
        if not Path(args.checkpoint).exists()
        else Path(args.checkpoint).name
    )
    output = args.output or Path(f"eval_results/{label}.json")

    run_eval(
        checkpoint=args.checkpoint,
        eval_data_path=args.eval_data,
        output_path=output,
        model_label=label,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        channel_sources=set(args.channel_sources.split(",")) if args.channel_sources else None,
        with_score=not args.no_score_prompt,
        score_last=args.score_last,
        dual_axis=args.dual_axis,
        hier=args.hier,
        verbalize_out=args.verbalize,
        source_protocols=source_protocols,
        no_cond_images=args.no_cond_images,
        api_backend=args.api_backend,
        api_model=args.api_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build axolotl-format SFT data for the patch-JSON task.

Unlike the patch-classification approach (one 1/16×1/16 crop per sample,
256 forward passes per image), here each training sample is ONE full image.
The model's task: given the whole image + generation prompt, output a single
16×16 JSON array where 1 = problematic patch and 0 = clean patch.

GT labels come from the same visual annotations used elsewhere
(mask/heatmap/polygon/svg/bbox), rasterised to a 16×16 grid; the grid is
serialised as the assistant target.

This is a single-pass builder (no per-cell class balancing is needed because
each sample is a whole grid, not an individual cell):
  Pass 1 – stream the JSONL, resize+copy each image, emit one chat sample
  Pass 2 – shuffle + split → train.json + val.json

Usage:
    python viescore2/build_sft_data.py \\
        --input      sft_samples/sft_unified.jsonl \\
        --output-dir $DATA_ROOT/viescore2_data \\
        --data-root  data \\
        --seed       42 -v
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

GRID_SIZE = 16
MAX_PIXELS = 384 * 384  # cap visual tokens for Qwen3-VL (~192 tokens); lowered
                        # from 512^2 to cut training activation memory so the
                        # full finetune fits on 4x80GB without optimizer offload.
MAX_PIXELS_REF = 256 * 256  # input/reference images are CONTEXT (not the eval
                            # target); cap them smaller (~85 tokens each) so a
                            # multi-reference sample (up to 5 inputs + generated)
                            # stays within the sequence budget.

SYSTEM_PROMPT = (
    "You are an expert image quality evaluator.\n\n"
    "You may be shown one or more INPUT images — an original image to be edited "
    "and/or reference images that were given to the image generator — followed "
    "by the GENERATED image to be evaluated. The GENERATED image is ALWAYS the "
    "LAST image; earlier images are referred to as \"image 1\", \"image 2\", etc. "
    "in reading order.\n"
    "When input images are present, judge the generated image RELATIVE to them: "
    "whether the requested edit was applied correctly, whether content that "
    "should have been preserved was preserved, and whether referenced subjects "
    "are reproduced faithfully. When no input image is present, judge the "
    "generated image against the text prompt alone.\n\n"
    "The GENERATED image is divided into a 16×16 grid of cells (row 1 is the "
    "top, column 1 is the left).\n"
    "For every cell of the GENERATED image decide whether it contains visible "
    "quality problems: artifacts, distortions, blurriness, missing or malformed "
    "content, color abnormalities, unrealistic content, or — for edits — a "
    "failed/incorrect edit or a wrongly altered region.\n\n"
    "FIRST output one line with the overall quality score of the generated "
    "image in the form \"score: <0-10>/10\" (10 = flawless, 0 = unusable).\n"
    "THEN list ONLY the cells that HAVE problems, one line per affected row, in "
    "the form \"r<row>: <col>,<col>,...\" using 1-based indices (rows and "
    "columns range from 1 to 16). Emit a line only for rows that contain at "
    "least one problematic cell, in increasing row order. If the image has no "
    "problems at all, output exactly \"none\" instead of the cell lines.\n\n"
    "When the request asks for separate problem CHANNELS, report the cells in "
    "two sections instead of one list: a header line \"artifact:\" followed by "
    "the rows of cells with visual artifacts (distortions, malformed or "
    "implausible content, rendering errors), then a header line \"misalign:\" "
    "followed by the rows of cells whose content contradicts the prompt "
    "(wrong, missing or extra objects/attributes). Within each section use the "
    # NOTE ( M10): the "!" mark is NOT human-rated severity. The
    # released RichHF maps are binary, so major = (resized cell value >=
    # major_threshold), i.e. the cell is almost entirely inside the defect
    # region — a COVERAGE property. The wording below says "severe" because
    # every existing checkpoint was trained with this exact string and
    # run_eval imports this same prompt; changing it now would put evaluation
    # off-distribution from training. Rename it at the next training
    # generation, and call it a coverage mark everywhere else (metrics are
    # already reported as coverage_*).
    "same \"r<row>: <col>,...\" lines, appending \"!\" to a column number when "
    "that cell's problem is severe (e.g. \"r5: 8,9!,10\"). Write \"none\" "
    "under a section with no problems.\n\n"
    "When the request asks for TWO scores instead of one, rate perceptual "
    "quality (freedom from artifacts/distortions) as \"pq: <0-10>/10\" and "
    "semantic consistency with the prompt as \"sc: <0-10>/10\", one line "
    "each, before the cell lines."
)


# ── GT annotation → 16×16 binary grid ────────────────────────────────────────

def gt_visual_to_grid(
    visual_reason: Dict[str, Any],
    data_root: str = ".",
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
            path = Path(data)
            if not path.is_absolute():
                path = Path(data_root) / data
            try:
                img = Image.open(path).convert("L")
                arr = np.array(img, dtype=np.float32)
                # Value-range adaptive: {0,255} masks normalise; {0,1}-pixel
                # labels (e.g. PAL4VST) pass through — /255 would zero them.
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


def extract_axis_scores(sample: Dict[str, Any]) -> tuple:
    """VIEScore-style dual axes from source-native sub-scores, both in [0,1]:
    ``(pq, sc)`` — perceptual quality (artifacts/aesthetics) and semantic
    consistency (prompt alignment). Returns ``(None, None)`` when the source
    has no per-axis labels (PAL4VST; malformed records). The axes map 1:1
    onto the output channels: PQ ↔ artifact cells, SC ↔ misalign cells."""
    src = sample.get("meta", {}).get("source", "")
    orig = sample.get("meta", {}).get("orig", {}) or {}
    if src == "RichHF-18K":
        art = orig.get("artifact_score")
        aes = orig.get("aesthetics_score")
        mis = orig.get("misalignment_score")
        if art is None or mis is None:
            return None, None
        pq = (float(art) + float(aes)) / 2 if aes is not None else float(art)
        return pq, float(mis)
    if src.startswith("ImagenWorld"):
        rs = orig.get("raw_scores") or {}
        try:
            art = float(rs["artifact"])
            aq = float(rs["aesthetic_quality"])
            pr = float(rs["prompt_relevance"])
        except (KeyError, TypeError, ValueError):
            return None, None
        # 1-5 Likert → [0,1]
        return ((art - 1) / 4 + (aq - 1) / 4) / 2, (pr - 1) / 4
    if src == "COCO-real":
        return 1.0, 1.0
    return None, None


def heatmap_png_to_grids(
    png_path: str,
    data_root: str = ".",
    grid_size: int = GRID_SIZE,
    major_threshold: float = 0.9,
) -> tuple:
    """Rasterise one RichHF channel sidecar (continuous heatmap saved as an
    8-bit grayscale PNG) into ``(problem, major)`` 16×16 binary grids.

    The *problem* decision reproduces the legacy pipeline bit-for-bit
    (binarise at 0.5 full-res, LANCZOS-resize, cell > 0.5) so channel grids
    stay distribution-compatible with the frozen eval GT. *major* marks the
    problem cells whose defect COVERAGE (the resized map value; the RichHF
    release maps are binary, so the resized value is the fraction of the
    cell inside the defect region) reaches ``major_threshold`` — i.e. cells
    that are essentially fully defective, vs partially-touched boundary
    cells. Measured on RichHF-train: 0.9 → 38% of problem cells are major.
    """
    from PIL import Image

    zeros = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not png_path:
        return zeros, zeros
    path = Path(png_path)
    if not path.is_absolute():
        candidate = Path(data_root) / png_path
        path = candidate if candidate.exists() else path
    try:
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    except Exception:
        log.warning(f"Failed to load channel sidecar {png_path}")
        return zeros, zeros

    binary = (arr > 0.5).astype(np.float32)
    small_bin = np.array(
        Image.fromarray((binary * 255).astype(np.uint8)).resize(
            (grid_size, grid_size), Image.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0
    problem = (small_bin > 0.5).astype(np.uint8)

    small_raw = np.array(
        Image.fromarray((arr * 255).astype(np.uint8)).resize(
            (grid_size, grid_size), Image.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0
    major = ((small_raw >= major_threshold) & (problem > 0)).astype(np.uint8)
    return problem, major


# ── Prompt / target formatting ───────────────────────────────────────────────

SCORE_LAST_SUFFIX = (
    ' Finally, after the cell lines, output the overall quality score on the '
    'last line as "score: <0-10>/10".'
)


def build_user_prompt(
    instruction: str, n_inputs: int = 0, grid_size: int = GRID_SIZE,
    with_score: bool = True, channels: bool = False, score_last: bool = False,
    axes: bool = False, hier: bool = False,
) -> str:
    """``with_score=False`` is for sources WITHOUT a human quality score
    (e.g. PAL4VST): the prompt then only asks for the problem cells, matching
    a target that has no ``score:`` line — never ask for what the label can't
    supervise.

    ``channels=True`` is for sources with SEPARATE artifact / misalignment
    annotations (RichHF): the prompt then asks for the two-section output with
    \"!\" severity marks. Sources whose masks conflate the two stay on the
    single flat list — same principle as ``with_score``.

    ``score_last=True`` (order ablation, ImageDoctor-style predict-then-
    summarize) asks for the cells FIRST and the score line LAST; built as the
    score-free prompt + a fixed suffix so data converters can reproduce it
    exactly by construction."""
    if score_last and with_score:
        return build_user_prompt(
            instruction, n_inputs, grid_size, with_score=False,
            channels=channels, hier=hier,
        ) + SCORE_LAST_SUFFIX
    hier_part = (
        " For each listed column, append in square brackets which quarters "
        "of that cell are affected (1=top-left, 2=top-right, 3=bottom-left, "
        "4=bottom-right), e.g. \"r5: 8[1,3],9\"; a bare column means the "
        "whole cell is affected."
    ) if hier else ""
    if n_inputs > 0:
        labels = ", ".join(f"image {i + 1}" for i in range(n_inputs))
        is_are = "is" if n_inputs == 1 else "are"
        intro = (
            f"The first {n_inputs} image{'' if n_inputs == 1 else 's'} ({labels}) "
            f"{is_are} the input/reference image{'' if n_inputs == 1 else 's'} that "
            f"{'was' if n_inputs == 1 else 'were'} given to the generator. "
            f"The LAST image is the GENERATED image to evaluate; it is divided "
            f"into a {grid_size}×{grid_size} grid of equal cells.\n"
        )
    else:
        intro = (
            f"This generated image is divided into a {grid_size}×{grid_size} grid "
            f"of equal cells.\n"
        )
    if axes and with_score:
        score_part = (
            "First rate the GENERATED image on two axes — perceptual quality "
            "(artifacts, distortions, rendering flaws) as \"pq: <0-10>/10\" and "
            "semantic consistency with the prompt as \"sc: <0-10>/10\", one "
            "line each — then list"
        )
    else:
        score_part = (
            "First give the overall quality score as \"score: <0-10>/10\", then list"
            if with_score else "List"
        )
    if channels:
        return (
            intro +
            f'Generation/edit prompt: "{instruction}"\n'
            f"{score_part} the problem cells of the GENERATED image in two "
            f"channels: a \"artifact:\" section for cells with visual "
            f"artifacts, then a \"misalign:\" section for cells whose content "
            f"contradicts the prompt. In each section use one line per "
            f"affected row as \"r<row>: <cols>\", append \"!\" to severely "
            f"affected columns, and write \"none\" under an empty section."
            + hier_part
        )
    return (
        intro +
        f'Generation/edit prompt: "{instruction}"\n'
        f"{score_part} the cells of the GENERATED image that contain visible "
        f"quality problems, one line per affected row as \"r<row>: <cols>\". "
        f"Output \"none\" instead of cell lines if there are no problems."
        + hier_part
    )


def grid_to_target(grid: np.ndarray) -> str:
    """Serialise a 16×16 grid as a SPARSE per-row enumeration of problem cells.

    Only rows containing at least one problem cell get a line; columns are
    1-based and comma-separated::

        r5: 8,9
        r6: 8,9,10,11

    A fully clean grid serialises to the single token ``none``. Unlike a dense
    16×16 array of 0/1, this target contains NO "clean" (0) tokens at all, so
    the autoregressive loss is spent entirely on the positive signal and the
    model cannot minimise it by predicting an empty grid.
    """
    lines = []
    for r in range(grid.shape[0]):
        cols = [c + 1 for c in range(grid.shape[1]) if grid[r][c]]
        if cols:
            lines.append(f"r{r + 1}: " + ",".join(str(c) for c in cols))
    return "\n".join(lines) if lines else "none"


def _hier_col_tokens(
    g32: np.ndarray, r: int, cols: List[int], major: Optional[np.ndarray] = None,
) -> List[str]:
    """Column tokens for one row of a hierarchical target: bare column =
    whole cell defective (all four 2×2 subcells at 32×32), ``8[1,3]`` = only
    those quarters (1=TL, 2=TR, 3=BL, 4=BR). A problem cell whose subcells
    all miss the 32-level threshold (boundary case) stays bare. ``!``
    severity composes before the bracket: ``8![1,3]``."""
    toks = []
    for c in cols:
        subs = [s for s in range(4) if g32[r * 2 + s // 2, c * 2 + s % 2]]
        tok = str(c + 1) + ("!" if major is not None and major[r][c] else "")
        if 0 < len(subs) < 4:
            tok += "[" + ",".join(str(s + 1) for s in subs) + "]"
        toks.append(tok)
    return toks


def grid_to_target_hier(grid: np.ndarray, g32: np.ndarray) -> str:
    """Hierarchical flat target: 16×16 rows with 2×2 subcell refinement."""
    lines = []
    for r in range(grid.shape[0]):
        cols = [c for c in range(grid.shape[1]) if grid[r][c]]
        if cols:
            lines.append(f"r{r + 1}: " + ",".join(_hier_col_tokens(g32, r, cols)))
    return "\n".join(lines) if lines else "none"


def channel_to_lines_hier(problem: np.ndarray, major: np.ndarray, p32: np.ndarray) -> str:
    lines = []
    for r in range(problem.shape[0]):
        cols = [c for c in range(problem.shape[1]) if problem[r][c]]
        if cols:
            lines.append(f"r{r + 1}: " + ",".join(_hier_col_tokens(p32, r, cols, major)))
    return "\n".join(lines) if lines else "none"


def channels_to_target_hier(channel_grids: Dict[str, tuple], hier_grids: Dict[str, np.ndarray]) -> str:
    parts = []
    for name in ("artifact", "misalign"):
        problem, major = channel_grids[name]
        parts.append(f"{name}:\n{channel_to_lines_hier(problem, major, hier_grids[name])}")
    return "\n".join(parts)


def channel_to_lines(problem: np.ndarray, major: np.ndarray) -> str:
    """Serialise ONE channel like :func:`grid_to_target`, with ``!`` appended
    to the columns whose problem is severe (``r5: 8,9!,10``)."""
    lines = []
    for r in range(problem.shape[0]):
        cols = [
            f"{c + 1}!" if major[r][c] else f"{c + 1}"
            for c in range(problem.shape[1]) if problem[r][c]
        ]
        if cols:
            lines.append(f"r{r + 1}: " + ",".join(cols))
    return "\n".join(lines) if lines else "none"


def channels_to_target(channel_grids: Dict[str, tuple]) -> str:
    """Serialise the dual-channel target::

        artifact:
        r5: 8,9!
        misalign:
        none

    ``channel_grids`` maps channel name → ``(problem, major)`` grids. Both
    sections are always emitted (RichHF annotates both channels on every
    sample, so an absent map means clean, not unknown).
    """
    parts = []
    for name in ("artifact", "misalign"):
        problem, major = channel_grids[name]
        parts.append(f"{name}:\n{channel_to_lines(problem, major)}")
    return "\n".join(parts)


def make_sample(
    image_ref: str,
    instruction: str,
    grid: np.ndarray,
    input_refs: Optional[List[str]] = None,
    score: Optional[float] = None,
    channel_grids: Optional[Dict[str, tuple]] = None,
    axis_scores: Optional[tuple] = None,
    hier_grids: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Build one chat sample.

    ``input_refs`` are the conditioning images (original-to-edit and/or
    references) shown BEFORE the generated image. The generated image
    (``image_ref``) is always last, and the 16×16 grid target applies to it.

    ``score`` is the human GT quality score in [0, 1]; when given, the target
    starts with a ``score: <0-10>/10`` line BEFORE the grid lines, so ONE
    generative pass yields both a VIEScore-comparable scalar and the
    fine-grained localization (the score line is parseable & verifiable, and
    can later serve as an RL reward signal).

    ``channel_grids`` (channel name → ``(problem, major)``) switches the
    target to the dual-channel sections with severity marks; the matching
    channel-aware user prompt is used and ``grid`` is ignored for the target.
    """
    input_refs = input_refs or []
    if channel_grids:
        target = (channels_to_target_hier(channel_grids, hier_grids)
                  if hier_grids else channels_to_target(channel_grids))
    else:
        target = (grid_to_target_hier(grid, hier_grids["flat"])
                  if hier_grids else grid_to_target(grid))
    use_axes = axis_scores is not None and axis_scores[0] is not None
    if use_axes:
        pq, sc = axis_scores
        target = (f"pq: {round(float(pq) * 10)}/10\n"
                  f"sc: {round(float(sc) * 10)}/10\n" + target)
    elif score is not None:
        target = f"score: {round(float(score) * 10)}/10\n" + target
    user_content: List[Dict[str, Any]] = [
        {"type": "image", "path": ref} for ref in input_refs
    ]
    user_content.append({"type": "image", "path": image_ref})
    user_content.append(
        {"type": "text", "text": build_user_prompt(
            instruction, len(input_refs), channels=channel_grids is not None,
            axes=use_axes, hier=hier_grids is not None,
        )}
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            },
        ]
    }


def derive_input_images(generated_path: Path) -> tuple[List[Path], Optional[str]]:
    """Recover the conditioning images for an ImagenWorld editing/reference sample.

    The generated image lives at ``<cond>/outputs/<model>/out.png``; the images
    given to the generator are in ``<cond>/input/`` and listed (in order) by
    ``metadata.json``'s ``cond_images``. Returns ``(ordered_input_paths,
    prompt_refined_or_None)``.

    Returns ``([], None)`` for text-only generation (TIG) and for COCO/RichHF
    samples, whose paths have no sibling ``input/`` directory, so those samples
    stay single-image exactly as before.
    """
    try:
        cond = generated_path.parents[2]  # out.png → <model> → outputs → <cond>
    except IndexError:
        return [], None
    input_dir = cond / "input"
    if not input_dir.is_dir():
        return [], None

    meta: Optional[Dict[str, Any]] = None
    meta_path = input_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None

    paths: List[Path] = []
    cond_images = (meta or {}).get("cond_images")
    if isinstance(cond_images, list):
        for name in cond_images:
            p = input_dir / str(name)
            if p.exists():
                paths.append(p)
    if not paths:  # fall back to any image files in input/
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)

    refined = (meta or {}).get("prompt_refined")
    refined = refined.strip() if isinstance(refined, str) and refined.strip() else None
    return paths, refined


# ── Image loading ────────────────────────────────────────────────────────────

def resize_and_save(src: Path, dst: Path, max_pixels: int = MAX_PIXELS) -> bool:
    """Copy ``src`` into ``dst``, downscaling if above ``max_pixels``."""
    from PIL import Image

    if dst.exists():
        return True
    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        img.save(dst)
        return True
    except Exception as e:
        log.warning(f"Failed to load/resize {src}: {e}")
        return False


# ── Build ────────────────────────────────────────────────────────────────────

_ANNOTATOR_RE = re.compile(r"_annotator\d+$")
# ImagenWorld case = the sample dir above outputs/<generator>/out.png. Different
# generators of one case share the source image + edit instruction, so training
# on one and evaluating on another is (source+prompt)-level leakage even though
# the pixels differ (: 4 such near-duplicate cases survived image-level
# exclusion). We hold out the whole case.
_IW_CASE_RE = re.compile(r"/([A-Z]+_[A-Z]+_\d+)/outputs/[^/]+/out\.\w+$")


def imagenworld_case(path: str):
    m = _IW_CASE_RE.search(path or "")
    return m.group(1) if m else None


def load_exclude_ids(eval_paths: str) -> set:
    """Collect holdout keys for every sample in one or more eval sets
    (comma-separated jsonl/json paths). Keys: meta.id, the id with any
    ``_annotatorN`` suffix stripped, the resolved image path, and (for
    ImagenWorld) the case id ``CASE::<case>``. Id-only exclusion is NOT
    image-level holdout: ImagenWorld stores the same image under one id per
    annotator ( F1), and different generators of one case are
    (source+prompt)-level twins. Match training samples via sample_excluded()."""
    keys: set = set()
    for eval_path in str(eval_paths).split(","):
        eval_path = eval_path.strip()
        if not eval_path:
            continue
        p = Path(eval_path)
        if not p.exists():
            log.warning(f"--exclude-eval file not found: {eval_path} (no holdout applied)")
            continue
        text = p.read_text(encoding="utf-8").strip()
        records = []
        if text.startswith("["):
            records = json.loads(text)
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        n_ids = n_imgs = n_case = 0
        for s in records:
            sid = s.get("meta", {}).get("id")
            if sid:
                keys.add(sid)
                keys.add(_ANNOTATOR_RE.sub("", sid))
                n_ids += 1
            img = s.get("image", "")
            if img:
                keys.add(str(Path(img).resolve()))
                n_imgs += 1
                case = imagenworld_case(img)
                if case:
                    keys.add(f"CASE::{case}")
                    n_case += 1
        log.info(f"Holdout: {n_ids} ids + {n_imgs} image paths + {n_case} IW cases "
                 f"from {eval_path}")
    return keys


def sample_excluded(sample: dict, keys: set) -> bool:
    """True if a pool record matches any holdout key: exact id, annotator-
    stripped id, resolved image path, or ImagenWorld case id."""
    if not keys:
        return False
    sid = sample.get("meta", {}).get("id")
    if sid and (sid in keys or _ANNOTATOR_RE.sub("", sid) in keys):
        return True
    img = sample.get("image", "")
    if img:
        if str(Path(img).resolve()) in keys:
            return True
        case = imagenworld_case(img)
        if case and f"CASE::{case}" in keys:
            return True
    return False


def build_samples(
    input_path: str,
    output_dir: Path,
    data_root: str,
    drop_empty_grids: bool,
    verbose: bool,
    exclude_ids: Optional[set] = None,
    major_threshold: float = 0.6,
    dual_axis: bool = False,
    hierarchical: bool = False,
) -> tuple[List[Dict], List[Dict]]:
    """Return ``(problem_samples, clean_samples)`` kept separate so the caller
    can downsample the (majority) clean set to a target fraction."""
    images_dir = output_dir / "images"
    if not images_dir.exists():  # exists() follows symlinks; mkdir(exist_ok)
        images_dir.mkdir(parents=True)  # raises on a symlinked dir
    exclude_ids = exclude_ids or set()

    problem_samples: List[Dict] = []
    clean_samples: List[Dict] = []
    skipped = 0
    empty_dropped = 0
    excluded_eval = 0
    multi_image = 0
    channel_count = 0
    chan_problem_cells = 0
    chan_major_cells = 0

    with open(input_path, encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # Hold out eval samples so training never sees them (no leakage) —
            # matched at id AND image level (annotator twins,  F1).
            if exclude_ids and sample_excluded(sample, exclude_ids):
                excluded_eval += 1
                continue

            image_rel = sample.get("image", "")
            image_path = Path(image_rel)
            if not image_path.is_absolute():
                candidate = Path(data_root) / image_rel
                if candidate.exists():
                    image_path = candidate
            if not image_rel or not image_path.exists():
                skipped += 1
                continue

            resp = sample.get("response", {})
            if resp.get("score") is None:
                skipped += 1
                continue

            instruction = sample.get("instruction", "")

            # RichHF samples carrying raw per-channel heatmap sidecars get the
            # dual-channel + severity target; everything else (and any legacy
            # jsonl without sidecars) keeps the flat single-list target.
            orig_meta = sample.get("meta", {}).get("orig", {}) or {}
            has_sidecar = bool(
                orig_meta.get("artifact_map_png") or orig_meta.get("misalign_map_png")
            )
            channel_grids: Optional[Dict[str, tuple]] = None
            hier_grids: Optional[Dict[str, np.ndarray]] = None
            if has_sidecar:
                channel_grids = {
                    name: heatmap_png_to_grids(
                        orig_meta.get(f"{key}_map_png", ""),
                        data_root=data_root,
                        major_threshold=major_threshold,
                    )
                    for name, key in (("artifact", "artifact"), ("misalign", "misalign"))
                }
                grid = np.clip(
                    channel_grids["artifact"][0] + channel_grids["misalign"][0], 0, 1
                ).astype(np.uint8)
                if hierarchical:
                    hier_grids = {
                        name: heatmap_png_to_grids(
                            orig_meta.get(f"{name}_map_png", ""),
                            data_root=data_root, grid_size=GRID_SIZE * 2,
                            major_threshold=major_threshold,
                        )[0]
                        for name in ("artifact", "misalign")
                    }
            else:
                visual = resp.get("visual_reason", {})
                grid = gt_visual_to_grid(visual, data_root=data_root)
                if hierarchical:
                    hier_grids = {"flat": gt_visual_to_grid(
                        visual, data_root=data_root, grid_size=GRID_SIZE * 2
                    )}

            if drop_empty_grids and int(grid.sum()) == 0:
                empty_dropped += 1
                continue

            # Recover conditioning images (original-to-edit / references) for
            # editing & reference-based tasks. Empty for TIG/COCO/RichHF.
            input_paths, refined = derive_input_images(image_path)
            if refined:
                # The numbered prompt ("Edit image 1 ... using image 2") matches
                # the multi-image layout, so prefer it when inputs are present.
                instruction = refined

            new_img_path = images_dir / f"{i:06d}_{image_path.name}"
            if not resize_and_save(image_path, new_img_path):
                skipped += 1
                continue

            input_refs: List[str] = []
            for j, ip in enumerate(input_paths):
                dst = images_dir / f"{i:06d}_in{j}_{ip.name}"
                if resize_and_save(ip, dst, max_pixels=MAX_PIXELS_REF):
                    input_refs.append(str(dst.resolve()))

            if input_refs:
                multi_image += 1
            if channel_grids:
                channel_count += 1
                for problem, major in channel_grids.values():
                    chan_problem_cells += int(problem.sum())
                    chan_major_cells += int(major.sum())
            axis_scores = extract_axis_scores(sample) if dual_axis else None
            sample_out = make_sample(
                str(new_img_path.resolve()), instruction, grid,
                input_refs=input_refs, score=float(resp["score"]),
                channel_grids=channel_grids, axis_scores=axis_scores,
                hier_grids=hier_grids,
            )
            if int(grid.sum()) > 0:
                problem_samples.append(sample_out)
            else:
                clean_samples.append(sample_out)

            if verbose and (i + 1) % 500 == 0:
                kept = len(problem_samples) + len(clean_samples)
                log.debug(f"  line {i+1}: kept={kept} skipped={skipped}")

    log.info(
        f"Built {len(problem_samples) + len(clean_samples)} samples "
        f"({len(problem_samples)} with problems, {len(clean_samples)} clean; "
        f"{multi_image} multi-image with conditioning inputs), "
        f"skipped {skipped}, dropped {empty_dropped} empty grids, "
        f"excluded {excluded_eval} held-out eval samples"
    )
    if channel_count:
        major_frac = chan_major_cells / chan_problem_cells if chan_problem_cells else 0.0
        log.info(
            f"Dual-channel: {channel_count} samples with channel sidecars; "
            f"{chan_problem_cells} problem cells of which {chan_major_cells} "
            f"major ({major_frac:.1%}, threshold {major_threshold})"
        )
    return problem_samples, clean_samples


def shuffle_and_split(
    converted: List[Dict],
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rng.shuffle(converted)

    n_val = max(1, int(len(converted) * val_ratio)) if converted else 0
    val_samples = converted[:n_val]
    train_samples = converted[n_val:]

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    log.info(f"Train: {len(train_samples)},  Val: {len(val_samples)}")
    log.info(f"Written to {output_dir}/train.json and {output_dir}/val.json")


def downsample_clean(
    problem_samples: List[Dict],
    clean_samples: List[Dict],
    empty_target_frac: float,
    seed: int,
) -> List[Dict]:
    """Subsample the clean (all-zero grid) samples so they make up at most
    ``empty_target_frac`` of the final dataset.

    Clean images are the majority class that lets the model win by predicting
    "none". Even with the sparse target, over-representing them re-introduces a
    cheap shortcut, so we cap their share (default ~7%).
    """
    n_problem = len(problem_samples)
    if empty_target_frac <= 0:
        log.info(f"Dropping all {len(clean_samples)} clean samples (empty_target_frac=0)")
        return list(problem_samples)
    if empty_target_frac >= 1 or not problem_samples:
        return problem_samples + clean_samples

    # keep K clean so that K / (n_problem + K) == empty_target_frac
    keep = int(round(empty_target_frac * n_problem / (1.0 - empty_target_frac)))
    keep = min(keep, len(clean_samples))
    rng = random.Random(seed)
    kept_clean = rng.sample(clean_samples, keep) if keep < len(clean_samples) else clean_samples
    total = n_problem + len(kept_clean)
    log.info(
        f"Clean downsample: kept {len(kept_clean)}/{len(clean_samples)} clean "
        f"→ {len(kept_clean)}/{total} = {len(kept_clean)/total:.1%} of dataset "
        f"(target {empty_target_frac:.1%})"
    )
    return problem_samples + kept_clean


def build_dataset(
    input_path: str,
    output_dir: str,
    data_root: str = ".",
    val_ratio: float = 0.1,
    seed: int = 42,
    drop_empty_grids: bool = False,
    empty_target_frac: float = 0.07,
    verbose: bool = False,
    exclude_eval: Optional[str] = None,
    major_threshold: float = 0.9,
    dual_axis: bool = False,
    hierarchical: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exclude_ids = load_exclude_ids(exclude_eval) if exclude_eval else set()

    log.info("=== Building patch-JSON SFT samples ===")
    problem_samples, clean_samples = build_samples(
        input_path, out, data_root, drop_empty_grids, verbose,
        exclude_ids=exclude_ids, major_threshold=major_threshold,
        dual_axis=dual_axis, hierarchical=hierarchical,
    )

    if not problem_samples and not clean_samples:
        log.error("No samples built — check data paths.")
        sys.exit(1)

    converted = downsample_clean(problem_samples, clean_samples, empty_target_frac, seed)

    log.info("=== Shuffle + split ===")
    shuffle_and_split(converted, out, val_ratio, seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build patch-JSON SFT data")
    parser.add_argument("--input", default="sft_samples/sft_unified.jsonl")
    parser.add_argument("--output-dir", default="$DATA_ROOT/viescore2_data")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-empty-grids", action="store_true",
        help="Drop samples whose GT grid is all-zero (no problem patches)",
    )
    parser.add_argument(
        "--empty-target-frac", type=float, default=0.1,
        help="Cap clean (all-zero grid) samples at this fraction of the final "
             "dataset by random subsampling. 0 = drop all clean samples, "
             "1 = keep all. Default 0.07 (~7%%).",
    )
    parser.add_argument(
        "--exclude-eval", default=None,
        help="Path to the eval set (jsonl/json). Any sample whose meta.id appears "
             "there is held out of training to prevent train/eval leakage.",
    )
    parser.add_argument(
        "--major-threshold", type=float, default=0.9,
        help="Defect coverage (resized map value, 0-1) at or above which a "
             "problem cell is marked severe (\"!\") in dual-channel targets — "
             "\"fully defective\" vs partially-touched boundary cells.",
    )
    parser.add_argument(
        "--dual-axis", action="store_true",
        help="Emit VIEScore-style dual score lines (pq:/sc:) instead of the "
             "single score: line, for sources with per-axis labels "
             "(RichHF sub-scores, ImagenWorld raw_scores, COCO=10/10).",
    )
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Hierarchical 16→32 targets: refine each problem cell with 2×2 "
             "subcell brackets (bare column = whole cell). PAL4VST samples "
             "(separate builder) stay 16-only — prompt-matched per source.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    build_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_empty_grids=args.drop_empty_grids,
        empty_target_frac=args.empty_target_frac,
        verbose=args.verbose,
        exclude_eval=args.exclude_eval,
        major_threshold=args.major_threshold,
        dual_axis=args.dual_axis,
        hierarchical=args.hierarchical,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

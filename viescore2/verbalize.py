#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training-free faithful verbalizer.

Renders the evaluator's structured output (score lines + channel cells +
coverage marks) into a short English explanation. Fully deterministic and
model-free: every sentence is a mechanical function of the verifiable
output, so the explanation is faithful by construction — it cannot claim
anything the grid and score lines do not already state, and every claim
carries a row/column anchor that can be checked against the grid.

Usage:
    from verbalize import verbalize
    text = verbalize(raw_model_response)
"""

from typing import Dict, List, Optional

import numpy as np

from run_eval import (  # noqa: E402  (same parsers as eval)
    GRID_SIZE,
    parse_axis_scores,
    parse_channels_from_response,
    parse_grid_from_response,
    parse_score_from_response,
)

# Channel → noun phrase used in sentences.
_NOUN = {
    "artifact": "visual artifacts",
    "misalign": "content that does not match the prompt",
    "flat": "quality problems",
}
_LARGE = GRID_SIZE * GRID_SIZE // 3        # >1/3 of the image
_HUGE = GRID_SIZE * GRID_SIZE * 3 // 4     # >3/4 of the image


def _regions(problem: np.ndarray, major: Optional[np.ndarray] = None) -> List[Dict]:
    """4-connected components of a binary grid, largest first."""
    seen = np.zeros_like(problem, dtype=bool)
    out = []
    n = problem.shape[0]
    for r0 in range(n):
        for c0 in range(n):
            if not problem[r0][c0] or seen[r0][c0]:
                continue
            stack, cells = [(r0, c0)], []
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n and problem[rr][cc] and not seen[rr][cc]:
                        seen[rr][cc] = True
                        stack.append((rr, cc))
            n_major = sum(1 for (r, c) in cells if major is not None and major[r][c])
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            out.append({
                "cells": cells, "size": len(cells),
                "major_frac": n_major / len(cells),
                "r0": min(rows), "r1": max(rows), "c0": min(cols), "c1": max(cols),
            })
    out.sort(key=lambda g: -g["size"])
    return out


def _location(reg: Dict) -> str:
    """Centroid → 3×3 semantic position; very large regions get size words."""
    if reg["size"] >= _HUGE:
        return "throughout the image"
    rc = (reg["r0"] + reg["r1"]) / 2.0 / (GRID_SIZE - 1)
    cc = (reg["c0"] + reg["c1"]) / 2.0 / (GRID_SIZE - 1)
    row = "top" if rc < 1 / 3 else ("bottom" if rc > 2 / 3 else "middle")
    col = "left" if cc < 1 / 3 else ("right" if cc > 2 / 3 else "center")
    name = {
        ("top", "left"): "top-left", ("top", "center"): "top",
        ("top", "right"): "top-right",
        ("middle", "left"): "left side", ("middle", "center"): "center",
        ("middle", "right"): "right side",
        ("bottom", "left"): "bottom-left", ("bottom", "center"): "bottom",
        ("bottom", "right"): "bottom-right",
    }[(row, col)]
    if reg["size"] >= _LARGE:
        return f"a large region around the {name}"
    return f"the {name}"


def _anchor(reg: Dict) -> str:
    """Row/column anchor so every claim is checkable against the grid."""
    r = (f"row {reg['r0'] + 1}" if reg["r0"] == reg["r1"]
         else f"rows {reg['r0'] + 1}–{reg['r1'] + 1}")
    c = (f"column {reg['c0'] + 1}" if reg["c0"] == reg["c1"]
         else f"columns {reg['c0'] + 1}–{reg['c1'] + 1}")
    return f"{r}, {c}"


def _severity_word(reg: Dict) -> str:
    # "!" marks encode geometric COVERAGE (cell >=90% inside the defect),
    # not human-rated severity — the rendered words must say what is measured
    if reg["major_frac"] > 0.5:
        return "extensive"
    if reg["major_frac"] > 0:
        return "substantial"
    return "localized"


def _regions_phrase(regs: List[Dict], noun: str, max_regions: int = 3) -> str:
    parts = []
    for g in regs[:max_regions]:
        loc = _location(g)
        prep = "" if loc.startswith("throughout") else "in "
        parts.append(f"{_severity_word(g)} {noun} {prep}{loc} ({_anchor(g)})")
    extra = len(regs) - max_regions
    text = " and ".join(parts)
    if extra > 0:
        text += f", plus {extra} smaller region{'s' if extra > 1 else ''}"
    return text


def _level(score: float) -> str:
    if score >= 8:
        return "high"
    if score >= 5:
        return "moderate"
    return "low"


def verbalize(response: str) -> str:
    """Structured evaluator output → faithful English explanation."""
    pq, sc = parse_axis_scores(response)
    overall = parse_score_from_response(response)
    channels = parse_channels_from_response(response)
    union = parse_grid_from_response(response)

    if union is None and pq is None and sc is None and overall is None:
        return "The evaluator output could not be parsed."

    zero = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if channels:
        art = _regions(channels.get("artifact", {}).get("problem", zero),
                       channels.get("artifact", {}).get("major"))
        mis = _regions(channels.get("misalign", {}).get("problem", zero),
                       channels.get("misalign", {}).get("major"))
        flat = []
    else:
        art, mis = [], []
        flat = _regions(union if union is not None else zero)

    sents: List[str] = []
    if pq is not None:
        s = f"Perceptual quality is {_level(pq)} ({pq:g}/10)"
        if art:
            joiner = ", driven by " if pq < 8 else "; the grid nevertheless flags "
            s += joiner + _regions_phrase(art, _NOUN["artifact"])
        elif channels:
            s += "; no artifact regions were flagged"
        sents.append(s + ".")
    elif art:
        sents.append(f"The grid flags {_regions_phrase(art, _NOUN['artifact'])}.")

    if sc is not None:
        if not mis and sc >= 8:
            sents.append(f"The image follows the prompt faithfully "
                         f"(semantic consistency {sc:g}/10); no misaligned "
                         f"content was detected.")
        else:
            s = f"Semantic consistency is {_level(sc)} ({sc:g}/10)"
            if mis:
                joiner = ", driven by " if sc < 8 else "; the grid nevertheless flags "
                s += joiner + _regions_phrase(mis, _NOUN["misalign"])
            elif channels:
                s += "; no misaligned regions were flagged"
            sents.append(s + ".")
    elif mis:
        sents.append(f"The grid flags {_regions_phrase(mis, _NOUN['misalign'])}.")

    if overall is not None and pq is None and sc is None:
        s = f"Overall quality is {_level(overall)} ({overall:g}/10)"
        if flat:
            joiner = ", driven by " if overall < 8 else "; the grid nevertheless flags "
            s += joiner + _regions_phrase(flat, _NOUN["flat"])
        sents.append(s + ".")
    elif flat and overall is None:
        sents.append(f"The grid flags {_regions_phrase(flat, _NOUN['flat'])}.")

    if not sents and union is not None:
        sents.append("No visible defects were detected.")

    return " ".join(sents)


if __name__ == "__main__":
    import sys
    print(verbalize(sys.stdin.read()))

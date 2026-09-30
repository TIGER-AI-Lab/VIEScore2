#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Output-representation formats for the controlled representation ablation.

Each format is a pair (emit, parse) that maps a 16x16 binary defect grid to
a text target and back, so every representation is trained and evaluated on
the SAME samples/images/prompts and scored by the SAME rasterized cell IoU.
Only the output form differs — the controlled variable the reviewer asked
for. All formats are pure localization (no score / channel lines).
"""
import re
from typing import Optional

import numpy as np

GRID_SIZE = 16


def _components(grid: np.ndarray):
    """4-connected components as lists of (r, c)."""
    seen = np.zeros_like(grid, dtype=bool)
    comps = []
    n = grid.shape[0]
    for r0 in range(n):
        for c0 in range(n):
            if not grid[r0][c0] or seen[r0][c0]:
                continue
            stack, cells = [(r0, c0)], []
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n and grid[rr][cc] and not seen[rr][cc]:
                        seen[rr][cc] = True
                        stack.append((rr, cc))
            comps.append(cells)
    return comps


# ── sparse row enumeration (ours) ────────────────────────────────────────────
def emit_sparse(g: np.ndarray) -> str:
    lines = []
    for r in range(g.shape[0]):
        cols = [c + 1 for c in range(g.shape[1]) if g[r][c]]
        if cols:
            lines.append(f"r{r + 1}: " + ",".join(map(str, cols)))
    return "\n".join(lines) if lines else "none"


def parse_sparse(text: str, grid_size: int = GRID_SIZE) -> Optional[np.ndarray]:
    g = np.zeros((grid_size, grid_size), np.uint8)
    found = False
    for m in re.finditer(r'(?:r|row)?\s*(\d{1,2})\s*:\s*([0-9][0-9,\s]*)', text, re.I):
        row = int(m.group(1))
        if not (1 <= row <= grid_size):
            continue
        for cs in re.findall(r'\d{1,2}', m.group(2)):
            col = int(cs)
            if 1 <= col <= grid_size:
                g[row - 1, col - 1] = 1
                found = True
    if found:
        return g
    return g if re.search(r'\bnone\b', text, re.I) else None


# ── dense 16x16 bitmap ───────────────────────────────────────────────────────
def emit_dense(g: np.ndarray) -> str:
    return "\n".join("".join(str(int(g[r][c])) for c in range(g.shape[1]))
                     for r in range(g.shape[0]))


def parse_dense(text: str) -> Optional[np.ndarray]:
    rows = re.findall(r'[01]{16}', text)
    if len(rows) < GRID_SIZE:
        return None
    g = np.array([[int(ch) for ch in rows[r]] for r in range(GRID_SIZE)], np.uint8)
    return g


# ── bounding boxes over connected components ─────────────────────────────────
def emit_bbox(g: np.ndarray) -> str:
    comps = _components(g)
    if not comps:
        return "none"
    lines = []
    for cells in comps:
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        lines.append(f"box: {min(rs)+1},{min(cs)+1},{max(rs)+1},{max(cs)+1}")
    return "\n".join(lines)


def parse_bbox(text: str) -> Optional[np.ndarray]:
    g = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
    found = False
    for m in re.finditer(r'box\s*:\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})', text, re.I):
        r0, c0, r1, c1 = (int(m.group(i)) for i in range(1, 5))
        if not all(1 <= v <= GRID_SIZE for v in (r0, c0, r1, c1)):
            continue
        g[min(r0, r1) - 1:max(r0, r1), min(c0, c1) - 1:max(c0, c1)] = 1
        found = True
    if found:
        return g
    return g if re.search(r'\bnone\b', text, re.I) else None


# ── flat point list ──────────────────────────────────────────────────────────
def emit_points(g: np.ndarray) -> str:
    pts = [f"{r+1},{c+1}" for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r][c]]
    return " ".join(pts) if pts else "none"


def parse_points(text: str) -> Optional[np.ndarray]:
    g = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
    found = False
    for m in re.finditer(r'(\d{1,2})\s*,\s*(\d{1,2})', text):
        r, c = int(m.group(1)), int(m.group(2))
        if 1 <= r <= GRID_SIZE and 1 <= c <= GRID_SIZE:
            g[r - 1, c - 1] = 1
            found = True
    if found:
        return g
    return g if re.search(r'\bnone\b', text, re.I) else None


FORMATS = {
    "sparse": (emit_sparse, parse_sparse,
               'list every problematic cell grouped by row as "r<row>: '
               '<col>,<col>"; output "none" if there are no problems'),
    "dense": (emit_dense, parse_dense,
              'output a 16-line bitmap, one line per row, each line 16 '
              'characters of 0 (clean) or 1 (problem)'),
    "bbox": (emit_bbox, parse_bbox,
             'output one bounding box per problem region as '
             '"box: <row0>,<col0>,<row1>,<col1>"; output "none" if there '
             'are no problems'),
    "points": (emit_points, parse_points,
               'output every problematic cell as a space-separated list of '
               '"<row>,<col>" points; output "none" if there are no problems'),
}

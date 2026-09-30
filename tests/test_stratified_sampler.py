#!/usr/bin/env python
"""Stratified GRPO sampler tests (P0-3).

Run: python tests/test_stratified_sampler.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viescore2"))

import numpy as np  # noqa: E402
from train_grpo import stratified_sample, _stratum_key  # noqa: E402


def mk(schema, cond="single", ncells=0, uid=0):
    grid = np.zeros((16, 16), dtype=np.uint8)
    flat = grid.reshape(-1)
    flat[:ncells] = 1
    row = {
        "image_paths": ["/a.png", "/b.png"] if cond == "multi" else ["/a.png"],
        "user_text": "p", "target_text": "", "has_loc_gt": True,
        "gt_grid": grid.tolist(), "gt_major": grid.tolist(),
        "gt_score": -1.0, "gt_pq": -1.0, "gt_sc": -1.0, "uid": uid,
    }
    if schema == "score_only":
        row["has_loc_gt"] = False
        row["target_text"] = "score: 5/10"
        row["gt_score"] = 5.0
    elif schema == "channels":
        row["target_text"] = "artifact:\nr5: 8\nmisalign:\nnone"
    elif schema == "grid+score":
        row["target_text"] = "pq: 5/10\nsc: 5/10\nr5: 8"
        row["gt_pq"] = 5.0
    else:  # grid_only
        row["target_text"] = "r5: 8"
    return row


def test_keys():
    assert _stratum_key(mk("score_only"))[0] == "score_only"
    assert _stratum_key(mk("channels"))[0] == "channels"
    assert _stratum_key(mk("grid+score"))[0] == "grid+score"
    assert _stratum_key(mk("grid_only"))[0] == "grid_only"
    assert _stratum_key(mk("grid_only", cond="multi"))[1] == "multi"
    assert _stratum_key(mk("grid_only", ncells=0))[2] == "0"
    assert _stratum_key(mk("grid_only", ncells=24))[2] == "1-24"
    assert _stratum_key(mk("grid_only", ncells=97))[2] == "97+"
    print("ok: stratum keys")


def test_proportions_and_determinism():
    pool = ([mk("score_only", uid=i) for i in range(600)]
            + [mk("channels", ncells=30, uid=1000 + i) for i in range(300)]
            + [mk("grid+score", cond="multi", ncells=10, uid=2000 + i) for i in range(100)])
    k = 100
    a = stratified_sample(pool, k, seed=7)
    b = stratified_sample(pool, k, seed=7)
    assert [r["uid"] for r in a] == [r["uid"] for r in b], "not deterministic"
    c = stratified_sample(pool, k, seed=8)
    assert [r["uid"] for r in a] != [r["uid"] for r in c], "seed has no effect"
    assert len(a) == k
    n_so = sum(1 for r in a if not r["has_loc_gt"])
    n_ch = sum(1 for r in a if "artifact" in r["target_text"])
    n_ms = sum(1 for r in a if len(r["image_paths"]) > 1)
    # exact proportional allocation: 60/30/10
    assert n_so == 60 and n_ch == 30 and n_ms == 10, (n_so, n_ch, n_ms)
    print("ok: proportions + determinism")


def test_full_take():
    pool = [mk("grid_only", uid=i) for i in range(10)]
    out = stratified_sample(pool, 0, seed=1)
    assert len(out) == 10
    out = stratified_sample(pool, 50, seed=1)
    assert len(out) == 10
    print("ok: full take when k<=0 or k>=n")


if __name__ == "__main__":
    test_keys()
    test_proportions_and_determinism()
    test_full_take()
    print("ALL PASS: test_stratified_sampler")

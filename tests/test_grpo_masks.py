#!/usr/bin/env python
"""Supervision-mask tests for the GRPO rewards.

Run: python tests/test_grpo_masks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viescore2"))

import numpy as np  # noqa: E402
from train_grpo import _extract_row, make_reward_funcs  # noqa: E402


def chat_row(target, n_images=1):
    content = [{"type": "image", "path": f"/img/{i}.png"} for i in range(n_images)]
    content.append({"type": "text", "text": "prompt"})
    return {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]}


def test_extract_row_masks():
    # score-only target (EvalMuse booster) → NO localization GT
    r = _extract_row(chat_row("score: 2/10"))
    assert r["has_loc_gt"] is False, "score-only row must not carry loc GT"
    assert r["gt_score"] == 2.0
    assert not np.asarray(r["gt_grid"]).any()

    # grid target → loc GT present
    r = _extract_row(chat_row("pq: 3/10\nsc: 4/10\nr5: 8,9"))
    assert r["has_loc_gt"] is True
    assert np.asarray(r["gt_grid"]).sum() == 2

    # clean grid target ("none") → loc GT present (a REAL clean label)
    r = _extract_row(chat_row("pq: 10/10\nsc: 10/10\nnone"))
    assert r["has_loc_gt"] is True
    assert not np.asarray(r["gt_grid"]).any()

    # corrupt grid section must raise, never silently become "clean"
    try:
        _extract_row(chat_row("cells: garbage %%"))
        raise AssertionError("corrupt grid target must raise")
    except ValueError:
        pass
    print("ok: _extract_row masks")


def test_localization_reward_masked():
    loc, fmt, score = make_reward_funcs()
    zeros = np.zeros((16, 16), dtype=np.uint8).tolist()
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4, 7] = 1

    # EvalMuse-style row (has_loc_gt=False): reward must be CONSTANT across
    # wildly different completions — constant ⇒ zero advantage ⇒ no gradient.
    comps = ["none", "r5: 8,9,10", "score: 7/10", "garbage"]
    out = loc(completions=comps, gt_grid=[zeros] * 4, gt_major=[zeros] * 4,
              has_loc_gt=[False] * 4)
    assert len(set(out)) == 1, f"masked row leaked gradient: {out}"

    # The OLD behavior this replaces: unmasked all-zero GT pays 1.0 for "none"
    # and 0.0 otherwise — that differential is exactly what must not happen
    # on unlabeled rows.
    out_old = loc(completions=comps, gt_grid=[zeros] * 4, gt_major=[zeros] * 4,
                  has_loc_gt=[True] * 4)
    assert out_old[0] == 1.0 and out_old[1] == 0.0, "sanity: real clean rows still score"

    # supervised row unaffected by the mask plumbing
    out2 = loc(completions=["r5: 8", "none"], gt_grid=[gt.tolist()] * 2,
               gt_major=[zeros] * 2, has_loc_gt=[True] * 2)
    assert out2[0] == 1.0 and out2[1] == 0.0
    print("ok: localization reward masking")


def test_format_reward_schema_aware():
    loc, fmt, score = make_reward_funcs()
    zeros = np.zeros((16, 16), dtype=np.uint8).tolist()
    # score-only rows: a bare score IS well-formed; a grid is not required
    out = fmt(completions=["score: 7/10", "pq: 3/10\nsc: 5/10", "junk!!"],
              gt_grid=[zeros] * 3, has_loc_gt=[False] * 3)
    assert out == [1.0, 1.0, 0.0], f"score-only format reward wrong: {out}"
    # grid rows: unchanged behavior
    out = fmt(completions=["none", "r5: 8,9", "junk!!"],
              gt_grid=[zeros] * 3, has_loc_gt=[True] * 3)
    assert out == [1.0, 1.0, 0.0]
    print("ok: format reward schema-aware")


def test_score_reward_pal_masked():
    loc, fmt, score = make_reward_funcs()
    zeros = np.zeros((16, 16), dtype=np.uint8).tolist()
    # PAL4VST rows (no score GT anywhere): whole group 0 → no gradient
    out = score(completions=["score: 7/10", "none"], gt_grid=[zeros] * 2,
                gt_score=[-1.0] * 2, gt_pq=[-1.0] * 2, gt_sc=[-1.0] * 2)
    assert out == [0.0, 0.0]
    # EvalMuse rows: single-score reward active
    out = score(completions=["score: 7/10", "score: 2/10"], gt_grid=[zeros] * 2,
                gt_score=[7.0, 7.0], gt_pq=[-1.0] * 2, gt_sc=[-1.0] * 2)
    assert out[0] == 1.0 and out[1] == 0.5
    print("ok: score reward masking (unchanged)")


if __name__ == "__main__":
    test_extract_row_masks()
    test_localization_reward_masked()
    test_format_reward_schema_aware()
    test_score_reward_pal_masked()
    print("ALL PASS: test_grpo_masks")

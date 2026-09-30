#!/usr/bin/env python
"""Source-matched eval protocol tests (P0-1).

Checks that configs/eval_protocol.json (a) covers every source in
eval_suite.jsonl, (b) encodes the schemas measured on the training corpus, and
(c) the score-only prompt reproduces the EvalMuse booster's training prompt
byte-for-byte.

Run: python tests/test_protocol_config.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viescore2"))

from run_eval import (  # noqa: E402
    SCORE_ONLY_PROMPT_PREFIX, SCORE_ONLY_PROMPT_SUFFIX,
)
from build_sft_data import build_user_prompt  # noqa: E402

CONFIG = ROOT / "configs" / "eval_protocol.json"
EVAL_SUITE = ROOT / "eval_samples" / "eval_suite.jsonl"


def load_config():
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def test_covers_eval_suite_sources():
    # eval_suite.jsonl is a build artifact (viescore2/build_eval.py), not tracked;
    # skip rather than fail when running from a fresh clone.
    if not EVAL_SUITE.exists():
        print(f"skip: {EVAL_SUITE.relative_to(ROOT)} absent "
              "(build it with viescore2/build_eval.py)")
        return
    proto = load_config()
    sources = set()
    with open(EVAL_SUITE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sources.add(json.loads(line).get("meta", {}).get("source", ""))
    missing = sources - set(proto)
    assert not missing, f"protocol config misses eval_suite sources: {missing}"
    print(f"ok: covers all {len(sources)} eval_suite sources")


def test_schemas_match_training():
    proto = load_config()
    # measured on viescore2_data_full/train.json (2026-07-27 scan)
    assert proto["RichHF-18K"] == {"with_score": True, "axes": True, "channels": True}
    assert proto["PAL4VST-test"] == {"with_score": False, "axes": False, "channels": False}
    assert proto["EvalMuse"] == {"score_only": True}
    for s in ("TIG", "TIE", "SRIG", "SRIE", "MRIG", "MRIE"):
        assert proto[f"ImagenWorld-{s}"] == {"with_score": True, "axes": True,
                                             "channels": False}, s
    assert proto["COCO-real"] == {"with_score": True, "axes": True, "channels": False}
    print("ok: schemas match the measured training prompts")


def test_prompt_fingerprints():
    # PAL4VST protocol prompt must not request any score line
    pal = build_user_prompt("x", 0, with_score=False, channels=False, axes=False)
    assert "score" not in pal.lower() and 'pq:' not in pal, pal
    # dual-axis prompt requests pq/sc but no single overall score line
    dual = build_user_prompt("x", 0, with_score=True, axes=True)
    assert 'pq: <0-10>/10' in dual and 'sc: <0-10>/10' in dual
    assert 'score: <0-10>/10' not in dual
    # channel prompt requests the two sections
    chan = build_user_prompt("x", 0, with_score=True, axes=True, channels=True)
    assert '"artifact:" section' in chan and 'misalign' in chan
    # EvalMuse score-only template, byte-identical to the booster rows
    em = SCORE_ONLY_PROMPT_PREFIX + "blacksmith crafts sword, Panorama" + SCORE_ONLY_PROMPT_SUFFIX
    expected = ('Generation prompt: "blacksmith crafts sword, Panorama"\n'
                'Give the overall quality score of the GENERATED image as '
                '"score: <0-10>/10".')
    assert em == expected, f"\n{em!r}\n!=\n{expected!r}"
    print("ok: prompt fingerprints")


if __name__ == "__main__":
    test_covers_eval_suite_sources()
    test_schemas_match_training()
    test_prompt_fingerprints()
    print("ALL PASS: test_protocol_config")

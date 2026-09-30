#!/usr/bin/env python
"""COCO caption hygiene tests strict mode never synthesizes captions.

Run: python tests/test_coco_caption_hygiene.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_builder.ingest.coco import (  # noqa: E402
    COCOIngestor, is_template_caption, _COCO_CAPTION_TEMPLATES,
)


def test_template_detector():
    for t in _COCO_CAPTION_TEMPLATES:
        rendered = t.format(subj="a dog and a cat", subj_cap="A dog and a cat")
        assert is_template_caption(rendered), f"detector misses: {rendered}"
    for human in [
        "A photo of a orange tabby cat wearing a cowboy hat, full body",
        "Two men riding horses on a beach at sunset.",
        "",
    ]:
        assert not is_template_caption(human), f"false positive: {human}"
    print("ok: template detector")


def _ingestor(strict):
    return COCOIngestor(data_root=Path("/nonexistent-coco"), max_samples=1,
                        strict_captions=strict)


def test_strict_never_templates():
    ing = _ingestor(strict=True)
    # official caption present → human, regardless of strictness
    cap, src = ing._extract_caption({"caption": "A dog on a couch."})
    assert (cap, src) == ("A dog on a couch.", "human")
    # no caption, only object categories → strict SKIPS (no synthesis)
    cap, src = ing._extract_caption({"objects": {"category": [16, 17]}},
                                    category_names=None, image_id=7)
    assert cap == "" and src == "none"
    assert ing._template_skipped == 1
    print("ok: strict mode skips instead of synthesizing")


def test_optin_template_path():
    ing = _ingestor(strict=False)
    cap, src = ing._extract_caption({"objects": {"category": [16, 17]}},
                                    category_names=None, image_id=7)
    assert src == "template" and is_template_caption(cap), (cap, src)
    # deterministic per image_id across rebuilds
    cap2, _ = ing._extract_caption({"objects": {"category": [16, 17]}},
                                   category_names=None, image_id=7)
    assert cap == cap2
    print("ok: opt-in template path (non-strict only), deterministic")


if __name__ == "__main__":
    test_template_detector()
    test_strict_never_templates()
    test_optin_template_path()
    print("ALL PASS: test_coco_caption_hygiene")

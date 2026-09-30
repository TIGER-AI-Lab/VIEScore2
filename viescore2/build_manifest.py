#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset manifest builder 

The chat rows in the training corpora carry NO metadata — only ``messages`` —
so source identity, supervision type and composition were previously
unrecorded and had to be re-derived by hand. This script freezes them into a
manifest JSON:

  * per data file: md5, row count;
  * per source (inferred from image paths + prompt/target schema, the same
    fingerprints measured on the 2026-07-27 corpus scan): row count,
    clean/problem counts, conditioning-image stats, supervision flags
    (localization / channels / overall score / pq / sc), prompt schema,
    GT-cell and target-length percentiles;
  * COCO caption hygiene: template-caption count (must be 0 in official
    builds, see sft_builder/ingest/coco.py::is_template_caption);
  * for eval jsonl files: per-meta.source counts and GT availability.

Usage:
    python viescore2/build_manifest.py \
        --train $DATA_ROOT/viescore2_data_full/train.json \
        --train $DATA_ROOT/viescore2_data_full/val.json \
        --eval eval_samples/eval_suite.jsonl \
        --out manifests/data_manifest.json
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_builder.ingest.coco import is_template_caption  # noqa: E402


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_chat_row(sample):
    """Infer (source, schema flags) for one chat row from paths + prompts."""
    msgs = sample["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    utext = next(c["text"] for c in user["content"]
                 if isinstance(c, dict) and c.get("type") == "text")
    asst = next(m for m in msgs if m["role"] == "assistant")
    t = asst["content"][0]["text"] if isinstance(asst["content"], list) else asst["content"]
    paths = [c.get("path", "") for c in user["content"]
             if isinstance(c, dict) and c.get("type") == "image"]
    joined = " ".join(paths)

    prompt_channels = '"artifact:" section' in utext
    prompt_axes = 'pq: <0-10>/10' in utext
    prompt_score1 = 'score: <0-10>/10' in utext and not prompt_axes
    prompt_grid = 'r<row>' in utext

    if "images_evalmuse" in joined:
        source = "EvalMuse"
    elif "pal4vst" in joined.lower():
        source = "PAL4VST"
    elif "coco" in joined.lower():
        source = "COCO-real"
    elif prompt_channels:
        source = "RichHF-18K"
    elif len(paths) > 1:
        source = "ImagenWorld-edit"
    else:
        source = "core-flat-single"

    target_grid = bool(re.search(r"^r\d+:", t, re.M)) or bool(
        re.search(r"(^|\n)\s*none\s*(\n|$)", t))
    target_channels = bool(re.search(r"^(artifact|misalign):", t, re.M))
    target_score1 = bool(re.search(r"^score:", t, re.M))
    target_pq = bool(re.search(r"^pq:", t, re.M))
    target_sc = bool(re.search(r"^sc:", t, re.M))
    clean = target_grid and not re.search(r"^r\d+:", t, re.M)
    ncells = 0
    for line in t.splitlines():
        m = re.match(r"r\d+:\s*(.+)", line)
        if m:
            ncells += len([x for x in m.group(1).split(",") if x.strip()])
    instruction = ""
    m = re.search(r'prompt: "(.*?)"\n', utext, re.S)
    if m:
        instruction = m.group(1)
    return {
        "source": source,
        "prompt": {"axes": prompt_axes, "score1": prompt_score1,
                   "channels": prompt_channels, "grid": prompt_grid},
        "target": {"grid": target_grid, "channels": target_channels,
                   "score": target_score1, "pq": target_pq, "sc": target_sc},
        "n_conditioning_images": len(paths) - 1,
        "clean": clean,
        "gt_cells": ncells,
        "target_chars": len(t),
        "instruction": instruction,
    }


def manifest_chat_file(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    per = defaultdict(list)
    for s in rows:
        info = classify_chat_row(s)
        per[info["source"]].append(info)
    sources = {}
    for src, infos in sorted(per.items()):
        cells = sorted(i["gt_cells"] for i in infos)
        chars = sorted(i["target_chars"] for i in infos)
        prompt_kinds = Counter(json.dumps(i["prompt"], sort_keys=True) for i in infos)
        target_kinds = Counter(json.dumps(i["target"], sort_keys=True) for i in infos)
        entry = {
            "n": len(infos),
            "n_clean": sum(i["clean"] for i in infos),
            "n_problem": sum((not i["clean"]) and i["target"]["grid"] for i in infos),
            "n_conditioned": sum(i["n_conditioning_images"] > 0 for i in infos),
            "supervision": {
                "localization": sum(i["target"]["grid"] for i in infos),
                "channels": sum(i["target"]["channels"] for i in infos),
                "overall_score": sum(i["target"]["score"] for i in infos),
                "pq": sum(i["target"]["pq"] for i in infos),
                "sc": sum(i["target"]["sc"] for i in infos),
            },
            "prompt_schemas": {k: v for k, v in prompt_kinds.most_common()},
            "target_schemas": {k: v for k, v in target_kinds.most_common()},
            "gt_cells": {"p50": _pct(cells, .5), "p90": _pct(cells, .9),
                         "p99": _pct(cells, .99), "max": cells[-1] if cells else None},
            "target_chars": {"p50": _pct(chars, .5), "p90": _pct(chars, .9),
                             "p99": _pct(chars, .99), "max": chars[-1] if chars else None},
        }
        if src == "COCO-real":
            entry["template_captions"] = sum(
                is_template_caption(i["instruction"]) for i in infos)
        sources[src] = entry
    return {"path": str(path), "md5": _md5(path), "n_rows": len(rows),
            "sources": sources}


def manifest_eval_file(path):
    per = defaultdict(lambda: {"n": 0, "with_score_gt": 0, "with_spatial_gt": 0,
                               "template_captions": 0})
    n = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        s = json.loads(line)
        n += 1
        src = s.get("meta", {}).get("source", "unknown")
        d = per[src]
        d["n"] += 1
        resp = s.get("response", {}) or {}
        if resp.get("score") is not None:
            d["with_score_gt"] += 1
        orig = s.get("meta", {}).get("orig", {}) or {}
        if resp.get("visual_reason") or orig.get("artifact_map_png") \
                or orig.get("misalign_map_png") or src == "COCO-real":
            d["with_spatial_gt"] += 1
        if src == "COCO-real" and is_template_caption(s.get("instruction", "")):
            d["template_captions"] += 1
    return {"path": str(path), "md5": _md5(path), "n_rows": n,
            "sources": dict(sorted(per.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", default=[],
                    help="chat-format corpus json (repeatable)")
    ap.add_argument("--eval", action="append", default=[],
                    help="eval jsonl (repeatable)")
    ap.add_argument("--out", default="manifests/data_manifest.json")
    args = ap.parse_args()

    out = {"training_data": [manifest_chat_file(p) for p in args.train],
           "eval_data": [manifest_eval_file(p) for p in args.eval],
           "notes": {
               "source_inference": "chat rows carry no metadata; source is "
                   "inferred from image paths (EvalMuse/PAL4VST/COCO) and "
                   "prompt schema (channels→RichHF, multi-image→ImagenWorld "
                   "edit); core-flat-single = single-image dual-axis rows "
                   "(ImagenWorld TIG et al.)",
               "evalmuse_provenance": "EvalMuse score-only booster rows are "
                   "appended from $DATA_ROOT/inspector_data_em by "
                   "bash_scripts/finish_clean_data.sh; the raw-to-chat "
                   "builder for them is NOT in this repo (known gap)",
           }}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for f in out["training_data"] + out["eval_data"]:
        srcs = {k: v["n"] for k, v in f["sources"].items()}
        print(f"{f['path']}  md5={f['md5'][:10]}…  n={f['n_rows']}  {srcs}")
    print(f"manifest → {out_path}")


if __name__ == "__main__":
    main()

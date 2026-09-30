#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FGA-BLIP2 (EvalMuse specialist) as an external SCORE baseline, on any of our
eval sets. Runs the published checkpoint on the generated image + instruction
(FGA-BLIP2 is single-image, so conditioning images are not shown) and reports
tie-aware correlation with our human overall score, non-COCO.

Committed into the repo so the baseline is reproducible (the FGA
runner/artifacts were previously uncommitted). Caveat: FGA-BLIP2 predicts
text-image ALIGNMENT MOS; our overall-quality GT differs, so this is a
cross-objective reference, not a ceiling.

    python viescore2/eval_fga_blip2.py \
        --eval-data eval_samples/eval_suite.jsonl \
        --output eval_results/fga_blip2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "$DATA_ROOT/evalmuse_repo")

# Relative image paths in the eval sets are repo-root relative.
REPO_ROOT = Path(__file__).resolve().parent.parent


def rank_avg(x):
    """Average ranks with tie handling (scipy-style). Inlined so this runs in
    the evalmuse venv without importing the VIEScore2 stack ()."""
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


def corr(rs):
    if len(rs) < 3:
        return {"n": len(rs)}
    p = np.array([r["pred"] for r in rs], float)
    g = np.array([r["gt"] for r in rs], float)
    if p.std() == 0 or g.std() == 0:
        return {"n": len(rs), "pearson": None, "spearman": None}
    return {"n": len(rs),
            "pearson": round(float(np.corrcoef(p, g)[0, 1]), 4),
            "spearman": round(float(np.corrcoef(rank_avg(p), rank_avg(g))[0, 1]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-data", default="eval_samples/eval_suite.jsonl")
    ap.add_argument("--output", default="eval_results/fga_blip2.json")
    ap.add_argument("--checkpoint", default="$DATA_ROOT/evalmuse_repo/checkpoints/fga_blip2.pth")
    args = ap.parse_args()

    from lavis.models import load_model_and_preprocess
    device = "cuda"
    model, vis_processors, text_processors = load_model_and_preprocess(
        "fga_blip2", "coco", device=device, is_eval=True)
    model.load_checkpoint(args.checkpoint)
    print("checkpoint loaded", flush=True)

    samples = [json.loads(l) for l in open(args.eval_data) if l.strip()]
    rows = []
    for i, s in enumerate(samples):
        path = s["image"]
        if not path.startswith("/"):
            path = str(REPO_ROOT / path)
        if s["response"].get("score") is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[{i}] skip {s['meta'].get('id')}: {e}", flush=True)
            continue
        img_t = vis_processors["eval"](img).unsqueeze(0).to(device)
        txt = text_processors["eval"](s.get("instruction", ""))
        with torch.no_grad():
            alignment_score, _ = model.element_score(img_t, [txt])
        rows.append({"id": s["meta"].get("id"), "source": s["meta"].get("source"),
                     "pred": float(alignment_score), "gt": float(s["response"]["score"])})
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(samples)}", flush=True)

    out = {"eval_data": args.eval_data, "spearman": "tie_averaged_ranks",
           "all": corr(rows),
           "non_coco": corr([r for r in rows if r["source"] != "COCO-real"]),
           "rows": rows}
    # per-source breakdown
    srcs = sorted({r["source"] for r in rows})
    out["per_source"] = {s: corr([r for r in rows if r["source"] == s]) for s in srcs}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()

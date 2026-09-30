#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MMRB2 preference accuracy from paired run_eval scores.

For each expert-annotated pair, the side with the higher pred_score is the
predicted preference. Accuracy counts a correct pick as 1, an exact score
tie (or an unparsed side) as 0.5 — stated, and tie/unparsed rates reported
separately. Per-subset (T2I / Edit) with paired bootstrap 95% CIs (2,000
resamples, seed 42) plus per-prompt-source breakdown.

  python viescore2/analyze_mmrb2.py \
      --results eval_results_external/mmrb2_rl.json \
      --output  eval_results_external/mmrb2_rl_accuracy.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def credit(pair):
    a, b, chosen = pair["a"], pair["b"], pair["chosen"]
    if a is None or b is None or a == b:
        return 0.5
    return 1.0 if (("A" if a > b else "B") == chosen) else 0.0


def block(pairs, rng):
    cr = [credit(p) for p in pairs]
    acc = float(np.mean(cr))
    boots = [float(np.mean([cr[i] for i in rng.integers(0, len(cr), len(cr))]))
             for _ in range(2000)]
    ties = sum(1 for p in pairs
               if p["a"] is not None and p["b"] is not None and p["a"] == p["b"])
    unparsed = sum(1 for p in pairs if p["a"] is None or p["b"] is None)
    return {"n": len(pairs), "accuracy": round(acc, 4),
            "CI95": [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)],
            "score_ties": ties, "unparsed_pairs": unparsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--score-key", default="pred_score",
                    help="per-row field holding the scalar (pred_score for "
                         "run_eval files; pred for eval_mllm_scorers; a "
                         "scorer name for eval_scalar_scorers dumps)")
    args = ap.parse_args()

    res = json.load(open(args.results))["results"]
    pairs = defaultdict(dict)
    for r in res:
        o = r.get("orig") or {}
        if not o:  # run_eval keeps meta.orig? fall back to id parsing
            rid = r["id"]  # mmrb2_<subset>_<pairid>_<side>
            side = rid.rsplit("_", 1)[1]
            key = rid.rsplit("_", 1)[0]
            pairs[key][side] = r
        else:
            pairs[f'{r["source"]}|{o["pair_id"]}'][o["side"]] = r

    # need chosen + prompt_source: recover from the eval jsonl
    meta = {}
    for line in open(ROOT / "eval_samples/mmrb2_eval.jsonl"):
        s = json.loads(line)
        o = s["meta"]["orig"]
        meta[s["meta"]["id"]] = (s["meta"]["source"], o["pair_id"],
                                 o["side"], o["chosen"], o["prompt_source"])

    grouped = defaultdict(dict)
    for r in res:
        m = meta.get(r["id"])
        if m is None:
            continue
        source, pid, side, chosen, psrc = m
        g = grouped[(source, pid)]
        v = r
        for part in args.score_key.split("."):   # supports e.g. scores.overall
            v = v.get(part) if isinstance(v, dict) else None
        g[side] = None if v in (None, "") else float(v)
        g["chosen"], g["psrc"] = chosen, psrc

    per_subset = defaultdict(list)
    per_psrc = defaultdict(list)
    for (source, pid), g in grouped.items():
        if "a" not in g and "b" not in g:
            continue
        p = {"a": g.get("a"), "b": g.get("b"), "chosen": g["chosen"]}
        per_subset[source].append(p)
        per_psrc[(source, g["psrc"])].append(p)

    rng = np.random.default_rng(42)
    out = {s: block(ps, rng) for s, ps in sorted(per_subset.items())}
    out["per_prompt_source"] = {
        f"{s}/{p}": block(ps, rng)["accuracy"]
        for (s, p), ps in sorted(per_psrc.items()) if len(ps) >= 20}
    out["_protocol"] = (
        "pairwise preference accuracy vs expert chosen label; higher "
        "pred_score wins; exact tie or unparsed side scores 0.5 (rates "
        "reported); paired bootstrap 2,000 resamples seed 42; scores from "
        "run_eval under the frozen source-matched prompts "
        "(MMRB2-T2I/-Edit protocol entries = copies of TIG/TIE)")
    json.dump(out, open(args.output, "w"), indent=1)
    for s in per_subset:
        b = out[s]
        print(f"{s}: acc {b['accuracy']} {b['CI95']} n={b['n']} "
              f"ties={b['score_ties']} unparsed={b['unparsed_pairs']}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()

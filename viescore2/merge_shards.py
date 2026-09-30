#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge sharded run_eval outputs (plus any leftover serial .partial rows)
into one final result file, deduped by row id.

  python viescore2/merge_shards.py --out X/gpt56sol_had_eval.json \
      --expect eval_samples/had_eval.jsonl shard1.json shard2.json ...

Rows from shard files win over .partial leftovers (same contract either way).
Exits nonzero unless every expected row id is present, so the sweep script's
skip-if-exists can trust any file this writes.
"""
import argparse
import json
import sys
from pathlib import Path


def rows_of(path):
    try:
        d = json.load(open(path))
    except Exception:
        return [], None
    if isinstance(d, dict):
        return d.get("results", []), d.get("summary")
    return d, None  # .partial files are bare lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", required=True,
                    help="eval jsonl; every row id must be present")
    ap.add_argument("--partial", default=None,
                    help="optional serial-run .partial to salvage rows from")
    args = ap.parse_args()

    by_id, summary = {}, None
    if args.partial and Path(args.partial).exists():
        rows, _ = rows_of(args.partial)
        for r in rows:
            if "error" not in r:
                by_id[r["id"]] = r
        print(f"salvaged {len(by_id)} rows from {args.partial}")
    for p in args.shards:
        rows, s = rows_of(p)
        summary = summary or s
        for r in rows:
            by_id[r["id"]] = r

    expected = [json.loads(l)["meta"]["id"] for l in open(args.expect)]
    missing = [i for i in expected if i not in by_id]
    if missing:
        print(f"INCOMPLETE: {len(missing)}/{len(expected)} rows missing "
              f"(e.g. {missing[:3]}); not writing {args.out}")
        sys.exit(2)
    summary = summary or {}
    summary["sharded_merge"] = {"n_shards": len(args.shards),
                                "n_rows": len(expected)}
    json.dump({"summary": summary,
               "results": [by_id[i] for i in expected]},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {args.out}: {len(expected)} rows")


if __name__ == "__main__":
    main()

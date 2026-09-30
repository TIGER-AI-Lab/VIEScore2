#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the evaluation suite — a multi-source benchmark with native holdouts.

  * >=1000 samples across FIVE sources, three of them NATIVE test splits that
    are held out by construction (0 pool overlap, verified by the audit);
  * RichHF drawn from the OFFICIAL test split WITH dual-channel sidecars, so
    the eval GT is the artifact-union-misalign union that matches training;
  * ImagenWorld / COCO portions taken from the pool one-annotator-per-image,
    added to the training exclusion set and checked by leakage_gate.py.

Composition (target ~1300):
    RichHF-test      400   native   dual-channel GT + score
    PAL4VST-test     300   native   artifact mask GT, no score
    EvalMuse-test    200   native   score only (test-only prompt families)
    ImagenWorld      300   heldout  visual GT + dual-axis score (50/task)
    COCO             100   heldout  clean negatives

Random sampling with a fixed seed (no score stratification).

Output: eval_samples/eval_suite.jsonl (run_eval.py consumes it unchanged;
RichHF sidecars trigger the union-GT path automatically).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_sft_data import load_exclude_ids, _ANNOTATOR_RE  # noqa: E402

POOL = "sft_samples/sft_unified.jsonl"
QUOTAS = {"RichHF-test": 400, "PAL4VST-test": 300, "EvalMuse-test": 200,
          "ImagenWorld": 300, "COCO": 100}
IW_TASKS = ["TIG", "TIE", "SRIG", "SRIE", "MRIG", "MRIE"]


def em_pid(path):
    m = re.search(r"/(\d{5})", path)
    return m.group(1) if m else None


def sample_native(path, quota, rng, source_tag, split_tag, extra=None):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if extra:
        recs = [r for r in recs if extra(r)]
    recs = [r for r in recs if Path(r["image"]).exists()]
    rng.shuffle(recs)
    out = []
    for r in recs[:quota]:
        r.setdefault("meta", {})["split"] = split_tag
        out.append(r)
    print(f"{source_tag}: {len(out)}/{quota} (pool {len(recs)} eligible)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="eval_samples/eval_suite.jsonl")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--extra-exclude", default=None,
                    help="optional jsonl of records whose ids must be excluded")
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

    # optional extra exclusion list (ids never to sample into the suite)
    extra_keys = load_exclude_ids(args.extra_exclude) if args.extra_exclude else set()

    records = []

    # 1-2. native localization test splits (clean by construction)
    records += sample_native("eval_samples/richhf_test.jsonl",
                             QUOTAS["RichHF-test"], rng, "RichHF-test", "native_test")
    records += sample_native("eval_samples/pal4vst_test.jsonl",
                             QUOTAS["PAL4VST-test"], rng, "PAL4VST-test", "native_test")

    # 3. EvalMuse: prompt families NOT in the EM score booster. NOTE: these
    # rows come from the official TRAIN split (the official test set ships
    # unlabeled) — a prompt-heldout carve, and labeled as such.
    manifest = Path(__file__).parent / "manifests" / "evalmuse_aug_pids.json"
    if not manifest.exists():
        raise SystemExit(f"FATAL: {manifest} missing — cannot enforce the "
                         "EvalMuse prompt-family holdout")
    em_train_pids = set(json.loads(manifest.read_text()))
    def em_heldout(r):
        pid = em_pid(r["image"])
        return pid is not None and pid not in em_train_pids
    em_records = sample_native("eval_samples/evalmuse_test.jsonl",
                               QUOTAS["EvalMuse-test"], rng, "EvalMuse-test",
                               "train_carve_promptheldout", extra=em_heldout)
    for r in em_records:  # id must include the generator: one pid, many images
        p = Path(r["image"])
        r["meta"]["id"] = f"evalmuse_{p.parent.name}_{p.stem}"
    records += em_records

    # 4-5. ImagenWorld + COCO from the pool: one record per image, disjoint
    #      held out of training by the exclusion set.
    iw_by_task = defaultdict(dict)   # task -> image -> record
    coco = {}
    with open(POOL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            sid = s.get("meta", {}).get("id", "")
            img = s.get("image", "")
            if not img or not Path(img).exists():
                continue
            # skip anything in the extra exclusion list (id, annotator-stripped id, or path)
            if sid in extra_keys or _ANNOTATOR_RE.sub("", sid) in extra_keys \
                    or str(Path(img).resolve()) in extra_keys:
                continue
            src = s["meta"]["source"]
            if src.startswith("ImagenWorld"):
                task = src.split("-")[1]
                iw_by_task[task].setdefault(img, s)   # first annotator wins
            elif src == "COCO-real":
                coco.setdefault(img, s)

    per_task = QUOTAS["ImagenWorld"] // len(IW_TASKS)
    for task in IW_TASKS:
        imgs = list(iw_by_task[task].values())
        rng.shuffle(imgs)
        picked = imgs[:per_task]
        for r in picked:
            r.setdefault("meta", {})["split"] = "heldout"
        records += picked
        print(f"ImagenWorld-{task}: {len(picked)}/{per_task} (unique imgs {len(imgs)})",
              flush=True)

    coco_recs = list(coco.values())
    rng.shuffle(coco_recs)
    for r in coco_recs[:QUOTAS["COCO"]]:
        r.setdefault("meta", {})["split"] = "heldout"
    records += coco_recs[:QUOTAS["COCO"]]
    print(f"COCO: {min(len(coco_recs), QUOTAS['COCO'])}/{QUOTAS['COCO']}", flush=True)

    rng.shuffle(records)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    comp = Counter(r["meta"]["source"] for r in records)
    print(f"\neval_suite: {len(records)} samples")
    for k, v in sorted(comp.items()):
        print(f"  {k}: {v}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()

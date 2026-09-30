#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train/eval leakage gate. Run before ANY training; exit 1 on contamination.

Checks every (train, eval) pair for:
  1. id overlap — exact meta.id AND annotator-stripped id (ImagenWorld stores
     one id per annotator for the same image; id-only checks leaked 117/240
     eval images into training,  F1);
  2. image-content overlap — training images are RENAMED copies
     (``NNNNNN_basename``), so paths never match; we compare by file size then
     MD5 (eval images fully hashed; training images hashed only on size hit);
  3. prompt overlap — verbatim prompt strings; a HARD failure for EvalMuse
     rows (the "prompt-level holdout" claim,  F2), a warning elsewhere
     (cross-source prompt reuse can be benign).

Training rows may be pool jsonl ({"image": ..., "meta": ...}), axolotl
messages-format, or uniform/scaling rows ({"images": [...]}) — image paths and
prompt strings are harvested recursively, so any format works.

Usage:
  python viescore2/leakage_gate.py \
      --train $DATA_ROOT/viescore2_data/train.json [--train ...] \
      --eval eval_samples/eval_suite.jsonl [--eval ...]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
ANNOT_RE = re.compile(r"_annotator\d+$")
PROMPT_RE = re.compile(r'(?:Generation(?:/edit)? prompt|prompt):\s*"(.*?)"', re.S)


def load_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def harvest_prompts(obj, prompts):
    """Recursively collect prompt strings."""
    if isinstance(obj, str):
        for m in PROMPT_RE.finditer(obj):
            prompts.add(m.group(1).strip())
    elif isinstance(obj, dict):
        if isinstance(obj.get("instruction"), str) and obj["instruction"].strip():
            prompts.add(obj["instruction"].strip())
        for v in obj.values():
            harvest_prompts(v, prompts)
    elif isinstance(obj, list):
        for v in obj:
            harvest_prompts(v, prompts)


def em_prompts_of(rec):
    """Prompts of a record whose imagery is EvalMuse — the only prompts whose
    train/eval overlap is a real 'prompt-level holdout' violation ().
    A shared PAL4VST boilerplate string ('no text prompt available…') or a
    coincidental caption is NOT EvalMuse leakage."""
    imgs, prompts = set(), set()
    harvest_images_shallow(rec, imgs)
    if not any("evalmuse" in i.lower() for i in imgs):
        return set()
    harvest_prompts(rec, prompts)
    return prompts


def harvest_images_shallow(obj, imgs):
    if isinstance(obj, str):
        if obj.lower().endswith(IMG_EXT):
            imgs.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            harvest_images_shallow(v, imgs)
    elif isinstance(obj, list):
        for v in obj:
            harvest_images_shallow(v, imgs)


def target_image(rec):
    """The GENERATED/target image of a record — the thing whose leakage
    contaminates evaluation. Conditioning/reference images are shared between
    records by construction (same original, different edits) and are NOT
    gated. Convention across all our formats: the generated image is `image`,
    or the LAST entry of `images`, or the LAST image in the user turn."""
    if isinstance(rec.get("image"), str) and rec["image"]:
        return rec["image"]
    if isinstance(rec.get("images"), list) and rec["images"]:
        return rec["images"][-1]
    for msg in rec.get("messages", []):
        if msg.get("role") == "user":
            paths = [c.get("path") or c.get("image")
                     for c in msg.get("content", [])
                     if isinstance(c, dict) and c.get("type") == "image"]
            paths = [p for p in paths if isinstance(p, str)]
            if paths:
                return paths[-1]
    return None


def md5_of(path, cache={}):
    if path not in cache:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[path] = h.hexdigest()
    return cache[path]


def ahash_of(path, cache={}):
    """256-bit average hash (16x16 grayscale, median threshold), packed as 32
    bytes. Survives the builders' 384^2 LANCZOS resize, which byte-level MD5
    does not — resized training copies of eval images MUST still be caught."""
    if path not in cache:
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(path).convert("L")
            a = np.asarray(im.resize((16, 16), Image.LANCZOS), dtype=np.float32)
            abits = (a > np.median(a)).astype(np.uint8).ravel()
            d = np.asarray(im.resize((17, 16), Image.LANCZOS), dtype=np.float32)
            dbits = (d[:, 1:] > d[:, :-1]).astype(np.uint8).ravel()
            cache[path] = np.packbits(np.concatenate([abits, dbits])).tobytes()
        except Exception:
            cache[path] = None
    return cache[path]


def hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--eval", action="append", required=True)
    args = ap.parse_args()

    failures = 0
    for eval_path in args.eval:
        erecs = load_records(eval_path)
        eids, eimgs, eprompts, eem_prompts = set(), set(), set(), set()
        for r in erecs:
            sid = (r.get("meta") or {}).get("id")
            if sid:
                eids.add(sid)
                eids.add(ANNOT_RE.sub("", sid))
            t = target_image(r)
            if t:
                eimgs.add(t)
            harvest_prompts(r, eprompts)
            eem_prompts |= em_prompts_of(r)
        # index eval images: exact (size -> md5) AND perceptual (ahash list)
        esize = defaultdict(dict)
        ehashes, ehash_paths = [], []
        e_missing, miss_examples = 0, []
        for p in eimgs:
            fp = Path(p)
            if not fp.exists():
                e_missing += 1
                if len(miss_examples) < 2:
                    miss_examples.append(p)
                continue
            esize[fp.stat().st_size][md5_of(str(fp))] = p
            h = ahash_of(str(fp))
            if h is not None:
                ehashes.append(h)
                ehash_paths.append(p)
        if e_missing:
            print(f"[warn] {eval_path}: {e_missing} referenced images missing "
                  f"on disk, e.g. {miss_examples}")

        for train_path in args.train:
            trecs = load_records(train_path)
            tids, timgs, tprompts, tem_prompts = set(), set(), set(), set()
            for r in trecs:
                sid = (r.get("meta") or {}).get("id")
                if sid:
                    tids.add(sid)
                    tids.add(ANNOT_RE.sub("", sid))
                t = target_image(r)
                if t:
                    timgs.add(t)
                harvest_prompts(r, tprompts)
                tem_prompts |= em_prompts_of(r)

            import numpy as np
            id_hits = eids & tids
            img_hits = []
            hit_eval_imgs = set()
            pend_paths, pend_hashes = [], []
            for p in timgs:
                fp = Path(p)
                if not fp.exists():
                    continue
                bucket = esize.get(fp.stat().st_size)
                if bucket and md5_of(str(fp)) in bucket:
                    img_hits.append((p, bucket[md5_of(str(fp))]))
                    hit_eval_imgs.add(bucket[md5_of(str(fp))])
                    continue
                th = ahash_of(str(fp))
                if th is not None:
                    pend_paths.append(p)
                    pend_hashes.append(th)
            # perceptual match (vectorized): catches the builders' resized
            # copies, which never match by MD5.
            if pend_hashes and ehashes:
                E = np.unpackbits(
                    np.frombuffer(b"".join(ehashes), dtype=np.uint8).reshape(len(ehashes), 64),
                    axis=1).astype(np.int16)                       # (n_e, 512)
                T = np.unpackbits(
                    np.frombuffer(b"".join(pend_hashes), dtype=np.uint8).reshape(len(pend_hashes), 64),
                    axis=1).astype(np.int16)                       # (n_t, 512)
                for s in range(0, len(T), 1024):
                    chunk = T[s:s + 1024]                          # (c, 512)
                    dist = np.abs(chunk[:, None, :] - E[None, :, :]).sum(axis=2)  # (c, n_e)
                    ti, ei = np.where(dist <= 16)
                    for a, b in zip(ti.tolist(), ei.tolist()):
                        img_hits.append((pend_paths[s + a], ehash_paths[b]))
                        hit_eval_imgs.add(ehash_paths[b])
            prompt_hits = eprompts & tprompts
            # A real prompt-level-holdout violation () = a prompt tied
            # to an EvalMuse image on BOTH sides. PAL4VST boilerplate and
            # coincidental captions do not count.
            em_prompt_hits = eem_prompts & tem_prompts

            tag = f"{Path(train_path).parent.name}/{Path(train_path).name} vs {Path(eval_path).name}"
            print(f"--- {tag}: train_rows={len(trecs)} train_imgs={len(timgs)} "
                  f"eval_rows={len(erecs)}")
            print(f"    id overlap: {len(id_hits)}   image overlap: {len(img_hits)} rows "
                  f"/ {len(hit_eval_imgs)} distinct eval images   "
                  f"prompt overlap: {len(prompt_hits)}")
            if id_hits:
                print(f"    e.g. ids: {sorted(id_hits)[:3]}")
            if img_hits:
                print(f"    e.g. images: {img_hits[:3]}")
            if id_hits or img_hits:
                print("    VERDICT: LEAKED (id/image)")
                failures += 1
            elif em_prompt_hits:
                print(f"    VERDICT: LEAKED (EvalMuse prompt families: {len(em_prompt_hits)})")
                print(f"    e.g. EM prompts: {sorted(em_prompt_hits)[:2]}")
                failures += 1
            elif prompt_hits:
                print(f"    verdict: clean ids/images/EM-prompts; {len(prompt_hits)} shared "
                      f"prompt strings (PAL4VST boilerplate / coincidental) — benign")
            else:
                print("    VERDICT: clean")

    if failures:
        print(f"\nLEAKAGE GATE FAILED: {failures} contaminated pair(s)")
        return 1
    print("\nLEAKAGE GATE PASSED: all pairs clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Rebuild the EvalMuse score-only booster (train 9,000 / val 1,000 chat rows)
deterministically from the official EvalMuse-40K annotations.

These rows were
previously appended by finish_clean_data.sh from a server path whose builder
was not in the repository.

Inputs (all public or committed):
  --train-list   official annotations (HF dataset DY-Evalab/EvalMuse,
                 train_list.json)
  --images-root  official images, unzipped (images.zip.part-* from the same
                 HF dataset -> images/<GENERATOR>/<id>.png)
  --ids          manifests/evalmuse_aug_ids.json — the frozen row selection,
                 order and train/val split (derived once from the shipped
                 corpus; committed)
  --system-prompt manifests/evalmuse_aug_system_prompt.txt — the frozen
                 evaluator system prompt these rows were built with (an
                 earlier revision than build_sft_data.SYSTEM_PROMPT;
                 sha1 3894949c277f...)

Score target: round((mean(total_score) - 1) * 2.5), i.e. the 1–5 annotator
mean mapped linearly to 0–10 under Python (banker's) rounding — verified to
reproduce all 10,000 shipped rows exactly.

Output: <output-dir>/{train,val}.json (booster rows only) and flattened
image copies <output-dir>/images_evalmuse/<GENERATOR>_<id>.png, the layout
finish_clean_data.sh consumes.

Verification: --verify-against <dir> byte-compares the rebuilt rows with the
EvalMuse-referencing rows of an existing corpus (e.g. the restored
inspector_data_em) and exits 1 on any difference.
"""

import argparse
import ast
import json
import random
import statistics
import sys
from pathlib import Path

from build_sft_data import resize_and_save  # the corpus-wide 384²-area LANCZOS cap

USER_TEMPLATE = (
    'Generation prompt: "{prompt}"\n'
    'Give the overall quality score of the GENERATED image as "score: <0-10>/10".'
)


def official_index(train_list_path: Path):
    entries = json.loads(train_list_path.read_text())
    return {e["img_path"]: e for e in entries}


def target_score(entry) -> int:
    scores = entry["total_score"]
    if isinstance(scores, str):
        scores = ast.literal_eval(scores)
    mean = statistics.mean(float(s) for s in scores if s is not None)
    return round((mean - 1) * 2.5)


def build_row(entry, image_path: str, system_prompt: str):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": USER_TEMPLATE.format(prompt=entry["prompt"])},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"score: {target_score(entry)}/10"},
            ]},
        ]
    }


def em_rows(path: Path):
    rows = json.loads(path.read_text())
    return [s for s in rows
            if any("images_evalmuse" in (c.get("path", ""))
                   for m in s["messages"] for c in (m.get("content") or [])
                   if isinstance(c, dict))]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-list", type=Path, default=Path("data/evalmuse_hf/train_list.json"))
    p.add_argument("--images-root", type=Path, default=Path("data/evalmuse/images"))
    p.add_argument("--ids", type=Path, default=Path("manifests/evalmuse_aug_ids.json"))
    p.add_argument("--system-prompt", type=Path, default=Path("manifests/evalmuse_aug_system_prompt.txt"))
    p.add_argument("--output-dir", type=Path, default=Path("$DATA_ROOT/evalmuse_aug_rebuild"))
    p.add_argument("--image-root", default=None,
                   help="path prefix embedded in rows (default: <output-dir>/images_evalmuse)")
    p.add_argument("--skip-image-copy", action="store_true")
    p.add_argument("--verify-against", type=Path, default=None,
                   help="corpus dir with {train,val}.json to byte-compare the EM rows against")
    args = p.parse_args()

    index = official_index(args.train_list)
    ids = json.loads(args.ids.read_text())
    system_prompt = args.system_prompt.read_text()
    image_root = args.image_root or str(args.output_dir / "images_evalmuse")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_out = args.output_dir / "images_evalmuse"
    img_out.mkdir(exist_ok=True)

    failures = 0
    for split, img_paths in ids.items():
        rows = []
        for ip in img_paths:
            entry = index.get(ip)
            if entry is None:
                print(f"[{split}] MISSING official entry for {ip}", file=sys.stderr)
                failures += 1
                continue
            flat = ip.replace("/", "_")
            rows.append(build_row(entry, f"{image_root.rstrip('/')}/{flat}", system_prompt))
            if not args.skip_image_copy:
                src = args.images_root / ip
                if not src.exists():
                    print(f"[{split}] MISSING image {src}", file=sys.stderr)
                    failures += 1
                elif not resize_and_save(src, img_out / flat):
                    print(f"[{split}] RESIZE FAILED {src}", file=sys.stderr)
                    failures += 1
        out = args.output_dir / f"{split}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"[{split}] wrote {len(rows)} rows -> {out}")

    if args.verify_against:
        for split in ids:
            ref = em_rows(args.verify_against / f"{split}.json")
            new = json.loads((args.output_dir / f"{split}.json").read_text())
            # compare modulo the image-path prefix, which is deployment-specific
            def canon(rows, root):
                s = json.dumps(rows, ensure_ascii=False, sort_keys=True)
                return s.replace(root.rstrip("/") + "/", "IMAGES/")
            ref_root = next(c["path"] for m in ref[0]["messages"] for c in (m.get("content") or [])
                            if isinstance(c, dict) and c.get("path")).rsplit("/", 1)[0]
            if canon(ref, ref_root) == canon(new, image_root):
                print(f"[{split}] VERIFIED byte-identical ({len(ref)} rows, modulo path prefix)")
            else:
                print(f"[{split}] MISMATCH vs {args.verify_against}", file=sys.stderr)
                failures += 1
        # pixel-level image check on a seeded sample
        ref_imgs = args.verify_against / "images_evalmuse"
        if ref_imgs.is_dir() and not args.skip_image_copy:
            import numpy as np
            from PIL import Image
            names = sorted(p.name for p in ref_imgs.iterdir())
            bad = 0
            for n in random.Random(42).sample(names, min(100, len(names))):
                a = np.asarray(Image.open(ref_imgs / n).convert("RGB"))
                b = np.asarray(Image.open(img_out / n).convert("RGB"))
                if a.shape != b.shape or (a != b).any():
                    print(f"IMAGE PIXEL MISMATCH {n}", file=sys.stderr)
                    bad += 1
            print(f"images: {len(names)} rebuilt; sampled 100 -> {100 - bad} pixel-identical")
            failures += bad

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

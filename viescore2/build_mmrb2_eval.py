#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MMRB2 (arXiv 2512.16899) T2I + Edit subsets → our eval-jsonl schema.

Each of the 1,000 expert preference pairs per subset becomes TWO rows (side
a / side b) scored independently by run_eval under the byte-
identical frozen prompts; analyze_mmrb2.py then pairs the pred_scores and
reports preference accuracy vs the expert "chosen" label.

Edit pairs are materialised into the ImagenWorld directory convention
(<cond>/outputs/m/out.png + <cond>/input/{src image, metadata.json with
cond_images}) so run_eval's derive_input_images feeds the source image
through the exact conditioning-image pipeline the benchmark rows use.
T2I rows are plain single-image rows. Protocol entries MMRB2-T2I /
MMRB2-Edit are copies of the frozen ImagenWorld-TIG / -TIE entries
(additive to configs/eval_protocol.json — existing keys untouched).

  python viescore2/build_mmrb2_eval.py \
      --data $DATA_ROOT/mmrb2_data --layout $DATA_ROOT/mmrb2_eval_layout
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def text_of(content):
    return " ".join(c[1] for c in content if c[0] == "text").strip()


def images_of(content):
    return [c[1] for c in content if c[0] == "image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="$DATA_ROOT/mmrb2_data")
    ap.add_argument("--layout", default="$DATA_ROOT/mmrb2_eval_layout")
    ap.add_argument("--out", default=str(ROOT / "eval_samples/mmrb2_eval.jsonl"))
    args = ap.parse_args()
    data, layout = Path(args.data), Path(args.layout)

    rows, skipped = [], {"no_image": 0, "multi_image": 0, "missing_file": 0}
    for subset, source in (("t2i", "MMRB2-T2I"), ("edit", "MMRB2-Edit")):
        pairs = json.load(open(data / f"{subset}.json"))
        pairs = pairs["pairs"] if isinstance(pairs, dict) else pairs
        for pr in pairs:
            prompt = text_of(pr["prompt_content"])
            cond = [data / p for p in images_of(pr["prompt_content"])]
            for side in ("a", "b"):
                resp = pr[f"response_{side}"]
                imgs = images_of(resp["response_content"])
                if not imgs:
                    skipped["no_image"] += 1
                    continue
                if len(imgs) > 1:
                    skipped["multi_image"] += 1  # keep first, flag in meta
                img = data / imgs[0]
                if not img.exists() or any(not c.exists() for c in cond):
                    skipped["missing_file"] += 1
                    continue
                rid = f"mmrb2_{subset}_{pr['id']}_{side}"
                if subset == "edit" and cond:
                    d = layout / rid
                    (d / "input").mkdir(parents=True, exist_ok=True)
                    (d / "outputs/m").mkdir(parents=True, exist_ok=True)
                    names = []
                    for j, c in enumerate(cond):
                        dst = d / "input" / f"cond{j}{c.suffix}"
                        if not dst.exists():
                            os.symlink(c.resolve(), dst)
                        names.append(dst.name)
                    (d / "input/metadata.json").write_text(
                        json.dumps({"cond_images": names}))
                    out_img = d / "outputs/m/out.png"
                    if not out_img.exists():
                        os.symlink(img.resolve(), out_img)
                    img_path = str(out_img)
                else:
                    img_path = str(img.resolve())
                rows.append({
                    "image": img_path,
                    "instruction": prompt,
                    "response": {"score": None, "text_reason": "",
                                 "visual_reason": None},
                    "meta": {"source": source, "id": rid,
                             "split": "external_mmrb2",
                             "orig": {"pair_id": pr["id"], "side": side,
                                      "chosen": pr.get("chosen", ""),
                                      "prompt_source": pr.get("prompt_source", ""),
                                      "model": resp.get("model_name", ""),
                                      "n_resp_images": len(imgs)}},
                })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}; skipped {skipped}")

    # additive protocol entries (copies of the frozen TIG/TIE entries)
    pcfg_path = ROOT / "configs/eval_protocol.json"
    pcfg = json.loads(pcfg_path.read_text())
    changed = False
    for new, src in (("MMRB2-T2I", "ImagenWorld-TIG"),
                     ("MMRB2-Edit", "ImagenWorld-TIE")):
        if new not in pcfg and src in pcfg:
            pcfg[new] = dict(pcfg[src])
            changed = True
    if changed:
        pcfg_path.write_text(json.dumps(pcfg, indent=2) + "\n")
        print("protocol config: added MMRB2-T2I / MMRB2-Edit "
              "(copies of TIG / TIE; frozen entries untouched)")


if __name__ == "__main__":
    main()

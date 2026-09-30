#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SDG detector (arXiv 2606.06113) raw-prediction dumps on our benchmarks.

Runs the released P1n3/sdg-detector-grpo (Qwen3-VL-4B, GRPO) under its NATIVE
contract: the verbatim "thinkpe" question template (extracted at runtime from
the cloned official repo to avoid transcription drift), caption = the row's
instruction, temperature 0, max 2048 new tokens, the repo's min/max pixel
settings. Output per row: the parsed <answer> defect list (box_2d in the
model's [x0,y0,x1,y1] 0-1000 convention + label), parsed with the official
salvage logic (truncated JSON -> per-object regex). Grid conversion happens
in the analyzer.

  python viescore2/eval_sdg.py --input eval_samples/had_eval.jsonl \
      --output eval_results_external/sdg_raw_had.json
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CKPT = "$DATA_ROOT/sdg/detector_grpo"
TEMPLATE_SRC = Path("$DATA_ROOT/sdg/repo/sdg_detector/data_prep/prepare_all_datasets.py")
MIN_PIXELS, MAX_PIXELS = 256 * 32 * 32, 1280 * 32 * 32  # eval_qwen.py defaults


def load_template():
    src = TEMPLATE_SRC.read_text()
    m = re.search(r'SFT_pos_en_TEMPLATE = """(.*?)"""', src, re.S)
    assert m, "SDG template not found in the official repo file"
    return m.group(1)


def parse_answer(text):
    """Replicates eval_qwen.py parse_boxes_from_response: <answer> extraction,
    find('[')..rfind(']') json.loads, then per-object regex salvage (their
    truncation fallback). Returns None only when nothing at all parses."""
    m = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, re.S)
    seg = m.group(1) if m else text
    parsed, json_ok = [], False
    try:
        start, end = seg.find("["), seg.rfind("]")
        if start != -1 and end > start:
            data = json.loads(seg[start:end + 1])
            if isinstance(data, list):
                parsed, json_ok = data, True
    except Exception:
        pass
    if not parsed:
        # official pattern made whitespace-tolerant (model pretty-prints JSON);
        # objects are matched without requiring the closing brace so the
        # complete boxes emitted before a 2048-token cutoff are recovered
        pat = (r'"box_2d"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)'
               r'\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
               r'\s*,\s*"label"\s*:\s*"([^"]+)"')
        for g in re.findall(pat, seg, re.S):
            parsed.append({"box_2d": [float(g[0]), float(g[1]),
                                      float(g[2]), float(g[3])],
                           "label": g[4]})
    out = []
    for it in parsed:
        if isinstance(it, dict) and isinstance(it.get("box_2d"), list) \
                and len(it["box_2d"]) == 4:
            out.append({"box_2d": [float(v) for v in it["box_2d"]],
                        "label": str(it.get("label", "")).lower()})
    if out or json_ok:  # boxes found, or a genuine (possibly empty) JSON list
        return out
    return None  # nothing parseable -> parse failure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=8,
                    help="batched greedy decode (their eval used SGLang "
                         "continuous batching; single-row HF is ~70s/row)")
    args = ap.parse_args()

    template = load_template()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        CKPT, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    proc = AutoProcessor.from_pretrained(CKPT, min_pixels=MIN_PIXELS,
                                         max_pixels=MAX_PIXELS)

    rows = []
    for line in open(args.input):
        s = json.loads(line)
        p = Path(s["image"])
        rows.append({"id": s["meta"]["id"],
                     "path": str(p if p.is_absolute() else ROOT / p),
                     "caption": s.get("instruction", "")})
    print(f"{len(rows)} rows from {args.input}", flush=True)

    out_path = Path(args.output)
    done = {}
    if out_path.exists():
        try:
            done = {r["id"]: r for r in json.load(open(out_path))["results"]}
            print(f"resuming: {len(done)}", flush=True)
        except Exception:
            pass
    results = list(done.values())
    proc.tokenizer.padding_side = "left"
    todo = [r for r in rows if r["id"] not in done]
    ndone = 0
    for k0 in range(0, len(todo), args.batch):
        chunk = todo[k0:k0 + args.batch]
        imgs, texts = [], []
        for r in chunk:
            img = Image.open(r["path"]).convert("RGB")
            r["_wh"] = [img.width, img.height]
            imgs.append(img)
            msgs = [
                {"role": "system", "content": [{"type": "text",
                    # eval_qwen.py registry fallback (train.constants not released)
                    "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": template.format(caption=r["caption"])}]}]
            texts.append(proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        inputs = proc(text=texts, images=imgs, padding=True,
                      return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=2048,
                                 do_sample=False)
        resps = proc.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)
        for r, resp in zip(chunk, resps):
            results.append({"id": r["id"], "wh": r["_wh"],
                            "defects": parse_answer(resp),
                            "truncated": "</answer>" not in resp,
                            "raw_tail": resp[-300:]})
        ndone += len(chunk)
        if (k0 // args.batch) % 5 == 0 or ndone == len(todo):
            print(f"{ndone}/{len(todo)}", flush=True)
            json.dump({"summary": {"model": "SDG-GRPO (P1n3/sdg-detector-grpo, "
                                            "Qwen3-VL-4B), native thinkpe template, temp 0",
                                   "n": len(results), "input": args.input},
                       "results": results}, open(out_path, "w"))
    json.dump({"summary": {"model": "SDG-GRPO (P1n3/sdg-detector-grpo, "
                                    "Qwen3-VL-4B), native thinkpe template, temp 0",
                           "n": len(results), "input": args.input},
               "results": results}, open(out_path, "w"))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()

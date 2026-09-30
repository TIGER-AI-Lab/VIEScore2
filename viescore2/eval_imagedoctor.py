#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ImageDoctor (GYX97/ImageDoctor, Qwen2.5-VL-3B, arXiv:2510.01010) on a
jsonl benchmark in eval_suite format.

Closest published competitor with released weights: predicts 4 scalar scores
in [0, 1] (semantic alignment, aesthetics, plausibility, overall) from the
<answer> block of its CoT output, plus two 288x288 sigmoid heatmaps
(misalignment / artifact) decoded from the hidden states at its
<|misalignment|> / <|artifact|> special tokens (SAM-style prompt-encoder +
mask-decoder heads, per the model-card reference code). Heatmaps are
mean-pooled to a 16x16 grid per row. Rows with meta.source == "COCO-real"
or a missing image file are skipped; rows with null gt (PAL4VST) are scored
but excluded from the Spearman.

The released remote code targets late-4.x transformers (needs
modeling_flash_attention_utils.is_flash_attn_available and the 'default'
rope init, both gone/absent in the venv's 5.x). It runs against a pinned
shim dir that shadows only transformers + kernels; recreate with:

    uv pip install --python ./.venv/bin/python \
        --target $DATA_ROOT/checkpoints/imagedoctor_deps \
        transformers==4.57.1 kernels==0.10.4
    rm -rf $DATA_ROOT/checkpoints/imagedoctor_deps/numpy*   # keep venv numpy

Two model-card deviations, both required to run at all: the processor is
assembled manually (AutoProcessor can't resolve a video processor for the
custom "ImageDoctor" model_type) and cfg.vision_config.initializer_range is
backfilled (missing-key init crashes without it). Non-square images yield a
non-18x18 vision grid; the fixed-size heatmap head then gets the grid
bilinearly interpolated to 18x18 (model card assumes square 512^2 inputs).

    source setup_env.sh
    CUDA_VISIBLE_DEVICES=0 python viescore2/eval_imagedoctor.py \
        --output eval_results_external/imagedoctor_eval_suite.json
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

DEPS_DIR = "$DATA_ROOT/checkpoints/imagedoctor_deps"
if not os.path.isdir(DEPS_DIR):
    sys.exit(f"missing {DEPS_DIR} — recreate it with the uv pip command "
             "in this file's docstring")
sys.path.insert(0, DEPS_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = "GYX97/ImageDoctor"

# Verbatim from the GYX97/ImageDoctor model card (including its typos —
# the model was trained on this exact template).
PROMPT_TMPL = "Given a caption and an image generated based on this caption, please analyze the provided image in detail. Evaluate it on various dimensions including Semantic Alignment (How well the image content corresponds to the caption), Aesthetics (composition, color usage, and overall artistic quality), Plausibility (realism and attention to detail), and Overall Impression (General subjective assessment of the image's quality). For each evaluation dimension, provide a score between 0-1 and provide a concise rationale for the score. Use a chain-of-thought process to detail your reasoning steps, and enclose all potential important areas and detailed reasoning within <think> and </think> tags. The important areas are represented in following format: ” I need to focus on the bounding box area. Proposed regions (xyxy): ..., which is an enumerated list in the exact format:1.[x1,y1,x2,y2];\n2.[x1,y1,x2,y2];\n3.[x1,y1,x2,y2]… Here, x1,y1 is the top-left corner, and x2,y2 is the bottom-right corner. Then, within the <answer> and </answer> tags, summarize your assessment in the following format: \"Semantic Alignment score: ... \nMisalignment Locations: ...\nAesthetic score: ...\nPlausibility score: ... nArtifact Locations: ...\nOverall Impression score: ...\". No additional text is allowed in the answer section.\n\n Your actual evaluation should be based on the quality of the provided image.**\n\nYour task is provided as follows:\nText Caption: [{task_prompt}]"

SCORE_RES = {
    "semantic_alignment": re.compile(r"Semantic Alignment score:\s*([01](?:\.\d+)?)"),
    "aesthetics": re.compile(r"Aesthetic score:\s*([01](?:\.\d+)?)"),
    "plausibility": re.compile(r"Plausibility score:\s*([01](?:\.\d+)?)"),
    "overall": re.compile(r"Overall Impression score:\s*([01](?:\.\d+)?)"),
}


def rank_avg(x):
    """Tie-averaged ranks (matches scipy.stats.rankdata 'average')."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rank_avg(a), rank_avg(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None  # constant array — correlation undefined
    return float(np.corrcoef(ra, rb)[0, 1])


def load_model():
    from transformers import (AutoConfig, AutoImageProcessor,
                              AutoModelForCausalLM, AutoTokenizer,
                              Qwen2VLVideoProcessor, Qwen2_5_VLProcessor)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    processor = Qwen2_5_VLProcessor(
        image_processor=AutoImageProcessor.from_pretrained(
            CKPT, trust_remote_code=True),
        tokenizer=tok,
        video_processor=Qwen2VLVideoProcessor(),
        chat_template=tok.chat_template)
    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    if getattr(cfg.vision_config, "initializer_range", None) is None:
        cfg.vision_config.initializer_range = cfg.initializer_range
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    return model, processor


def parse_scores(decoded):
    m = re.search(r"<answer>(.*?)</answer>", decoded, re.S)
    block = m.group(1) if m else decoded
    scores = {}
    for k, pat in SCORE_RES.items():
        sm = pat.search(block)
        scores[k] = float(sm.group(1)) if sm else None
    return scores, (block.strip() if m else None)


def heatmaps_16(model, inputs, outputs, gen):
    """README heatmap recipe; each map mean-pooled 288x288 -> 16x16."""
    mis_mask = gen[:, 1:] == model.config.misalignment_token_id
    art_mask = gen[:, 1:] == model.config.artifact_token_id
    out = {"misalignment": None, "artifact": None}
    if not (mis_mask.any() or art_mask.any()) or outputs.hidden_states is None:
        return out
    all_gen_h = torch.cat([s[-1] for s in outputs.hidden_states[1:]], dim=1)
    last_hidden = model.text_hidden_fcs[0](all_gen_h)
    image_embeds = model.visual(inputs["pixel_values"].to(model.device),
                                grid_thw=inputs["image_grid_thw"].to(model.device))
    img_hidden = model.image_hidden_fcs[0](image_embeds.unsqueeze(0))
    _, gh, gw = inputs["image_grid_thw"][0].tolist()
    img_hidden = img_hidden.transpose(1, 2).view(1, -1, gh // 2, gw // 2)
    if (gh // 2, gw // 2) != (18, 18):  # non-square input; head is fixed 18x18
        img_hidden = F.interpolate(img_hidden.float(), size=(18, 18),
                                   mode="bilinear").to(img_hidden.dtype)

    def run(text_tokens):
        sparse, dense = model.prompt_encoder(points=None, boxes=None,
                                             masks=None, text_embeds=text_tokens)
        low_res = model.heatmap(image_embeddings=img_hidden,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse.to(img_hidden.dtype),
                                dense_prompt_embeddings=dense,
                                multimask_output=False)
        hm = model.sigmoid(low_res)[0, 0]  # [288, 288] in [0, 1]
        pooled = F.adaptive_avg_pool2d(hm.float()[None, None], (16, 16))[0, 0]
        return [[round(float(v), 4) for v in row] for row in pooled.cpu()]

    for key, mask in (("misalignment", mis_mask), ("artifact", art_mask)):
        if mask.any():
            out[key] = run(last_hidden[mask][:1].unsqueeze(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=str(ROOT / "eval_samples/eval_suite.jsonl"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N kept rows (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-heatmaps", action="store_true")
    args = ap.parse_args()

    model, processor = load_model()
    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    rows, skipped = [], {"coco_real": 0, "missing_image": 0}
    t_start = time.time()
    for s in samples:
        if s["meta"]["source"] == "COCO-real":
            skipped["coco_real"] += 1
            continue
        p = s["image"] if s["image"].startswith("/") else str(ROOT / s["image"])
        if not os.path.exists(p):
            skipped["missing_image"] += 1
            continue
        if args.limit and len(rows) >= args.limit:
            break
        img = Image.open(p).convert("RGB")
        r = math.sqrt(512 * 512 / (img.width * img.height))
        img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))),
                         resample=Image.BICUBIC)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",
             "text": PROMPT_TMPL.format(task_prompt=s["instruction"])}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                use_cache=True, return_dict_in_generate=True,
                output_hidden_states=not args.no_heatmaps)
        gen = outputs.sequences[:, inputs.input_ids.shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
        scores, answer = parse_scores(decoded)
        row = {"id": s["meta"]["id"], "image": s["image"],
               "source": s["meta"]["source"], "gt": s["response"]["score"],
               "scores": scores, "answer": answer}
        if not args.no_heatmaps:
            with torch.no_grad():
                row["heatmap_16"] = heatmaps_16(model, inputs, outputs, gen)
        rows.append(row)
        print(f"[{len(rows)}] {row['id']} gt={row['gt']} "
              + " ".join(f"{k}={v}" for k, v in scores.items()), flush=True)

    scored = [r for r in rows
              if r["gt"] is not None and r["scores"]["overall"] is not None
              and r["scores"]["semantic_alignment"] is not None]
    gt = np.array([r["gt"] for r in scored], dtype=np.float64)

    def rho(key):
        if not scored:
            return None
        v = spearman([r["scores"][key] for r in scored], gt)
        return round(v, 4) if v is not None else None

    out = {"summary": {
        "model": f"ImageDoctor ({CKPT}, released ckpt, greedy)",
        "input": os.path.relpath(args.input, ROOT),
        "n": len(rows),
        "n_scored": len(scored),
        "n_parse_failed": sum(1 for r in rows
                              if any(v is None for v in r["scores"].values())),
        "skipped": skipped,
        "spearman_overall": rho("overall"),
        "spearman_semantic_alignment": rho("semantic_alignment"),
        "seconds_per_sample": round((time.time() - t_start) / max(1, len(rows)), 2),
        "note": ("gt-null rows (PAL4VST) scored but excluded from spearman; "
                 "heatmap_16 = 16x16 mean-pool of the 288x288 sigmoid maps"),
    }, "results": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Released MLLM quality scorers on the E_q 900-row slice (tab:scorecompare).

Fills the three planned chi_0 rows: each model runs under its NATIVE
inference contract (own prompts, own scale — SRCC is scale-invariant):

  qinsight     ByteDance/Q-Insight `score_degradation` (Qwen2.5-VL-7B GRPO).
               Native demo_score.py prompt: rating 1-5 as JSON in <answer>.
               MUST run under the transformers-4.51.3 shim
               ($DATA_ROOT/checkpoints/qinsight_deps: transformers 4.51.3,
               tokenizers 0.21.4, hub 0.30.2, an empty deepspeed stub) with
               use_cache FORCED True after load: the checkpoint's exported
               `use_cache: null` breaks cache init — doubled-KV crash on
               4.49/4.51, silent garbage logits on 4.57 (weights verified
               fine via no-cache manual decode). Decoding uses the
               checkpoint's own generation_config (top_k=1 near-greedy).
  omniquality  yeeeeeyy/VisualScore (OmniQuality-R, Qwen2.5-VL-7B).
               Three native dimension prompts (alignment / technical /
               aesthetic, 0-5 in <answer>); the reported scalar is the mean
               of the three — pre-registered to match the model's
               "all-encompassing" framing, components stored per row.
  qalign       q-future/one-align (Q-Align ICML'24, mPLUG-Owl2).
               model.score(task_="quality") in [1,5]. REQUIRES the
               transformers==4.36.1 shim:
                 uv pip install --python .venv/bin/python \
                   --target $DATA_ROOT/checkpoints/qalign_deps --no-deps \
                   transformers==4.36.1 tokenizers==0.15.2 icecream
                 PYTHONPATH=$DATA_ROOT/checkpoints/qalign_deps python ...

chi_0 contract: generated image only (OmniQuality's alignment dimension also
receives the instruction TEXT, like the VQAScore/PickScore chi_0 rows).
Greedy decoding; resume-safe (rows already in --output are skipped).

  python viescore2/eval_mllm_scorers.py --model qinsight \
      --output eval_results_external/mllm_qinsight_eq900.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HUB = "$DATA_ROOT/hf_cache/hub"
PATHS = {
    "qinsight": f"{HUB}/models--ByteDance--Q-Insight/snapshots",
    "omniquality": f"{HUB}/models--yeeeeeyy--VisualScore/snapshots",
    "qalign": f"{HUB}/models--q-future--one-align/snapshots",
}

QINSIGHT_SYS = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively")
QINSIGHT_Q = (
    "What is your overall rating on the quality of this picture? The rating "
    "should be a float between 1 and 5, rounded to two decimal places, with 1 "
    "representing very poor quality and 5 representing excellent quality. "
    'Return the final answer in JSON format with the following keys: '
    '"rating": The score.')
OMNI_PROMPTS = {
    "alignment": ('Judge the image alignment with the prompt: "{instr}"\n'
                  "Please evaluate how well the image matches the prompt. "
                  "Rate it from 0 to 5 (float, 2 decimals)."),
    "technical": ("Give a technical quality score for this picture between "
                  "0 and 5 (float, two decimal places)."),
    "aesthetic": ("Provide a float rating between 0 and 5 for the overall "
                  "aesthetics of this image, rounded to two places."),
}


def snap(key):
    d = Path(PATHS[key])
    s = sorted(d.glob("*"))[-1]
    return str(s / "score_degradation") if key == "qinsight" else str(s)


def parse_answer_float(text):
    """Float from <answer>...</answer> (JSON rating or bare float), else any
    trailing float — None if nothing parseable."""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    seg = m.group(1) if m else text
    jm = re.search(r'"rating"\s*:\s*"?(-?\d+(?:\.\d+)?)', seg)
    if jm:
        return float(jm.group(1))
    fm = re.findall(r"-?\d+(?:\.\d+)?", seg)
    return float(fm[-1]) if fm else None


def rank_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def srcc(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = rank_avg(a), rank_avg(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def load_qwen(path):
    # Run under the transformers-4.57 shim (scalar_scorer_deps on PYTHONPATH):
    # 5.9's strict config validation rejects these checkpoints' use_cache=null.
    # Native resolution (no pixel cap), matching both models' demo scripts.
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = True            # exported null poisons cache init
    model.generation_config.use_cache = True
    proc = AutoProcessor.from_pretrained(path)
    return model, proc


# qinsight: only cap tokens — its checkpoint generation_config (do_sample
# with top_k=1, temp 0.1, rep 1.05) is near-greedy and deterministic.
# omniquality: plain greedy (verified stable).
GEN_KWARGS = {
    "qinsight": dict(max_new_tokens=1024, use_cache=True),
    "omniquality": dict(max_new_tokens=512, do_sample=False),
}


@torch.no_grad()
def qwen_ask(model, proc, image, question, system=None, gen_kwargs=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append({"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
    out = model.generate(**inputs,
                         **(gen_kwargs or GEN_KWARGS["omniquality"]))
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:],
                             skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PATHS))
    ap.add_argument("--input", default=str(ROOT / "eval_samples/eval_suite.jsonl"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep-null-gt", action="store_true",
                    help="keep rows without a GT score (e.g. MMRB2 pairs — "
                         "only per-row preds are needed; SRCC comes out n/a)")
    args = ap.parse_args()

    rows = []
    for line in open(args.input, encoding="utf-8"):
        if not line.strip():
            continue
        s = json.loads(line)
        if s["response"].get("score") is None and not args.keep_null_gt:
            continue
        if s["meta"]["source"] == "COCO-real":
            continue
        gt_raw = s["response"].get("score")
        rows.append({"id": s["meta"]["id"], "source": s["meta"]["source"],
                     "image": s["image"], "instruction": s.get("instruction", ""),
                     "gt": None if gt_raw is None else float(gt_raw)})
    print(f"slice: {len(rows)} rows", flush=True)

    out_path = Path(args.output)
    done = {}
    if out_path.exists():
        try:
            done = {r["id"]: r for r in json.load(open(out_path))["results"]}
            print(f"resuming: {len(done)} rows already scored", flush=True)
        except Exception:
            pass

    mp = snap(args.model)
    print(f"loading {args.model} from {mp}", flush=True)
    if args.model == "qalign":
        import transformers
        assert transformers.__version__.startswith("4.36"), \
            f"qalign needs the 4.36 shim on PYTHONPATH, got {transformers.__version__}"
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            mp, trust_remote_code=True, attn_implementation="eager",
            torch_dtype=torch.float16, device_map="cuda")
    else:
        model, proc = load_qwen(mp)

    results = list(done.values())
    for k, r in enumerate(rows):
        if r["id"] in done:
            continue
        img = Image.open(ROOT / r["image"]).convert("RGB")
        rec = dict(r)
        try:
            if args.model == "qalign":
                rec["pred"] = float(model.score([img], task_="quality",
                                                input_="image").item())
            elif args.model == "qinsight":
                torch.manual_seed(42)
                resp = qwen_ask(model, proc, img, QINSIGHT_Q, system=QINSIGHT_SYS,
                                gen_kwargs=GEN_KWARGS["qinsight"])
                rec["pred"] = parse_answer_float(resp)
                rec["raw_tail"] = resp[-160:]
            else:  # omniquality: mean of the three native dimensions
                comps = {}
                for dim, tpl in OMNI_PROMPTS.items():
                    q = tpl.format(instr=r["instruction"]) if dim == "alignment" else tpl
                    comps[dim] = parse_answer_float(qwen_ask(model, proc, img, q))
                rec["components"] = comps
                vals = [v for v in comps.values() if v is not None]
                rec["pred"] = float(np.mean(vals)) if vals else None
        except Exception as e:
            rec["pred"] = None
            rec["error"] = repr(e)[:200]
        results.append(rec)
        if (k + 1) % 50 == 0 or k + 1 == len(rows):
            print(f"{args.model} {k+1}/{len(rows)}", flush=True)
            _write(out_path, args, results)
    _write(out_path, args, results)


def _write(out_path, args, results):
    scored = [r for r in results if r.get("pred") is not None]
    with_gt = [r for r in scored if r.get("gt") is not None]
    srccs = {"All": (round(srcc([r["pred"] for r in with_gt],
                                [r["gt"] for r in with_gt]), 3)
                     if with_gt else "n/a (no GT scores)")}
    for src in sorted({r["source"] for r in with_gt}):
        sel = [r for r in with_gt if r["source"] == src]
        srccs[src] = round(srcc([r["pred"] for r in sel],
                                [r["gt"] for r in sel]), 3)
    payload = {"summary": {
        "model": args.model, "checkpoint": snap(args.model),
        "n_slice": 900, "n_scored": len(scored),
        "parse_failures": len(results) - len(scored),
        "srcc_tie_corrected": srccs,
        "protocol": ("chi_0 native-contract scoring on the E_q 900-row slice; "
                     "greedy decoding; SRCC method identical to "
                     "tab:scorecompare (rank-average ties)")},
        "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()

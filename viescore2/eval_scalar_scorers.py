#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Open-weight scalar T2I scorer baselines on eval_suite: ImageReward, PickScore,
HPSv2 and VQAScore, one shared runner (style of eval_fga_our_slice.py).

Scores every eval_suite row with response.score != null and meta.source !=
"COCO-real" (PAL4VST rows are null-gt and drop out; missing images are
skipped), then reports per-scorer tie-averaged-rank Spearman vs the human GT,
overall and per source (RichHF-18K / EvalMuse / ImagenWorld tasks by id
prefix TIG/TIE/SRIG/SRIE/MRIG/MRIE). Scorers are loaded one at a time and
freed before the next.

Checkpoints (all released weights, no finetuning):
  imagereward  THUDM/ImageReward ImageReward.pt   (pip image-reward 1.5)
  pickscore    yuvalkirstain/PickScore_v1          (CLIP-H; processor from
                                                    laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
  hpsv2        xswu/HPSv2 HPS_v2.1_compressed.pt   (ViT-H-14 fork, pip hpsv2 1.2.0)
  vqascore     zhiqiulin/clip-flant5-xl            (t2v-metrics 3.0; the xxl
               variant needs ~22 GB bf16 and GPU 0 is shared, so xl is used)

Runs against a pinned shim dir that shadows the venv's transformers 5.x
(ImageReward's and t2v_metrics' bundled BLIP/BERT code target 4.x); recreate
with:

    uv pip install --python ./.venv/bin/python \
        --target $DATA_ROOT/checkpoints/scalar_scorer_deps \
        transformers==4.57.1 kernels==0.10.4
    uv pip install --python ./.venv/bin/python \
        --target $DATA_ROOT/checkpoints/scalar_scorer_deps --no-deps \
        image-reward hpsv2 t2v-metrics open_clip_torch timm fairscale ftfy \
        braceexpand webdataset clint omegaconf iopath openai-clip wcwidth \
        "antlr4-python3-runtime==4.9.3" decord
    rm -rf $DATA_ROOT/checkpoints/scalar_scorer_deps/numpy*  # keep venv numpy
    # the hpsv2 wheel omits its bundled CLIP BPE vocab; open_clip ships the
    # identical file:
    cp $DATA_ROOT/checkpoints/scalar_scorer_deps/open_clip/bpe_simple_vocab_16e6.txt.gz \
       $DATA_ROOT/checkpoints/scalar_scorer_deps/hpsv2/src/open_clip/

Two shim-era patches, both required: helpers that moved from
transformers.modeling_utils to .pytorch_utils are aliased back (ImageReward /
lavis med.py import them from the old spot), and t2v_metrics is imported via
stub parent packages so its model registry (which pulls llava-onevision,
mplug, tarsier, ... deps we do not need for clip-flant5) never runs.

    source setup_env.sh
    CUDA_VISIBLE_DEVICES=0 python viescore2/eval_scalar_scorers.py
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

DEPS_DIR = "$DATA_ROOT/checkpoints/scalar_scorer_deps"
if not os.path.isdir(DEPS_DIR):
    sys.exit(f"missing {DEPS_DIR} — recreate it with the uv pip commands "
             "in this file's docstring")
sys.path.insert(0, DEPS_DIR)

HF_CACHE = os.environ.get("HF_HOME", "$DATA_ROOT/hf_cache")
os.environ.setdefault("HPS_ROOT", os.path.join(HF_CACHE, "hpsv2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

# transformers 4.57.1 moved these helpers to pytorch_utils; the bundled
# BLIP/BERT code in ImageReward and t2v_metrics imports them from the old spot.
import transformers.modeling_utils as _mu  # noqa: E402
import transformers.pytorch_utils as _pu  # noqa: E402
for _n in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices",
           "prune_linear_layer"):
    setattr(_mu, _n, getattr(_pu, _n))

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ORDER = ["RichHF-18K", "EvalMuse", "TIG", "TIE", "SRIG", "SRIE",
                "MRIG", "MRIE"]


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


def source_group(s):
    src = s["meta"]["source"]
    if src.startswith("ImagenWorld"):
        return s["meta"]["id"].split("_")[0]  # TIG/TIE/SRIG/SRIE/MRIG/MRIE
    return src


def free_gpu(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- scorers ---
def run_imagereward(samples):
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0", device="cuda",
                    download_root=os.path.join(HF_CACHE, "ImageReward"))
    scores = []
    with torch.no_grad():
        for i, s in enumerate(samples):
            img = Image.open(s["path"]).convert("RGB")
            scores.append(float(model.score(s["prompt"], img)))
            if (i + 1) % 200 == 0:
                print(f"  imagereward {i+1}/{len(samples)}", flush=True)
    free_gpu(model)
    return scores, "THUDM/ImageReward ImageReward.pt (ImageReward-v1.0, pip image-reward 1.5)"


def run_pickscore(samples):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained(
        "yuvalkirstain/PickScore_v1").eval().to("cuda")
    scores = []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(samples), bs):
            batch = samples[i:i + bs]
            imgs = [Image.open(s["path"]).convert("RGB") for s in batch]
            image_inputs = processor(images=imgs, return_tensors="pt").to("cuda")
            text_inputs = processor(text=[s["prompt"] for s in batch],
                                    padding=True, truncation=True,
                                    max_length=77, return_tensors="pt").to("cuda")
            img_emb = model.get_image_features(**image_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = model.get_text_features(**text_inputs)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * (txt_emb * img_emb).sum(-1)
            scores.extend(float(v) for v in logits)
            if (i + bs) % 320 < bs:
                print(f"  pickscore {min(i+bs, len(samples))}/{len(samples)}",
                      flush=True)
    free_gpu(model, processor)
    return scores, "yuvalkirstain/PickScore_v1 (processor laion/CLIP-ViT-H-14-laion2B-s32B-b79K)"


def run_hpsv2(samples):
    """img_score.py recipe, but init once: random-init ViT-H-14 then load the
    full HPS_v2.1 state dict (skips the pointless laion2B download)."""
    import huggingface_hub
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14", "", precision="amp", device="cuda", jit=False,
        force_quick_gelu=False, force_custom_text=False,
        force_patch_dropout=False, force_image_size=None,
        pretrained_image=False, image_mean=None, image_std=None,
        light_augmentation=True, aug_cfg={}, output_dict=True,
        with_score_predictor=False, with_region_predictor=False)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
    model.load_state_dict(torch.load(cp, map_location="cpu")["state_dict"])
    model = model.to("cuda").eval()
    tokenizer = get_tokenizer("ViT-H-14")
    scores = []
    with torch.no_grad():
        for i, s in enumerate(samples):
            image = preprocess_val(Image.open(s["path"]).convert("RGB")
                                   ).unsqueeze(0).to("cuda")
            text = tokenizer([s["prompt"]]).to("cuda")
            with torch.cuda.amp.autocast():
                out = model(image, text)
                score = (out["image_features"] @ out["text_features"].T)[0, 0]
            scores.append(float(score))
            if (i + 1) % 200 == 0:
                print(f"  hpsv2 {i+1}/{len(samples)}", flush=True)
    free_gpu(model)
    return scores, "xswu/HPSv2 HPS_v2.1_compressed.pt (ViT-H-14, pip hpsv2 1.2.0)"


def run_vqascore(samples):
    """clip-flant5 via t2v_metrics, importing clip_t5_model directly through
    stub parent packages (the package __init__ imports 18 model families,
    several with unmet heavyweight deps)."""
    import importlib.machinery
    import importlib.util

    def _stub(name, path):
        if name in sys.modules:
            return
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        mod = importlib.util.module_from_spec(spec)
        mod.__path__ = [path]
        sys.modules[name] = mod

    base = os.path.join(DEPS_DIR, "t2v_metrics")
    _stub("t2v_metrics", base)
    _stub("t2v_metrics.models", os.path.join(base, "models"))
    _stub("t2v_metrics.models.vqascore_models",
          os.path.join(base, "models", "vqascore_models"))
    from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model
    model = CLIPT5Model(model_name="clip-flant5-xl", device="cuda",
                        cache_dir=os.path.join(HF_CACHE, "hub"))
    scores = []
    bs = 8
    for i in range(0, len(samples), bs):
        batch = samples[i:i + bs]
        probs = model.forward(images=[s["path"] for s in batch],
                              texts=[s["prompt"] for s in batch])
        scores.extend(float(v) for v in probs)
        if (i + bs) % 160 < bs:
            print(f"  vqascore {min(i+bs, len(samples))}/{len(samples)}",
                  flush=True)
    free_gpu(model)
    return scores, ("zhiqiulin/clip-flant5-xl (t2v-metrics 3.0; xl not xxl — "
                    "xxl needs ~22 GB bf16 and GPU 0 is shared)")


SCORERS = {"imagereward": run_imagereward, "pickscore": run_pickscore,
           "hpsv2": run_hpsv2, "vqascore": run_vqascore}


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=str(ROOT / "eval_samples/eval_suite.jsonl"))
    ap.add_argument("--output", default=str(
        ROOT / "eval_results_external/scalar_scorers.json"))
    ap.add_argument("--scorers", default=",".join(SCORERS),
                    help="comma-separated subset of " + ",".join(SCORERS))
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N kept rows (0 = all)")
    ap.add_argument("--keep-null-gt", action="store_true",
                    help="keep rows without a GT score (e.g. MMRB2 preference "
                         "pairs — only the per-row scalars are needed; SRCC "
                         "columns come out nan)")
    args = ap.parse_args()

    raw = [json.loads(l) for l in open(args.input) if l.strip()]
    samples, skipped = [], {"coco_real": 0, "null_gt": 0, "missing_image": 0}
    for s in raw:
        if s["meta"]["source"] == "COCO-real":
            skipped["coco_real"] += 1
            continue
        if s["response"].get("score") is None and not args.keep_null_gt:
            skipped["null_gt"] += 1
            continue
        p = s["image"] if s["image"].startswith("/") else str(ROOT / s["image"])
        if not os.path.exists(p):
            skipped["missing_image"] += 1
            continue
        gt_raw = s["response"].get("score")
        samples.append({"id": s["meta"]["id"], "image": s["image"], "path": p,
                        "prompt": s["instruction"], "source": source_group(s),
                        "gt": float("nan") if gt_raw is None else float(gt_raw)})
        if args.limit and len(samples) >= args.limit:
            break
    print(f"{len(samples)} rows to score, skipped={skipped}", flush=True)

    gt = np.array([s["gt"] for s in samples], dtype=np.float64)
    rows = [{"id": s["id"], "image": s["image"], "source": s["source"],
             "gt": s["gt"]} for s in samples]
    summary = {}
    for name in args.scorers.split(","):
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        try:
            scores, ckpt = SCORERS[name](samples)
            assert len(scores) == len(samples)
            assert all(np.isfinite(v) for v in scores), "non-finite score"
        except Exception as e:
            traceback.print_exc()
            summary[name] = f"failed: {type(e).__name__}: {e}"
            gc.collect()
            torch.cuda.empty_cache()
            continue
        for r, v in zip(rows, scores):
            r[name] = round(v, 6)
        per_source = {}
        for src in SOURCE_ORDER:
            idx = [i for i, s in enumerate(samples) if s["source"] == src]
            if not idx:
                continue
            rho = spearman([scores[i] for i in idx], gt[idx])
            per_source[src] = {"rho": round(rho, 4) if rho is not None else None,
                               "n": len(idx)}
        overall = spearman(scores, gt)
        summary[name] = {
            "checkpoint": ckpt,
            "overall": round(overall, 4) if overall is not None else None,
            "per_source": per_source,
            "n": len(samples),
            "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps({name: summary[name]}, indent=2), flush=True)

    summary["meta"] = {
        "input": os.path.relpath(args.input, ROOT),
        "filter": 'response.score != null and meta.source != "COCO-real"',
        "n_rows": len(samples),
        "skipped": skipped,
        "metric": "tie-averaged-rank Spearman vs human GT",
    }
    out = {"summary": summary, "results": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

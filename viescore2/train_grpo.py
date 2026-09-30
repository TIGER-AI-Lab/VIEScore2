#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grpo — GRPO reinforcement fine-tuning for the evaluator.

Rationale (see Visual-RFT, arXiv 2503.01785): the SFT objective is token-level
cross-entropy, but the metric we care about is a SET-level localisation score
(cell Dice / IoU). GRPO with a VERIFIABLE reward optimises that metric
directly — the ground-truth 16×16 grid gives an exact, hack-proof reward, no
learned reward model needed. This closes the train/eval objective gap.

Pipeline:
  • Start from the merged SFT checkpoint (RL needs a competent cold-start).
  • Reuse the SFT train.json (multi-image messages) as the prompt source; parse
    its assistant target back into the GT grid for the reward.
  • TRL GRPOTrainer. --full-finetune (the published setting) implies --no-vllm;
    the vLLM colocate path is reachable only in the LoRA configuration.
  • Reward = 1.0 × cell Dice [masked on score-only rows]
             + 0.3 × score accuracy + 0.1 × output format.

Env note: vLLM 0.23 ships CUDA-13 libs; the launchers export LD_LIBRARY_PATH so
they are found. transformers/torch are unchanged.

IMPORTANT — the argparse defaults are NOT the published recipe. The paper's
GRPO runs pass the recipe explicitly:
    python viescore2/train_grpo.py \\
        --policy   $DATA_ROOT/checkpoints/qwen3_8B_viescore2_sft_merged \\
        --train    $DATA_ROOT/viescore2_data_full/train.json \\
        --output   $DATA_ROOT/checkpoints/qwen3_8B_viescore2_rl \\
        --full-finetune --kl-beta 0 --learning-rate 1e-5 \\
        --score-reward-weight 0.3 --num-generations 4 --grad-accum 8 \\
        --max-completion-length 768 --max-samples 2400
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling imports
from run_eval import (  # noqa: E402  (shares EXACT prompt/parse logic)
    SYSTEM_PROMPT,
    parse_grid_from_response,
    parse_channels_from_response,
    parse_score_from_response,
    parse_axis_scores,
    resize_to_train_res,
    _resize_to,
    MAX_PIXELS_REF,
    GRID_SIZE,
)


# ── Reward helpers ────────────────────────────────────────────────────────────

def _completion_text(completion: Any) -> str:
    """Extract assistant text from a TRL completion (str or chat list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{"role":"assistant","content": str|blocks}]
        parts = []
        for msg in completion:
            c = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                parts.extend(b.get("text", "") for b in c if isinstance(b, dict))
        return "\n".join(parts)
    return str(completion)


def _dice_from_grids(
    gt: np.ndarray, pred: Optional[np.ndarray],
    major: Optional[np.ndarray] = None, w_major: float = 2.0,
) -> float:
    """Cell-level Dice. Both-empty (correct clean) → 1.0; unparseable → 0.0.

        Dice = 2 TP_w / (2 TP_w + FP + FN_w)

    ``major`` (coverage weighting): GT cells marked "!" (high annotation
    coverage) count ``w_major``× in TP/FN, so missing a fully-covered cell
    costs more than missing a boundary cell. FP stays unweighted (a false
    alarm is a false alarm). ``major=None``/all-zero gives the unweighted
    cell Dice."""
    if pred is None:
        return 0.0
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    if not gt_b.any() and not pred_b.any():
        return 1.0  # correctly called a clean image clean
    w = np.ones_like(gt, dtype=np.float64)
    if major is not None:
        w = w + (w_major - 1.0) * major.astype(np.float64)
    tp = float((w * (gt_b & pred_b)).sum())
    fp = float((~gt_b & pred_b).sum())
    fn = float((w * (gt_b & ~pred_b)).sum())
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def make_reward_funcs(w_major: float = 2.0):
    """Build the (localization, format, score) reward functions.

    Supervision masking: rows whose target carries NO localization GT
    (``has_loc_gt=False`` — the EvalMuse score-only booster) must produce ZERO
    localization gradient. "Unlabeled" is not "clean": the earlier code built
    an all-zero GT grid for them, so the reward paid 1.0 for "none" and 0.0
    for anything else — actively teaching the model that every EvalMuse image
    is defect-free, the exact convention the 07-22 eval fix abolished. Now the
    localization reward returns a constant for the whole group on such rows
    (constant reward → zero advantage under group scaling → no gradient),
    mirroring how the score reward already masks PAL4VST rows."""

    def reward_localization(prompts=None, completions=None, gt_grid=None,
                            gt_major=None, has_loc_gt=None, **kwargs) -> List[float]:
        out = []
        majors = gt_major if gt_major is not None else [None] * len(completions)
        locs = has_loc_gt if has_loc_gt is not None else [True] * len(completions)
        for comp, g, m, has_loc in zip(completions, gt_grid, majors, locs):
            if not has_loc:
                out.append(0.0)  # no loc GT → constant → no gradient
                continue
            gt = np.asarray(g, dtype=np.uint8)
            major = np.asarray(m, dtype=np.uint8) if m is not None else None
            pred = parse_grid_from_response(_completion_text(comp))
            out.append(_dice_from_grids(gt, pred, major=major, w_major=w_major))
        return out

    def reward_format(prompts=None, completions=None, gt_grid=None,
                      has_loc_gt=None, **kwargs) -> List[float]:
        # +1 if the output matches the TARGET's schema, 0 for garbage. Small
        # weight — just discourages malformed text. Schema-aware: a score-only
        # target (EvalMuse) is well-formed when a score parses; demanding a
        # grid there would penalize the correct SFT format and push the model
        # to append spurious "none" grids to score-only outputs.
        locs = has_loc_gt if has_loc_gt is not None else [True] * len(completions)
        out = []
        for c, has_loc in zip(completions, locs):
            text = _completion_text(c)
            if has_loc:
                ok = parse_grid_from_response(text) is not None
            else:
                ppq, psc = parse_axis_scores(text)
                ok = (parse_score_from_response(text) is not None
                      or (ppq is not None and psc is not None))
            out.append(1.0 if ok else 0.0)
        return out

    def reward_score(prompts=None, completions=None, gt_grid=None,
                     gt_score=None, gt_pq=None, gt_sc=None, **kwargs) -> List[float]:
        # Score lines are ALSO exactly verifiable: 1 - |pred - gt|/10 per
        # line, 0 when missing/unparseable. Dual-axis targets are rewarded
        # per axis (mean of pq and sc accuracies); single-score targets as
        # before. Samples without any GT score (PAL4VST) return 0 for the
        # whole group → zero advantage under group scaling → no gradient.
        n = len(completions)
        gts = gt_score if gt_score is not None else [-1.0] * n
        pqs = gt_pq if gt_pq is not None else [-1.0] * n
        scs = gt_sc if gt_sc is not None else [-1.0] * n
        out = []
        for c, g, qp, qs in zip(completions, gts, pqs, scs):
            text = _completion_text(c)
            if qp is not None and qs is not None and qp >= 0 and qs >= 0:
                ppq, psc = parse_axis_scores(text)
                acc_pq = 0.0 if ppq is None else 1.0 - abs(ppq - qp) / 10.0
                acc_sc = 0.0 if psc is None else 1.0 - abs(psc - qs) / 10.0
                out.append((acc_pq + acc_sc) / 2.0)
            elif g is not None and g >= 0:
                ps = parse_score_from_response(text)
                out.append(0.0 if ps is None else 1.0 - abs(ps - g) / 10.0)
            else:
                out.append(0.0)
        return out

    reward_localization.__name__ = "dice"
    reward_format.__name__ = "format"
    reward_score.__name__ = "score"
    return [reward_localization, reward_format, reward_score]


# ── Dataset (lightweight rows + lazy image load) ─────────────────────────────

def _extract_row(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """From one SFT chat sample → {image_paths, user_text, gt_grid}.

    The GENERATED image is the LAST image; the assistant target (sparse text)
    is parsed back into the 16×16 GT grid for the reward."""
    msgs = sample["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    asst = next(m for m in msgs if m["role"] == "assistant")
    img_paths = [c["path"] for c in user["content"] if c["type"] == "image"]
    user_text = next(c["text"] for c in user["content"] if c["type"] == "text")
    if not img_paths:
        return None
    target = asst["content"][0]["text"]
    grid = parse_grid_from_response(target)
    has_loc_gt = grid is not None
    if grid is None:
        # legit only for score-only targets (no grid section at all); a grid
        # section that FAILS to parse is corrupt data, never "clean"
        if re.search(r"\b(cells?|artifact|misalign)\s*:", target):
            raise ValueError(f"unparseable grid target: {target[:120]!r}")
        # placeholder for dataset-schema uniformity; NEVER used as a reward
        # target — has_loc_gt=False masks the localization reward entirely
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    # Severity ("!") marks from dual-channel targets → importance weights for
    # the reward. Flat targets have no channels → all-zero major grid.
    major = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for ch in parse_channels_from_response(target).values():
        major |= ch["major"]
    gt_score = parse_score_from_response(target)
    gt_pq, gt_sc = parse_axis_scores(target)
    return {"image_paths": img_paths, "user_text": user_text,
            "target_text": target, "has_loc_gt": has_loc_gt,
            "gt_grid": grid.tolist(), "gt_major": major.tolist(),
            "gt_score": -1.0 if gt_score is None else float(gt_score),
            "gt_pq": -1.0 if gt_pq is None else float(gt_pq),
            "gt_sc": -1.0 if gt_sc is None else float(gt_sc)}


def _stratum_key(row: Dict[str, Any]) -> tuple:
    """Stratum = (supervision schema, conditioning, GT-density bucket).

    These are exactly the axes a head-of-file subset can silently skew:
    score-only vs grid-only vs grid+score vs channel rows, multi-image
    (editing/reference) vs single-image, and clean/sparse/dense GT grids."""
    tgt = row["target_text"]
    if not row["has_loc_gt"]:
        schema = "score_only"
    elif re.search(r"^(artifact|misalign):", tgt, re.M):
        schema = "channels"
    elif row["gt_pq"] >= 0 or row["gt_score"] >= 0:
        schema = "grid+score"
    else:
        schema = "grid_only"
    ncells = int(np.asarray(row["gt_grid"], dtype=np.uint8).sum())
    dens = ("0" if ncells == 0 else
            "1-24" if ncells <= 24 else
            "25-96" if ncells <= 96 else "97+")
    cond = "multi" if len(row["image_paths"]) > 1 else "single"
    return (schema, cond, dens)


def stratified_sample(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    """Proportional stratified subsample (largest-remainder allocation),
    deterministic under ``seed``. Replaces the old ``rows[:k]`` head-slice,
    which guaranteed nothing about source/task/clean/density composition.
    Logs the per-stratum allocation so every run records what it trained on."""
    from collections import defaultdict

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_stratum_key(r)].append(r)
    keys = sorted(groups)

    if k <= 0 or k >= len(rows):
        picked = list(rows)
        alloc = {kk: len(groups[kk]) for kk in keys}
    else:
        rng = np.random.default_rng(seed)
        quotas = {kk: len(groups[kk]) * k / len(rows) for kk in keys}
        alloc = {kk: int(quotas[kk]) for kk in keys}
        for kk in sorted(keys, key=lambda x: quotas[x] - alloc[x],
                         reverse=True)[: k - sum(alloc.values())]:
            alloc[kk] += 1
        picked = []
        for kk in keys:
            g = groups[kk]
            for i in rng.permutation(len(g))[: alloc[kk]]:
                picked.append(g[i])
        order = np.random.default_rng(seed + 1).permutation(len(picked))
        picked = [picked[i] for i in order]

    print(f"stratified sample: {len(picked)}/{len(rows)} rows, seed={seed}")
    print(f"  {'schema':<11s} {'cond':<7s} {'gt-cells':<8s} {'picked':>7s} {'pool':>7s}")
    for kk in keys:
        print(f"  {kk[0]:<11s} {kk[1]:<7s} {kk[2]:<8s} {alloc[kk]:>7d} {len(groups[kk]):>7d}")
    n_loc = sum(1 for r in picked if r["has_loc_gt"])
    n_score = sum(1 for r in picked if r["gt_score"] >= 0 or r["gt_pq"] >= 0)
    n_multi = sum(1 for r in picked if len(r["image_paths"]) > 1)
    n_clean = sum(1 for r in picked
                  if r["has_loc_gt"] and not np.asarray(r["gt_grid"]).any())
    print(f"  totals: loc-GT {n_loc}, score-GT {n_score}, "
          f"multi-image {n_multi}, clean-grid {n_clean}")
    return picked


def build_dataset(train_json: str, max_samples: int = 0, seed: int = 42):
    """Returns ``(dataset, rows)`` — rows are kept so the caller can audit
    target token lengths against ``max_completion_length`` before training."""
    from datasets import Dataset
    from PIL import Image

    records = json.loads(Path(train_json).read_text(encoding="utf-8"))
    rows = [r for r in (_extract_row(s) for s in records) if r is not None]
    rows = stratified_sample(rows, max_samples, seed)
    ds = Dataset.from_list(rows)

    def to_prompt(batch):
        prompts = []
        for paths, user_text in zip(batch["image_paths"], batch["user_text"]):
            imgs = []
            for j, p in enumerate(paths):
                im = Image.open(p).convert("RGB")
                # generated image is last → train res; earlier → reference res.
                im = resize_to_train_res(im) if j == len(paths) - 1 else _resize_to(im, MAX_PIXELS_REF)
                imgs.append(im)
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": user_text})
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ])
        # Only emit columns the trainer needs: prompt (with inline images) +
        # the GT fields and supervision mask passed through to the rewards.
        return {"prompt": prompts, "gt_grid": batch["gt_grid"],
                "gt_major": batch["gt_major"], "gt_score": batch["gt_score"],
                "gt_pq": batch["gt_pq"], "gt_sc": batch["gt_sc"],
                "has_loc_gt": batch["has_loc_gt"]}

    ds = ds.with_transform(to_prompt)
    return ds, rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="grpo: GRPO with verifiable cell-level Dice reward")
    p.add_argument("--policy", default="$DATA_ROOT/checkpoints/qwen3_8B_viescore2_sft_merged",
                   help="Cold-start policy = merged SFT checkpoint.")
    p.add_argument("--train", default="$DATA_ROOT/viescore2_data_full/train.json")
    p.add_argument("--output", default="$DATA_ROOT/checkpoints/qwen3_8B_viescore2_rl")
    p.add_argument("--format-reward-weight", type=float, default=0.1,
                   help="Weight of the schema-format reward (lambda_f in the "
                        "reward-decomposition ablation; mainline 0.1).")
    p.add_argument("--score-reward-weight", type=float, default=0.0,
                   help="Weight of the verifiable score-accuracy reward "
                        "(1-|pred-gt|/10). 0 disables it (earlier runs' behaviour).")
    p.add_argument("--major-weight", type=float, default=2.0,
                   help="TP/FN weight of GT severe (\"!\") cells in the "
                        "localization reward. 1.0 = plain unweighted cell Dice "
                        "(importance-weighting ablation).")
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size.")
    p.add_argument("--kl-beta", type=float, default=0.04, help="GRPOConfig.beta (KL coeff).")
    p.add_argument("--seed", type=int, default=42,
                   help="training seed — a second seed backs the RL claims "
                        "with more than one run")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=8)
    # longest GT target in the RL pool is 710 tokens; a lower cap silently
    # zeroes the reward on the densest grids. The run now AUDITS the sampled
    # targets against this cap and refuses to start on overflow (the 448-cap
    # run trained under exactly that silent truncation).
    p.add_argument("--max-completion-length", type=int, default=768)
    p.add_argument("--allow-truncation", action="store_true",
                   help="Proceed even if some GT targets exceed "
                        "--max-completion-length (audited at startup). "
                        "Off by default: silent truncation is a config error.")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=0, help="Cap dataset size (0=all).")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.3,
                   help="vLLM colocate GPU memory fraction (leave room for training).")
    p.add_argument("--no-vllm", action="store_true",
                   help="Fallback: HF generate instead of vLLM (slower; use if the "
                        "vllm0.23↔TRL1.5.1 colocate path fails at runtime).")
    p.add_argument("--load-4bit", action="store_true",
                   help="Load the policy in 4-bit NF4 (QLoRA) so GRPO fits in a "
                        "~24GB free slice. Base ~6GB instead of 16GB; LoRA trains on "
                        "top. Implies --no-vllm (vLLM is driver-blocked here AND can't "
                        "serve a bnb-quantized HF model).")
    p.add_argument("--full-finetune", action="store_true",
                   help="FULL-parameter bf16 GRPO (no LoRA). Stronger than QLoRA but "
                        "needs a whole ~80GB card: uses adamw_8bit + gradient "
                        "checkpointing, and pairs best with --kl-beta 0 (no ref model "
                        "→ saves a forward + memory). Mutually exclusive with --load-4bit.")
    args = p.parse_args()
    if args.full_finetune and args.load_4bit:
        p.error("--full-finetune and --load-4bit are mutually exclusive.")
    if args.full_finetune:
        # vLLM is driver-blocked on this box (bundled cu13 kernels need driver
        # >=580; we have 570). Only --load-4bit forced HF generate before; do the
        # same for full-finetune, else use_vllm stays True and crashes at init.
        args.no_vllm = True

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    print(f"Policy (cold-start): {args.policy}")
    processor = AutoProcessor.from_pretrained(args.policy, trust_remote_code=True)

    # BUGFIX (multi-image GRPO): Qwen3-VL's processor emits an mm_token_type_ids
    # whose image-token count (e.g. 114) disagrees with the real <|image_pad|>
    # count and with image_grid_thw's vision-token count (144). TRL forwards that
    # tensor unchanged on the normal VLM path (it only rebuilds mm_token_type_ids
    # from input_ids in its *tools* path), so get_rope_index builds llm_positions
    # 30 longer than the sequence → "shape mismatch [3,651] vs [3,621]" and the run
    # dies at step ~2 on the first multi-image batch. Fix: rebuild mm_token_type_ids
    # straight from input_ids (image_pad→1, video_pad→2) so the image-token count
    # always matches image_grid_thw. Root cause confirmed in scratchpad/diag_capture.py.
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as _q3vl
        _IMG_PAD_ID = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        _VID_PAD_ID = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        _orig_rope = _q3vl.Qwen3VLModel.get_rope_index

        def _fixed_rope(self, input_ids, image_grid_thw=None, video_grid_thw=None,
                        attention_mask=None, mm_token_type_ids=None, **kw):
            if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
                mm = torch.zeros_like(input_ids)
                mm[input_ids == _IMG_PAD_ID] = 1
                if _VID_PAD_ID is not None:
                    mm[input_ids == _VID_PAD_ID] = 2
                mm_token_type_ids = mm
            return _orig_rope(self, input_ids, image_grid_thw=image_grid_thw,
                              video_grid_thw=video_grid_thw, attention_mask=attention_mask,
                              mm_token_type_ids=mm_token_type_ids, **kw)

        _q3vl.Qwen3VLModel.get_rope_index = _fixed_rope
        print("Patched Qwen3VLModel.get_rope_index: mm_token_type_ids rebuilt from input_ids.")
    except Exception as _e:  # pragma: no cover
        print(f"WARN: could not patch get_rope_index ({_e}); multi-image may crash.")

    quant_config = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        args.no_vllm = True  # vLLM can't serve a bnb model (and is driver-blocked)
        print("4-bit QLoRA: NF4 base (~6GB) + LoRA on top; vLLM off (HF generate).")

    model = ModelCls.from_pretrained(
        args.policy,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map={"": 0} if args.load_4bit else None,
    )
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    dataset, rows = build_dataset(args.train, max_samples=args.max_samples,
                                  seed=args.seed)
    print(f"GRPO dataset: {len(dataset)} prompts")

    # Truncation audit (P0-4): the completion cap must cover every GT target
    # in the sampled pool, else the policy physically cannot emit the densest
    # grids and their reward silently saturates below 1.
    tok = processor.tokenizer
    tlens = sorted(len(tok(r["target_text"]).input_ids) for r in rows)
    pct = lambda q: tlens[min(len(tlens) - 1, int(len(tlens) * q))]  # noqa: E731
    n_over = sum(1 for L in tlens if L > args.max_completion_length)
    print(f"target tokens: p50={pct(.5)} p90={pct(.9)} p95={pct(.95)} "
          f"p99={pct(.99)} max={tlens[-1]}; cap={args.max_completion_length} "
          f"→ {n_over} target(s) over cap")
    if n_over and not args.allow_truncation:
        raise SystemExit(
            f"ABORT: {n_over} GT targets exceed --max-completion-length="
            f"{args.max_completion_length} (max {tlens[-1]} tokens). Raise the "
            f"cap or pass --allow-truncation to accept the bias explicitly.")

    reward_funcs = make_reward_funcs(w_major=args.major_weight)

    if args.full_finetune:
        peft_config = None  # train all parameters (bf16); see optim below
        print("FULL-parameter GRPO: no LoRA, paged_adamw_8bit optimizer.")
    else:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    cfg = GRPOConfig(
        seed=args.seed,
        output_dir=args.output,
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        beta=args.kl_beta,
        max_completion_length=args.max_completion_length,
        reward_weights=[1.0, args.format_reward_weight, args.score_reward_weight],  # Dice dominant,
                                            # format a nudge, score optional
        scale_rewards="group",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.full_finetune else "adamw_torch",
        logging_steps=5,
        save_strategy="steps",      # volatile shared box: checkpoint often so an
        save_steps=25,              # OOM mid-run loses <=25 steps; resume picks up
        save_total_limit=3,
        report_to="none",
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        log_completions=True,
        # FIX (multi-image): the policy occasionally samples a vision special
        # token (image_pad / vision_start / ...) into the completion. TRL builds
        # mm_token_type_ids = (prompt_completion_ids == image_pad) over the WHOLE
        # sequence, so that stray token is mistaken for a real image patch and
        # get_rope_index desyncs → StopIteration / "shape mismatch [3,591] vs
        # [3,590]". Suppressing these ids at generation keeps completions
        # text-only; an evaluator's grid output never legitimately contains image
        # tokens, so this is lossless. (Confirmed via scratchpad/diag_rope.py.)
        generation_kwargs={"suppress_tokens": [151646, 151647, 151652,
                                               151653, 151654, 151655, 151656]},
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    # Resilience on a volatile shared box: if a previous attempt OOM'd mid-run,
    # resume from its latest checkpoint instead of restarting from scratch.
    import glob
    has_ckpt = bool(glob.glob(str(Path(args.output) / "checkpoint-*")))
    if has_ckpt:
        print(f"Found existing checkpoint(s) in {args.output} → resuming.")
    trainer.train(resume_from_checkpoint=has_ckpt)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\nGRPO done. Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

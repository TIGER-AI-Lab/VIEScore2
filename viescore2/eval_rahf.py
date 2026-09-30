#!/usr/bin/env python3
"""RAHF (CVPR'24 Rich Human Feedback) head-to-head on RichHF-test.

The first author's released checkpoint (github.com/youweiliang/RichHF,
ViT-L/16-384 + t5-base multi-head, CC BY-NC 4.0) run under EXACTLY the
ImageDoctor comparison protocol (score_imagedoctor.py): sigmoid heatmaps
mean-pooled to 16x16, union = max(implausibility, misalignment), decision
threshold tuned IN ITS FAVOR on the first 100 benchmark rows (grid
t in 0.01..0.90 step 0.01, micro-F1), scored on the remaining 840 against
the same GT grids our models use; paired bootstrap vs our RL rows on
the identical 840 (2,000 resamples, rng seed 0). RAHF's four score heads
are additionally correlated (tie-averaged Spearman) against the human GT.

Runs against the transformers-4.57 shim dir (RAHF targets 4.32; the venv's
5.9 is too new for its ViT/T5 usage), same mechanism as
eval_scalar_scorers.py.

Usage:
  python viescore2/eval_rahf.py infer   --output eval_results_external/rahf_richhf_test.json
  python viescore2/eval_rahf.py analyze --output eval_results_external/rahf_analysis.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM = "$DATA_ROOT/checkpoints/scalar_scorer_deps"
RAHF_REPO = "$DATA_ROOT/rahf_repo"
CKPT = "$DATA_ROOT/checkpoints/rahf_multihead.pt"
EXT = ROOT / "eval_results_external"
OURS = EXT / "viescore2_rl_richhf_test.json"
N_TUNE = 100


def rank_avg(x):
    import numpy as np
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


def spearman(a, b):
    import numpy as np
    ra, rb = rank_avg(a), rank_avg(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def infer(args):
    sys.path.insert(0, SHIM)       # transformers 4.57 shadows the venv's 5.9
    sys.path.insert(0, RAHF_REPO)  # provides model.RAHF
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor
    from model import RAHF  # noqa: E402  (rahf_repo)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RAHF(vit_model="google/vit-large-patch16-384", t5_model="t5-base",
                 multi_heads=True, patch_size=16, image_size=384)
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    proc = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")

    rows_in = [json.loads(l) for l in open(args.input)]
    out_rows = []
    B = args.batch_size
    with torch.no_grad():
        for s in range(0, len(rows_in), B):
            batch = rows_in[s:s + B]
            imgs = [Image.open(ROOT / r["image"]).convert("RGB") for r in batch]
            px = proc(imgs, return_tensors="pt")["pixel_values"].to(device)
            caps = [r["instruction"] for r in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(px, caps)
            hms = {k: v.float().cpu() for k, v in out["heatmaps"].items()}
            scs = {k: v.float().cpu() for k, v in out["scores"].items()}
            for i, r in enumerate(batch):
                def pool(t):
                    hm = t[i]
                    if hm.dim() == 3:  # (1, H, W)
                        hm = hm[0]
                    p = F.adaptive_avg_pool2d(hm[None, None], (16, 16))[0, 0]
                    return [[round(float(v), 4) for v in row] for row in p]
                out_rows.append({
                    "id": r["meta"]["id"],
                    "heatmap_16": {
                        "artifact": pool(hms["implausibility"]),
                        "misalignment": pool(hms["misalignment"]),
                    },
                    "scores": {k: round(float(scs[k][i]), 4) for k in scs},
                })
            if (s // B) % 10 == 0:
                print(f"{s + len(batch)}/{len(rows_in)}", flush=True)
    out = {"summary": {"model": "RAHF multi-head (youweiliang/RichHF released ckpt)",
                       "checkpoint": CKPT, "n": len(out_rows),
                       "note": "heatmap_16 = 16x16 mean-pool of the sigmoid heatmaps; "
                               "scores in [0,1] (plausibility/alignment/aesthetics/overall)"},
           "results": out_rows}
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output)


def analyze(args):
    import numpy as np
    rahf = json.load(open(EXT / "rahf_richhf_test.json"))["results"]
    ours = {r["id"]: r for r in json.load(open(OURS))["results"]}

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    rows = []
    for r in rahf:
        o = ours.get(r["id"])
        if not o or not r.get("heatmap_16"):
            continue
        a = np.array(r["heatmap_16"]["artifact"], np.float32)
        m = np.array(r["heatmap_16"]["misalignment"], np.float32)
        rows.append((np.maximum(a, m), np.array(o["gt_grid"], np.uint8), o, r))
    tune, rep = rows[:N_TUNE], rows[N_TUNE:]

    def micro(sel, t):
        tp = fp = fn = 0
        for hm, gt, *_ in sel:
            pred = hm > t
            gb = gt.astype(bool)
            tp += int((gb & pred).sum()); fp += int((~gb & pred).sum())
            fn += int((gb & ~pred).sum())
        return prf(tp, fp, fn)

    best_t, best_f1 = None, -1.0
    for t in np.arange(0.01, 0.91, 0.01):
        f1 = micro(tune, t)[2]
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), f1

    p, r, f1 = micro(rep, best_t)
    ious = []
    for hm, gt, *_ in rep:
        gb = gt.astype(bool)
        if not gb.any():
            continue
        pred = hm > best_t
        u = int((gb | pred).sum())
        ious.append(int((gb & pred).sum()) / u if u else 1.0)

    tp = sum(o["cell_tp"] for _, _, o, _ in rep)
    fp = sum(o["cell_fp"] for _, _, o, _ in rep)
    fn = sum(o["cell_fn"] for _, _, o, _ in rep)
    op, orc, of1 = prf(tp, fp, fn)
    oious = [o["grid_iou"] for _, gt, o, _ in rep if gt.astype(bool).any()]

    per = []
    for hm, gt, o, _ in rep:
        pred = hm > best_t
        gb = gt.astype(bool)
        itp = int((gb & pred).sum()); ifp = int((~gb & pred).sum())
        ifn = int((gb & ~pred).sum())
        iiou = None
        if gb.any():
            u = int((gb | pred).sum())
            iiou = (itp / u if u else 1.0)
        per.append((o["cell_tp"], o["cell_fp"], o["cell_fn"],
                    o["grid_iou"] if gb.any() else None, itp, ifp, ifn, iiou))
    rng = np.random.default_rng(0)
    d_f1, d_iou = [], []
    for _ in range(2000):
        k = rng.integers(0, len(per), len(per))
        sel = [per[i] for i in k]
        of = prf(sum(x[0] for x in sel), sum(x[1] for x in sel),
                 sum(x[2] for x in sel))[2]
        rf = prf(sum(x[4] for x in sel), sum(x[5] for x in sel),
                 sum(x[6] for x in sel))[2]
        d_f1.append(of - rf)
        oi = [x[3] for x in sel if x[3] is not None]
        ri = [x[7] for x in sel if x[7] is not None]
        d_iou.append(float(np.mean(oi)) - float(np.mean(ri)))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4),
                    round(float(np.percentile(v, 97.5)), 4)]

    # score heads vs human GT on all rows with both present (tie-averaged Spearman)
    score_rows = [(r, ours[r["id"]]) for r in rahf if r["id"] in ours]
    scores = {}
    for head in ("plausibility", "alignment", "aesthetics", "overall"):
        pred = [x[0]["scores"][head] for x in score_rows]
        for gt_key in ("gt_score", "gt_pq", "gt_sc"):
            gt = [x[1].get(gt_key) for x in score_rows]
            pairs = [(pv, gv) for pv, gv in zip(pred, gt) if gv is not None]
            if len(pairs) > 50:
                scores[f"{head}_vs_{gt_key}"] = round(
                    spearman([a for a, _ in pairs], [b for _, b in pairs]), 4)

    out = {
        "protocol": "identical to imagedoctor_analysis loc_richhf: union=max(a,m), "
                    "threshold tuned for RAHF on first 100 rows, reported on remaining 840, "
                    "paired bootstrap 2000 resamples rng(0)",
        "n_tune": len(tune), "n_report": len(rep),
        "threshold_tuned_for_rahf": best_t, "tune_F1": round(best_f1, 4),
        "rahf": {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
                 "IoU_p": round(float(np.mean(ious)), 4)},
        "viescore2_rl_same_840": {"P": round(op, 4), "R": round(orc, 4),
                                  "F1": round(of1, 4),
                                  "IoU_p": round(float(np.mean(oious)), 4)},
        "ours_minus_rahf_paired": {
            "F1": round(of1 - f1, 4), "F1_CI": ci(d_f1),
            "IoU_p": round(float(np.mean(oious)) - float(np.mean(ious)), 4),
            "IoU_p_CI": ci(d_iou)},
        "rahf_score_srcc": scores,
    }
    json.dump(out, open(args.output, "w"), indent=2)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("infer")
    a.add_argument("--output", default=str(EXT / "rahf_richhf_test.json"))
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--input", default=str(ROOT / "eval_samples/richhf_test.jsonl"),
                   help="rows jsonl (e.g. eval_samples/abhuman_eval.jsonl)")
    b = sub.add_parser("analyze")
    b.add_argument("--output", default=str(EXT / "rahf_analysis.json"))
    args = ap.parse_args()
    (infer if args.cmd == "infer" else analyze)(args)


if __name__ == "__main__":
    main()

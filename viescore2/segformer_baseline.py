#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Non-VLM localization baseline: SegFormer trained on the SAME 16x16 grids.

Answers the reviewer's "do you even need an autoregressive VLM for
localization?" A SegFormer-b0 dense encoder predicts a
per-cell defect probability over the same 16x16 grid the VLM emits as text,
supervised by the IDENTICAL grid GT, and scored with the same cell-F1 on
eval_suite. Matched supervision, matched metric — only the output head differs
(dense classifier vs. autoregressive text).

Data is read straight from the clean core (`viescore2_data`): the assistant
text carries the sparse grid, the last user image is the generated image at
384^2. No pool re-stream, no eval leakage (core is gate-clean).

    python viescore2/segformer_baseline.py train --epochs 8
    python viescore2/segformer_baseline.py eval  --eval-data eval_samples/eval_suite.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from repr_formats import parse_sparse  # noqa: E402
from run_eval import gt_visual_to_grid, prf_from_counts, GRID_SIZE  # noqa: E402

RES = 384
CKPT = "$DATA_ROOT/checkpoints/segformer_b0_grid"


def core_rows(core="$DATA_ROOT/viescore2_data_full", split="train"):
    rows = json.loads((Path(core) / f"{split}.json").read_text())
    out = []
    for s in rows:
        img = None
        for m in s.get("messages", []):
            if m.get("role") == "user":
                ps = [c.get("path") for c in m["content"]
                      if isinstance(c, dict) and c.get("type") == "image"]
                ps = [p for p in ps if p]
                if ps:
                    img = ps[-1]
            if m.get("role") == "assistant":
                txt = " ".join(c.get("text", "") for c in m["content"]
                               if isinstance(c, dict))
        if not img or not Path(img).exists():
            continue
        # union grid = every problematic cell (cells/artifact/misalign merged).
        # Coverage marks are stripped first: parse_sparse stops a row at "!",
        # so "r5: 8,9!,10" would silently lose every cell after the mark.
        g = parse_sparse(txt.replace("!", ""), grid_size=GRID_SIZE)
        if g is None:
            continue  # no grid supervision (score-only row) — not "clean"
        out.append((img, g.astype(np.float32)))
    return out


class GridSet(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img, g = self.rows[i]
        im = Image.open(img).convert("RGB").resize((RES, RES), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
        # imagenet norm
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std, torch.from_numpy(g)


def build_model():
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=1, ignore_mismatched_sizes=True)
    return m


def train(args):
    dev = "cuda"
    rows = core_rows()
    print(f"train rows: {len(rows)}", flush=True)
    dl = DataLoader(GridSet(rows), batch_size=args.batch_size, shuffle=True,
                    num_workers=4, drop_last=True)
    model = build_model().to(dev)
    # class imbalance: defect cells are the minority
    pos = np.mean([g.mean() for _, g in rows])
    pw = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=dev)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for x, g in dl:
            x, g = x.to(dev), g.to(dev)
            logits = model(pixel_values=x).logits          # (B,1,H/4,W/4)
            logits = nn.functional.interpolate(
                logits, size=(GRID_SIZE, GRID_SIZE), mode="bilinear",
                align_corners=False).squeeze(1)             # (B,16,16)
            loss = lossfn(logits, g)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{args.epochs} loss {tot/len(dl):.4f}", flush=True)
    Path(CKPT).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT)
    print("SAVED", CKPT, flush=True)

    # threshold tuned on the held-out val split, never on the benchmark
    val = core_rows(split="val")
    print(f"val rows: {len(val)}", flush=True)
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for x, g in DataLoader(GridSet(val), batch_size=args.batch_size,
                               num_workers=4):
            logits = model(pixel_values=x.to(dev)).logits
            logits = nn.functional.interpolate(
                logits, size=(GRID_SIZE, GRID_SIZE), mode="bilinear",
                align_corners=False).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            gts.append(g.numpy())
    probs = np.concatenate(probs); gts = np.concatenate(gts).astype(bool)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.05):
        pred = probs > t
        _, _, f1 = prf_from_counts(int((gts & pred).sum()),
                                   int((~gts & pred).sum()),
                                   int((gts & ~pred).sum()))
        print(f"  t={t:.2f} val F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), f1
    json.dump({"threshold": best_t, "val_F1": round(best_f1, 4)},
              open(Path(CKPT) / "threshold.json", "w"))
    print(f"TUNED threshold={best_t} (val F1 {best_f1:.4f})", flush=True)


def eval_(args):
    dev = "cuda"
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(CKPT).to(dev).eval()
    tfile = Path(CKPT) / "threshold.json"
    thr = json.load(open(tfile))["threshold"] if tfile.exists() else 0.5
    print(f"threshold: {thr}", flush=True)
    samples = [json.loads(l) for l in open(args.eval_data) if l.strip()]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tp = fp = fn = 0
    per = {}
    clean = {}  # source -> [n_clean_gt, n_clean_flagged]
    results = []  # per-sample records, bootstrap_ci-compatible → paired tests
    for i, s in enumerate(samples):
        path = s.get("image", "")
        if not Path(path).exists():
            continue
        if (s.get("meta") or {}).get("source") == "EvalMuse":
            continue  # no localization GT — "unlabeled" is not "clean"
        # GT grid via the same rasterizer the VLM eval uses
        o = (s.get("meta") or {}).get("orig") or {}
        if o.get("artifact_map_png") or o.get("misalign_map_png"):
            from run_eval import heatmap_png_to_grids
            g = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
            for k in ("artifact_map_png", "misalign_map_png"):
                if o.get(k):
                    g |= heatmap_png_to_grids(o[k])[0]
            gt = g
        else:
            gt = gt_visual_to_grid(s["response"].get("visual_reason") or {})
        im = Image.open(path).convert("RGB").resize((RES, RES), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - mean) / std).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(pixel_values=x).logits
            logits = nn.functional.interpolate(
                logits, size=(GRID_SIZE, GRID_SIZE), mode="bilinear",
                align_corners=False).squeeze()
            pred = (torch.sigmoid(logits) > thr).cpu().numpy().astype(bool)
        gb = gt.astype(bool)
        tp += int((gb & pred).sum()); fp += int((~gb & pred).sum())
        fn += int((gb & ~pred).sum())
        inter = int((gb & pred).sum()); union = int((gb | pred).sum())
        results.append({
            "id": (s.get("meta") or {}).get("id", f"sample_{i}"),
            "source": (s.get("meta") or {}).get("source", "?"),
            "cell_tp": int((gb & pred).sum()),
            "cell_fp": int((~gb & pred).sum()),
            "cell_fn": int((gb & ~pred).sum()),
            "grid_iou": (inter / union) if union else 1.0,
        })
        src = (s.get("meta") or {}).get("source", "?").split("-")[0]
        d = per.setdefault(src, [0, 0, 0])
        d[0] += int((gb & pred).sum()); d[1] += int((~gb & pred).sum()); d[2] += int((gb & ~pred).sum())
        if not gb.any():
            c = clean.setdefault(src, [0, 0])
            c[0] += 1
            c[1] += int(pred.any())
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(samples)}", flush=True)
    p, r, f1 = prf_from_counts(tp, fp, fn)
    summary = {"model": "SegFormer-b0 (dense, same 16x16 GT)", "eval_data": args.eval_data,
               "threshold": thr,
               "cell_P": round(p, 4), "cell_R": round(r, 4), "cell_F1": round(f1, 4)}
    summary["per_source_F1"] = {
        k: round(prf_from_counts(*v)[2], 4) for k, v in sorted(per.items())}
    summary["clean_images_flagged"] = {
        k: {"clean_gt": v[0], "flagged": v[1]} for k, v in sorted(clean.items())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(args.output, "w"), indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=6e-5)
    e = sub.add_parser("eval")
    e.add_argument("--eval-data", default="eval_samples/eval_suite.jsonl")
    e.add_argument("--output", default="eval_results/segformer.json")
    args = ap.parse_args()
    (train if args.cmd == "train" else eval_)(args)


if __name__ == "__main__":
    main()

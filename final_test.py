"""
Evaluation script for YOLOv8m and RT-DETR-L pothole detection.

Computes per-model:
  • TP / FP / FN counts
  • Precision, Recall, F1  (at fixed conf threshold)
  • mAP50  (via PR-curve integration across thresholds)
  • Saves annotated failure images  (FP = red box, FN = blue box)

Usage:
    # evaluate both models (default)
    python evaluate.py

    # evaluate a single model
    python evaluate.py --model yolov8m
    python evaluate.py --model rtdetr-l

    # custom paths
    python evaluate.py --test-images path/to/images --test-labels path/to/labels
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODELS = {
    "yolov8m":  "runs/compare/yolov8m/weights/best.pt",
    "rtdetr-l": "runs/compare/rtdetr-l/weights/best.pt",
}
TEST_IMAGES     = "test/images"
TEST_LABELS     = "test/labels"
CONF_THRESHOLD  = 0.3       # for fixed-threshold P/R/F1
IOU_THRESHOLD   = 0.5       # for TP/FP matching
DEVICE          = 0


# ──────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()) + ["both"], default="both")
    ap.add_argument("--test-images", default=TEST_IMAGES)
    ap.add_argument("--test-labels", default=TEST_LABELS)
    ap.add_argument("--conf",  type=float, default=CONF_THRESHOLD)
    ap.add_argument("--iou",   type=float, default=IOU_THRESHOLD)
    return ap.parse_args()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path: str, w: int, h: int) -> list[list[int]]:
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)
            boxes.append([x1, y1, x2, y2])
    return boxes


def draw_box(img, box, color, label=""):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


def compute_map50(all_confs: list[float], all_tp: list[int],
                  n_gt_total: int) -> float:
    """
    Compute mAP50 by integrating the PR curve over 101 recall thresholds.
    all_confs : confidence scores for every prediction across all images
    all_tp    : 1 if prediction is TP, 0 if FP (same length as all_confs)
    n_gt_total: total number of GT boxes in the dataset
    """
    if n_gt_total == 0 or len(all_confs) == 0:
        return 0.0

    order    = np.argsort(-np.array(all_confs))
    tp_arr   = np.array(all_tp)[order]
    fp_arr   = 1 - tp_arr

    tp_cum   = np.cumsum(tp_arr)
    fp_cum   = np.cumsum(fp_arr)

    recalls     = tp_cum / n_gt_total
    precisions  = tp_cum / (tp_cum + fp_cum + 1e-9)

    # 101-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        prec_at_thr = precisions[recalls >= thr]
        ap += prec_at_thr.max() if prec_at_thr.size > 0 else 0.0
    return ap / 101.0


# ──────────────────────────────────────────────
# CORE EVALUATION
# ──────────────────────────────────────────────
def evaluate(model_name: str, weights: str, args) -> dict:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Weights   : {weights}")
    print(f"{'='*60}")

    if not Path(weights).exists():
        print(f"  ✗ weights not found — skipping")
        return {}

    fail_dir = Path(f"failures/{model_name}")
    (fail_dir / "fp").mkdir(parents=True, exist_ok=True)
    (fail_dir / "fn").mkdir(parents=True, exist_ok=True)

    model       = YOLO(weights)
    img_paths   = sorted(
        p for p in Path(args.test_images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    # accumulators
    TP = FP = FN = 0
    n_gt_total  = 0
    all_confs   = []      # for mAP
    all_tp_flag = []      # for mAP

    # per-image failure tracking (avoid duplicate saves)
    img_fp_boxes: dict[str, list] = defaultdict(list)
    img_fn_boxes: dict[str, list] = defaultdict(list)
    img_cache:    dict[str, np.ndarray] = {}

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_cache[img_path.name] = img.copy()

        label_path = Path(args.test_labels) / (img_path.stem + ".txt")
        gt_boxes   = load_gt_boxes(str(label_path), w, h)
        n_gt_total += len(gt_boxes)

        # ── predict ──────────────────────────────────
        res = model.predict(str(img_path), conf=args.conf,
                            device=DEVICE, verbose=False)[0]

        if res.boxes is not None and len(res.boxes):
            preds     = res.boxes.xyxy.cpu().numpy()      # (N,4)
            confs_arr = res.boxes.conf.cpu().numpy()      # (N,)
        else:
            preds = np.zeros((0, 4))
            confs_arr = np.zeros(0)

        matched_gt  = set()
        matched_pred = set()

        # ── greedy matching (highest-IoU first) ──────
        iou_matrix = np.zeros((len(preds), len(gt_boxes)))
        for pi, pb in enumerate(preds):
            for gi, gb in enumerate(gt_boxes):
                iou_matrix[pi, gi] = compute_iou(pb, gb)

        # sort by IoU descending
        pairs = sorted(
            [(pi, gi) for pi in range(len(preds)) for gi in range(len(gt_boxes))],
            key=lambda x: -iou_matrix[x[0], x[1]]
        )
        for pi, gi in pairs:
            if pi in matched_pred or gi in matched_gt:
                continue
            if iou_matrix[pi, gi] >= args.iou:
                matched_pred.add(pi)
                matched_gt.add(gi)

        # ── count TP / FP / FN ───────────────────────
        for pi, (pb, cf) in enumerate(zip(preds, confs_arr)):
            is_tp = pi in matched_pred
            all_confs.append(float(cf))
            all_tp_flag.append(1 if is_tp else 0)
            if is_tp:
                TP += 1
            else:
                FP += 1
                img_fp_boxes[img_path.name].append(pb)

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                FN += 1
                img_fn_boxes[img_path.name].append(gt_boxes[gi])

    # ── save annotated failures ───────────────────────
    _save_failures(img_fp_boxes, img_fn_boxes, img_cache,
                   fail_dir, model, all_confs, preds if len(img_paths) else [])

    # ── metrics ──────────────────────────────────────
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    mAP50     = compute_map50(all_confs, all_tp_flag, n_gt_total)

    results = {
        "model":     model_name,
        "images":    len(img_paths),
        "GT_boxes":  n_gt_total,
        "TP":        TP,
        "FP":        FP,
        "FN":        FN,
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "mAP50":     round(mAP50, 4),
    }

    print(f"\n  Images evaluated : {results['images']}")
    print(f"  GT boxes total   : {n_gt_total}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  mAP50     : {mAP50:.4f}")
    print(f"\n  Failure images → {fail_dir}/fp  and  {fail_dir}/fn")

    return results


def _save_failures(fp_map, fn_map, cache, fail_dir, model, all_confs, preds):
    """Save annotated failure images with boxes drawn."""
    # FP images — draw predicted box in red
    for name, boxes in fp_map.items():
        img = cache.get(name)
        if img is None:
            continue
        vis = img.copy()
        for b in boxes:
            draw_box(vis, b, (0, 0, 220), "FP")
        out = fail_dir / "fp" / name
        cv2.imwrite(str(out), vis)

    # FN images — draw missed GT box in blue
    for name, boxes in fn_map.items():
        img = cache.get(name)
        if img is None:
            continue
        vis = img.copy()
        for b in boxes:
            draw_box(vis, b, (220, 60, 0), "FN-missed")
        out = fail_dir / "fn" / name
        cv2.imwrite(str(out), vis)


# ──────────────────────────────────────────────
# COMPARISON TABLE
# ──────────────────────────────────────────────
def print_comparison(all_results: list[dict]):
    if not all_results:
        return
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'mAP50':>8}")
    print(f"  {'-'*66}")
    for r in all_results:
        print(f"  {r['model']:<14} {r['TP']:>6} {r['FP']:>6} {r['FN']:>6} "
              f"{r['Precision']:>8.4f} {r['Recall']:>8.4f} "
              f"{r['F1']:>8.4f} {r['mAP50']:>8.4f}")
    print(f"{'='*70}\n")

    # simple winner call
    if len(all_results) == 2:
        a, b = all_results
        winner = a if a["mAP50"] >= b["mAP50"] else b
        loser  = b if winner == a else a
        diff   = abs(a["mAP50"] - b["mAP50"])
        print(f"    {winner['model']} wins on mAP50 "
              f"(+{diff:.4f} over {loser['model']})")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    to_run = (
        list(MODELS.items())
        if args.model == "both"
        else [(args.model, MODELS[args.model])]
    )

    all_results = []
    for name, weights in to_run:
        r = evaluate(name, weights, args)
        if r:
            all_results.append(r)

    print_comparison(all_results)


if __name__ == "__main__":
    main()

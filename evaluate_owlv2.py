"""
Evaluate fine-tuned OWL-ViT v2 on pothole test set.

Outputs the same metrics as evaluate.py so you can compare all three models:
    YOLOv8m  |  RT-DETR-L  |  OWL-ViT v2

    • TP / FP / FN counts
    • Precision, Recall, F1  (at fixed conf threshold)
    • mAP50  (101-point PR-curve integration)
    • Annotated failure images  (FP=red, FN=blue)

Usage:
    python evaluate_owlv2.py
    python evaluate_owlv2.py --weights checkpoints/owlv2/best.pt
    python evaluate_owlv2.py --zero-shot          # skip fine-tuned weights
    python evaluate_owlv2.py --conf 0.2 --iou 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tvops
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_ID      = "google/owlv2-base-patch16-ensemble"
WEIGHTS       = "checkpoints/owlv2/best.pt"    # fine-tuned weights
TEST_IMAGES   = "test/images"
TEST_LABELS   = "test/labels"
CONF_THRESH   = 0.25
IOU_THRESH    = 0.50
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# text queries — the same one used in training
TEXT_QUERIES  = [["a photo of a pothole on a road"]]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",     default=WEIGHTS,
                    help="Fine-tuned checkpoint. Ignored if --zero-shot.")
    ap.add_argument("--zero-shot",   action="store_true",
                    help="Use base model without fine-tuned weights")
    ap.add_argument("--test-images", default=TEST_IMAGES)
    ap.add_argument("--test-labels", default=TEST_LABELS)
    ap.add_argument("--conf",        type=float, default=CONF_THRESH)
    ap.add_argument("--iou",         type=float, default=IOU_THRESH)
    ap.add_argument("--save-json",   action="store_true")
    ap.add_argument("--queries",     nargs="+",
                    default=["a photo of a pothole on a road"],
                    help="Text queries (space-separated). Try adding 'road crack'.")
    return ap.parse_args()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def load_gt_boxes_xyxy(label_path: Path, W: int, H: int) -> list[list[float]]:
    """Load YOLO-format labels → xyxy pixel coordinates."""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            boxes.append([
                (cx - bw/2)*W, (cy - bh/2)*H,
                (cx + bw/2)*W, (cy + bh/2)*H,
            ])
    return boxes


def compute_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0


def draw_box(img, box, color, label=""):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def compute_map50(all_confs, all_tp, n_gt_total) -> float:
    if n_gt_total == 0 or not all_confs:
        return 0.0
    order     = np.argsort(-np.array(all_confs))
    tp_arr    = np.array(all_tp)[order]
    tp_cum    = np.cumsum(tp_arr)
    fp_cum    = np.cumsum(1 - tp_arr)
    recalls   = tp_cum / n_gt_total
    precs     = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at = precs[recalls >= thr]
        ap  += p_at.max() if p_at.size > 0 else 0.0
    return ap / 101.0


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model(args) -> tuple[Owlv2ForObjectDetection, Owlv2Processor]:
    print(f"\n  Loading base model: {MODEL_ID}")
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model     = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)

    if not args.zero_shot:
        ckpt = Path(args.weights)
        if not ckpt.exists():
            print(f"  ⚠  weights not found at {ckpt} — running zero-shot instead")
        else:
            state = torch.load(ckpt, map_location="cpu")
            # handle both raw state-dict and checkpoint dict
            sd = state.get("model", state)
            model.load_state_dict(sd, strict=False)
            print(f"  ✓  Loaded fine-tuned weights from {ckpt}")
    else:
        print("  Running in zero-shot mode (no fine-tuned weights)")

    model = model.to(DEVICE).eval()
    return model, processor


# ──────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, processor, args) -> dict:
    queries   = [args.queries]            # wrap: list[list[str]]
    fail_dir  = Path("failures/owlv2")
    (fail_dir/"fp").mkdir(parents=True, exist_ok=True)
    (fail_dir/"fn").mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in Path(args.test_images).iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    print(f"\n  Test images : {len(img_paths)}")
    print(f"  Conf thresh : {args.conf}   IoU thresh : {args.iou}")
    print(f"  Text queries: {args.queries}\n")

    TP = FP = FN = 0
    n_gt_total   = 0
    all_confs    = []
    all_tp_flag  = []
    all_preds    = []

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        W, H  = image.size

        gt_boxes = load_gt_boxes_xyxy(
            Path(args.test_labels) / (img_path.stem + ".txt"), W, H
        )
        n_gt_total += len(gt_boxes)

        # ── inference ────────────────────────────
        inputs  = processor(text=queries, images=image,
                            return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=DEVICE)
        results      = processor.post_process_grounded_object_detection(
            outputs, threshold=args.conf, target_sizes=target_sizes
        )[0]

        pred_boxes  = results["boxes"].cpu().numpy()    # (N,4) xyxy pixels
        pred_scores = results["scores"].cpu().numpy()   # (N,)
        pred_labels = results["labels"].cpu().numpy()   # (N,) query index

        # ── greedy IoU matching ───────────────────
        iou_matrix  = np.zeros((len(pred_boxes), len(gt_boxes)))
        for pi, pb in enumerate(pred_boxes):
            for gi, gb in enumerate(gt_boxes):
                iou_matrix[pi, gi] = compute_iou(pb, gb)

        pairs = sorted(
            [(pi, gi) for pi in range(len(pred_boxes))
                      for gi in range(len(gt_boxes))],
            key=lambda x: -iou_matrix[x[0], x[1]]
        )
        matched_pred = set()
        matched_gt   = set()
        for pi, gi in pairs:
            if pi in matched_pred or gi in matched_gt:
                continue
            if iou_matrix[pi, gi] >= args.iou:
                matched_pred.add(pi)
                matched_gt.add(gi)

        # ── collect TP / FP / FN ─────────────────
        img_cv       = cv2.imread(str(img_path))
        fp_vis = fn_vis = None

        for pi, (pb, sc) in enumerate(zip(pred_boxes, pred_scores)):
            is_tp = pi in matched_pred
            all_confs.append(float(sc))
            all_tp_flag.append(1 if is_tp else 0)
            if is_tp:
                TP += 1
            else:
                FP += 1
                if fp_vis is None and img_cv is not None:
                    fp_vis = img_cv.copy()
                if fp_vis is not None:
                    draw_box(fp_vis, pb, (0, 0, 220),
                             f"FP {sc:.2f}")

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                FN += 1
                if fn_vis is None and img_cv is not None:
                    fn_vis = img_cv.copy()
                if fn_vis is not None:
                    draw_box(fn_vis, gt_boxes[gi], (220, 60, 0),
                             "FN-missed")

        if fp_vis is not None:
            cv2.imwrite(str(fail_dir/"fp"/img_path.name), fp_vis)
        if fn_vis is not None:
            cv2.imwrite(str(fail_dir/"fn"/img_path.name), fn_vis)

        # record for JSON export
        all_preds.append({
            "image":  img_path.name,
            "n_gt":   len(gt_boxes),
            "n_pred": len(pred_boxes),
            "boxes":  [{"box": b.tolist(), "score": float(s)}
                       for b, s in zip(pred_boxes, pred_scores)],
        })

    # ── metrics ──────────────────────────────────
    P     = TP / (TP + FP + 1e-9)
    R     = TP / (TP + FN + 1e-9)
    F1    = 2*P*R / (P + R + 1e-9)
    mAP50 = compute_map50(all_confs, all_tp_flag, n_gt_total)

    results = {
        "model":     "owlv2-base" + ("" if not args.zero_shot else "-zeroshot"),
        "images":    len(img_paths),
        "GT_boxes":  n_gt_total,
        "TP":        TP,
        "FP":        FP,
        "FN":        FN,
        "Precision": round(P,     4),
        "Recall":    round(R,     4),
        "F1":        round(F1,    4),
        "mAP50":     round(mAP50, 4),
    }

    # ── print ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OWL-ViT v2  —  Final Test Results")
    print(f"{'='*60}")
    print(f"  Images evaluated : {results['images']}")
    print(f"  GT boxes total   : {n_gt_total}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision : {P:.4f}")
    print(f"  Recall    : {R:.4f}")
    print(f"  F1        : {F1:.4f}")
    print(f"  mAP50     : {mAP50:.4f}")
    print(f"\n  Failure images → {fail_dir}/fp  and  {fail_dir}/fn")
    print(f"{'='*60}\n")

    if args.save_json:
        out = Path("failures/owlv2/predictions.json")
        out.write_text(json.dumps(all_preds, indent=2))
        print(f"  Predictions → {out}")

    return results


# ──────────────────────────────────────────────
# COMPARISON TABLE  (paste results from evaluate.py)
# ──────────────────────────────────────────────
def print_three_way(owlv2: dict):
    """
    Paste your YOLOv8m and RT-DETR results here to see the full table.
    Leave as None to print only OWL-ViT v2.
    """
    yolo   = None   # e.g. {"model":"yolov8m",  "mAP50":0.68, "F1":0.67, ...}
    rtdetr = None   # e.g. {"model":"rtdetr-l", "mAP50":0.71, "F1":0.70, ...}

    rows = [r for r in [yolo, rtdetr, owlv2] if r is not None]
    if len(rows) < 2:
        return

    print(f"\n{'='*72}")
    print(f"  THREE-WAY COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Model':<18} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'mAP50':>8}")
    print(f"  {'-'*68}")
    for r in rows:
        print(f"  {r['model']:<18} {r.get('TP','-'):>5} {r.get('FP','-'):>5} "
              f"{r.get('FN','-'):>5} {r.get('Precision',0):>8.4f} "
              f"{r.get('Recall',0):>8.4f} {r.get('F1',0):>8.4f} "
              f"{r.get('mAP50',0):>8.4f}")
    best = max(rows, key=lambda x: x.get("mAP50", 0))
    print(f"\n  🏆  Best mAP50: {best['model']}  ({best['mAP50']:.4f})")
    print(f"{'='*72}\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args            = parse_args()
    model, processor = load_model(args)
    results          = evaluate(model, processor, args)
    print_three_way(results)


if __name__ == "__main__":
    main()

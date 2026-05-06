"""
Fine-tune OWL-ViT v2 (base) on pothole dataset.

  • Model   : google/owlv2-base-patch16-ensemble  (hardcoded)
  • Frozen  : vision backbone + text encoder      (hardcoded)
  • Trained : box_head + class_head only          (hardcoded)
  • Logging : Weights & Biases every step + epoch
  • Saves   : best.pt (by val-F1) + epoch_N.pt every N epochs

Dataset layout (YOLO format — same as your YOLO runs):
    train/images/   train/labels/
    val/images/     val/labels/

Usage:
    python train_owlv2.py
    python train_owlv2.py --epochs 50 --batch 4
    python train_owlv2.py --resume checkpoints/owlv2/best.pt

Install:
    pip install transformers scipy torchvision wandb
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.ops as tvops
import wandb
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# ══════════════════════════════════════════════════════
#  HARDCODED — do not change these
# ══════════════════════════════════════════════════════
MODEL_ID   = "google/owlv2-base-patch16-ensemble"
TEXT_QUERY = "a photo of a pothole on a road"
HEAD_KEYS  = ("box_head", "class_head")   # only these layers are trained

# DETR-style loss weights
W_CLS  = 1.0
W_L1   = 5.0
W_GIOU = 2.0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ══════════════════════════════════════════════════════
#  ARGS
# ══════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images",  default="train/images")
    ap.add_argument("--train-labels",  default="train/labels")
    ap.add_argument("--val-images",    default="val/images")
    ap.add_argument("--val-labels",    default="val/labels")
    ap.add_argument("--ckpt-dir",      default="checkpoints/owlv2")
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--batch",         type=int,   default=4)
    ap.add_argument("--lr-head",       type=float, default=1e-4)
    ap.add_argument("--grad-clip",     type=float, default=1.0)
    ap.add_argument("--workers",       type=int,   default=2)
    ap.add_argument("--conf",          type=float, default=0.25)
    ap.add_argument("--iou-match",     type=float, default=0.50)
    ap.add_argument("--ckpt-every",    type=int,   default=5)
    ap.add_argument("--log-every",     type=int,   default=10)
    ap.add_argument("--resume",        default=None)
    ap.add_argument("--wandb-project", default="pothole-comparison")
    ap.add_argument("--run-name",      default=None)
    return ap.parse_args()


# ══════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════
class PotholeDataset(Dataset):
    def __init__(self, img_dir: str, lbl_dir: str):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.samples = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        print(f"    {len(self.samples):>5} images  ←  {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image    = Image.open(img_path).convert("RGB")
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        return {"image": image, "boxes": self._load_labels(lbl_path)}

    @staticmethod
    def _load_labels(path: Path) -> torch.Tensor:
        """YOLO cxcywh-norm → (N, 4) float32."""
        if not path.exists():
            return torch.zeros((0, 4), dtype=torch.float32)
        rows = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                rows.append([cx, cy, w, h])
        return (torch.tensor(rows, dtype=torch.float32)
                if rows else torch.zeros((0, 4), dtype=torch.float32))


def make_collate(processor: Owlv2Processor):
    def collate(batch):
        images   = [s["image"] for s in batch]
        gt_boxes = [s["boxes"] for s in batch]
        encoding = processor(
            text=[[TEXT_QUERY]] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return encoding, gt_boxes
    return collate


# ══════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════
def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    prob  = logits.sigmoid()
    ce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t   = prob * targets + (1 - prob) * (1 - targets)
    w     = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - p_t) ** gamma
    return (w * ce).mean()


def hungarian_match(pred_logits, pred_boxes, gt_boxes):
    with torch.no_grad():
        N, M   = len(pred_logits), len(gt_boxes)
        cls_c  = -(pred_logits.sigmoid().squeeze(-1).unsqueeze(1).expand(N, M))
        l1_c   = torch.cdist(pred_boxes, gt_boxes, p=1)
        pb     = tvops.box_convert(pred_boxes.clamp(0, 1), "cxcywh", "xyxy")
        gb     = tvops.box_convert(gt_boxes.clamp(0, 1),   "cxcywh", "xyxy")
        giou_c = -tvops.generalized_box_iou(pb, gb)
        cost   = W_CLS * cls_c + W_L1 * l1_c + W_GIOU * giou_c
        r, c   = linear_sum_assignment(cost.cpu().numpy())
    return r, c


def compute_loss(outputs, gt_boxes_batch, device):
    logits_b   = outputs.logits       # (B, N, 1)
    pred_box_b = outputs.pred_boxes   # (B, N, 4)  cxcywh norm
    B          = len(gt_boxes_batch)
    loss_cls   = loss_l1 = loss_giou = torch.tensor(0., device=device)

    for i, gt in enumerate(gt_boxes_batch):
        gt          = gt.to(device)
        pred_logits = logits_b[i]
        pred_boxes  = pred_box_b[i]

        if len(gt) == 0:
            loss_cls = loss_cls + focal_bce(
                pred_logits, torch.zeros_like(pred_logits))
            continue

        row, col       = hungarian_match(pred_logits, pred_boxes, gt)
        cls_tgt        = torch.zeros_like(pred_logits)
        cls_tgt[row]   = 1.0
        loss_cls       = loss_cls + focal_bce(pred_logits, cls_tgt)

        mp        = pred_boxes[row].clamp(0, 1)
        mg        = gt[col].clamp(0, 1)
        loss_l1   = loss_l1  + F.l1_loss(mp, mg)
        loss_giou = loss_giou + (
            1 - torch.diag(tvops.generalized_box_iou(
                tvops.box_convert(mp, "cxcywh", "xyxy"),
                tvops.box_convert(mg, "cxcywh", "xyxy"),
            )).mean()
        )

    total = (W_CLS * loss_cls + W_L1 * loss_l1 + W_GIOU * loss_giou) / B
    return total, {
        "loss":      float(total),
        "loss/cls":  float(loss_cls)  / B,
        "loss/l1":   float(loss_l1)   / B,
        "loss/giou": float(loss_giou) / B,
    }


# ══════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════
def _iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    u=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/u if u>0 else 0.


@torch.no_grad()
def run_val(model, processor, img_dir, lbl_dir, conf, iou_thr, device):
    model.eval()
    TP = FP = FN = 0

    for img_path in sorted(p for p in Path(img_dir).iterdir()
                           if p.suffix.lower() in IMG_EXTS):
        image    = Image.open(img_path).convert("RGB")
        W, H     = image.size
        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        gt_raw   = PotholeDataset._load_labels(lbl_path)

        # cxcywh-norm → xyxy-pixel
        gt_xyxy = [[(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H]
                   for cx, cy, bw, bh in gt_raw.tolist()]

        inputs  = processor(text=[[TEXT_QUERY]], images=image,
                            return_tensors="pt").to(device)
        out     = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            out, threshold=conf,
            target_sizes=torch.tensor([[H, W]], device=device)
        )[0]
        preds = results["boxes"].cpu().numpy()

        matched_gt = set()
        for pb in preds:
            hit = False
            for gi, gb in enumerate(gt_xyxy):
                if gi not in matched_gt and _iou(pb, gb) >= iou_thr:
                    matched_gt.add(gi); hit = True; break
            if hit: TP += 1
            else:   FP += 1
        FN += len(gt_xyxy) - len(matched_gt)

    P  = TP / (TP + FP + 1e-9)
    R  = TP / (TP + FN + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)
    model.train()
    return {
        "val/TP": TP, "val/FP": FP, "val/FN": FN,
        "val/precision": round(P,  4),
        "val/recall":    round(R,  4),
        "val/f1":        round(F1, 4),
    }


# ══════════════════════════════════════════════════════
#  CHECKPOINT UTILS
# ══════════════════════════════════════════════════════
def save_ckpt(path, model, opt, scheduler, epoch, step):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "opt":       opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch, "step": step,
    }, path)
    print(f"    💾  saved → {path}")


def maybe_resume(path, model, opt, scheduler):
    if not path or not Path(path).exists():
        return 0, 0
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scheduler.load_state_dict(ck["scheduler"])
    print(f"    ✓  resumed  epoch={ck['epoch']}  step={ck['step']}")
    return ck["epoch"], ck["step"]


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. load base model ────────────────────────────
    print(f"\n[owlv2] Loading {MODEL_ID}")
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model     = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device)

    # ── 2. freeze all → unfreeze head only ───────────
    for param in model.parameters():
        param.requires_grad = False                         # freeze everything

    head_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in HEAD_KEYS):
            param.requires_grad = True                      # unfreeze head
            head_params.append(param)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n[owlv2] Parameters")
    print(f"    Total     : {n_total:>12,}")
    print(f"    Frozen    : {n_total - n_train:>12,}  ← backbone + text encoder")
    print(f"    Trainable : {n_train:>12,}  ← {HEAD_KEYS}")

    # ── 3. datasets + loader ─────────────────────────
    print(f"\n[owlv2] Datasets")
    train_ds = PotholeDataset(args.train_images, args.train_labels)
    PotholeDataset(args.val_images, args.val_labels)          # just to print count

    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
        collate_fn=make_collate(processor),
    )

    # ── 4. optimiser + cosine schedule ───────────────
    opt = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader), eta_min=1e-7
    )

    # ── 5. resume ─────────────────────────────────────
    start_epoch, step = maybe_resume(args.resume, model, opt, scheduler)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── 6. wandb ──────────────────────────────────────
    run_name = args.run_name or f"owlv2-base-headonly-ep{args.epochs}"
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model":         MODEL_ID,
            "frozen":        "backbone + text_encoder",
            "trainable":     "box_head + class_head",
            "text_query":    TEXT_QUERY,
            "epochs":        args.epochs,
            "batch":         args.batch,
            "lr_head":       args.lr_head,
            "grad_clip":     args.grad_clip,
            "conf_val":      args.conf,
            "iou_match":     args.iou_match,
            "trainable_params": n_train,
            "total_params":     n_total,
        },
    )
    print(f"\n[owlv2] wandb run : {run.name}")
    print(f"         url      : {run.url}\n")

    # ══════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════
    best_f1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for encoding, gt_boxes_batch in loader:
            encoding = {k: v.to(device) for k, v in encoding.items()}

            outputs      = model(**encoding)
            loss, log    = compute_loss(outputs, gt_boxes_batch, device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, args.grad_clip)
            opt.step()
            scheduler.step()

            epoch_losses.append(float(loss))

            # ── step log → wandb ─────────────────────
            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                run.log({"step": step, "lr": lr_now, **log})
                print(f"  ep {epoch:>3} | step {step:>6} | "
                      + " ".join(f"{k}={v:.4f}" for k, v in log.items())
                      + f" | lr={lr_now:.2e}")

            step += 1

        elapsed   = time.time() - t0
        mean_loss = sum(epoch_losses) / len(epoch_losses)

        # ── end-of-epoch val ─────────────────────────
        val = run_val(model, processor,
                      args.val_images, args.val_labels,
                      args.conf, args.iou_match, device)

        print(f"\n  ── epoch {epoch:>3} ──  "
              f"loss={mean_loss:.4f}  "
              + "  ".join(f"{k.split('/')[-1]}={v}" for k, v in val.items())
              + f"  ({elapsed:.0f}s)\n")

        # ── epoch log → wandb ─────────────────────────
        run.log({"epoch": epoch, "epoch/loss": mean_loss, **val})

        # ── checkpoint: every N epochs ────────────────
        if (epoch + 1) % args.ckpt_every == 0:
            save_ckpt(f"{args.ckpt_dir}/epoch_{epoch}.pt",
                      model, opt, scheduler, epoch, step)

        # ── checkpoint: best F1 ───────────────────────
        f1_now = val.get("val/f1", 0.0)
        if f1_now > best_f1:
            best_f1 = f1_now
            save_ckpt(f"{args.ckpt_dir}/best.pt",
                      model, opt, scheduler, epoch, step)
            print(f"    🏆  new best val F1 = {best_f1:.4f}")
            run.summary["best_f1"]    = best_f1
            run.summary["best_epoch"] = epoch

    # ── final save ────────────────────────────────────
    save_ckpt(f"{args.ckpt_dir}/last.pt",
              model, opt, scheduler, args.epochs - 1, step)
    print(f"\n[owlv2] Done.  Best val F1 = {best_f1:.4f}")
    print(f"  Weights → {args.ckpt_dir}/best.pt")
    run.finish()


if __name__ == "__main__":
    main()

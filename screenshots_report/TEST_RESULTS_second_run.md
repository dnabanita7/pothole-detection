# Pothole Detection — Final Test-Set Comparison Report

Run of [`final_test.py`](../../final_test.py) on the held-out RDD2022 test split,
comparing the two models trained by [`final_train.py`](../../final_train.py):
**YOLOv8m** vs **RT-DETR-L**. Ground truth filtered to the pothole class
(D40, class id 4) so the metrics are honest for a single-class pothole
model.

This run completes the first item on the "Suggested next steps" list from
[`TEST_RESULTS_first_run.md`](../first_run/TEST_RESULTS_first_run.md).

Generated: 2026-05-06.

---

## 1. Setup

| Item | Value |
| --- | --- |
| Models compared | YOLOv8m + RT-DETR-L (Ultralytics 8.4.46) |
| YOLOv8m weights | `runs/detect_first_run/runs/compare/yolov8m/weights/best.pt` (50 MB) |
| RT-DETR-L weights | `runs/detect/runs/compare/rtdetr-l/weights/best.pt` (63 MB) |
| Training data | `rdd_and_normal_vs_potholes` (single class `pothole`, `nc: 1`) |
| Training schedule | 100 epochs each, `imgsz=640`, global `batch=128` (4-GPU DDP, 32/GPU) |
| YOLOv8m optimizer | SGD, `lr0=1e-2`, warmup 3 epochs, `patience=20` |
| RT-DETR-L optimizer | AdamW, `lr0=1e-4`, warmup 5 epochs, `patience=0` |
| Test source | `RDD2022/RDD_SPLIT/test/images/` (5,758 images) |
| Test labels | `RDD2022/RDD_SPLIT/test/labels/`, filtered to class 4 (D40 / pothole) |
| GT pothole boxes after filter | **951** across the 5,758 images |
| Eval thresholds | `conf ≥ 0.30`, IoU match `≥ 0.50` (greedy, highest-IoU first) |
| Hardware | ROCm GPU inside docker container `shiv_temp` (single-GPU eval) |
| Eval wallclock | ~4.5 min for both models combined |

`final_test.py` was patched to add a `--gt-classes` flag — without it the
script discarded the class id and treated *every* RDD damage label as a
pothole, which would have inflated FN by ~6× and given garbage metrics.

The test inference itself was launched as:

```bash
docker exec shiv_temp bash -c "
cd /home/shivsing/pothole/runs/detect_second_run && \
python /home/shivsing/pothole/final_test.py \
  --model both \
  --test-images /home/shivsing/pothole/RDD2022/RDD_SPLIT/test/images \
  --test-labels /home/shivsing/pothole/RDD2022/RDD_SPLIT/test/labels \
  --conf 0.3 --iou 0.5 --gt-classes 4 \
  2>&1 | tee /home/shivsing/pothole/runs/detect_second_run/final_test_d40.log
"
```

---

## 2. Training summary (for context)

Read from each model's `results.csv`. Note these val metrics come from the
*training* val split (`rdd_and_normal_vs_potholes/val`), not the RDD test
split used in section 3.

### YOLOv8m
| Snapshot | Precision | Recall | val mAP@0.5 | val mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| Best (epoch 97 → `best.pt`) | 0.6218 | 0.4564 | **0.5036** | **0.2119** |
| Final (epoch 99 → `last.pt`) | 0.6308 | 0.4519 | 0.4995 | 0.2109 |

Plots in `runs/detect_first_run/runs/compare/yolov8m/` (`results.png`,
`BoxPR_curve.png`, `confusion_matrix.png`, etc.).

### RT-DETR-L
| Snapshot | Precision | Recall | val mAP@0.5 | val mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| Best (epoch 41 → `best.pt`) | 0.6554 | 0.4899 | **0.5224** | **0.2235** |
| Final (epoch 100 → `last.pt`) | 0.5939 | 0.4908 | 0.4674 | 0.1997 |

Note: RT-DETR-L's best mAP@0.5 (0.5224) was reached *very early* at epoch
41 and the model overfit thereafter — final is 0.057 worse than best. With
`patience=0` the script never early-stopped, so 60 epochs of overfitting
were trained for nothing. A future run should re-enable patience or stop
earlier.

Plots in `runs/detect/runs/compare/rtdetr-l/`.

---

## 3. Final-test results on the RDD test split

5,758 images, 951 pothole GT boxes, `conf=0.3`, IoU=0.5, GT filter = D40.

### 3.1 Headline numbers

| Model | TP | FP | FN | Precision | Recall | F1 | mAP@0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **yolov8m** | 380 | 172 | 571 | **0.6884** | 0.3996 | **0.5057** | 0.3417 |
| **rtdetr-l** | 528 | 842 | 423 | 0.3854 | **0.5552** | 0.4550 | **0.4269** |

(`mAP@0.5` here is computed via 101-point PR-curve interpolation on the
filtered RDD GT, not Ultralytics' built-in `model.val()`, so it is not
directly comparable with the val mAP in section 2.)

### 3.2 As image counts

How many *images* contain at least one of each failure type:

| Bucket | yolov8m | rtdetr-l |
| --- | --- | --- |
| Images with ≥1 FP box | 144 | 457 |
| Images with ≥1 FN box | 353 | 282 |

### 3.3 Output artifacts on the host

```
runs/detect_second_run/
├── RESULTS.md                            # this report's working copy
├── final_test_d40.log                    # full stdout from the eval run
├── runs/compare/                         # symlinks for relative-path reproducibility
│   ├── yolov8m   -> runs/detect_first_run/runs/compare/yolov8m
│   └── rtdetr-l  -> runs/detect/runs/compare/rtdetr-l
└── failures/                             # 293 MB total
    ├── yolov8m/{fp,fn}/                  # 144 + 353 annotated failure images
    └── rtdetr-l/{fp,fn}/                 # 457 + 282 annotated failure images
```

---

## 4. Findings

1. **The two models occupy opposite corners of the precision-recall plane.**
   At a single operating point of `conf=0.3`:
   - YOLOv8m is the **conservative** model — high precision (0.69), low
     recall (0.40). When it fires it's usually right, but it lets ~60 % of
     all pothole GT slip through.
   - RT-DETR-L is the **aggressive** model — low precision (0.39), much
     higher recall (0.56). It catches more potholes but flags 4.9× as many
     false positives as YOLOv8m.

2. **YOLOv8m wins F1 (+0.05); RT-DETR-L wins mAP@0.5 (+0.085).** F1 is a
   *single-threshold* score, mAP@0.5 *integrates* across all conf
   thresholds. So RT-DETR-L's PR curve dominates YOLOv8m's at most
   threshold choices, even though at the chosen `conf=0.3` YOLOv8m has the
   higher F1. Which model "wins" depends entirely on the downstream goal
   and on whether we are willing to tune `conf` per model.

3. **Test mAP@0.5 (0.34 / 0.43) is markedly lower than val mAP@0.5
   (0.50 / 0.52).** Two reasons:
   - The val split came from `rdd_and_normal_vs_potholes` (already
     remapped to single-class pothole), while the test split is the *raw*
     RDD2022 set with 5 damage classes (`alligator / transverse /
     longitudinal / other / pothole`). Even after filtering GT to class 4,
     the test images still contain many crack-like surfaces that the
     models weren't asked to ignore at training time, so they fire FPs
     against blank-but-cracked road.
   - RT-DETR-L specifically overfit hard between epoch 41 and 100 (val
     mAP@0.5 dropped from 0.522 to 0.467), so its `best.pt` is
     ~10 % weaker than it could have been with proper early stopping.

4. **RT-DETR-L's 842 FPs are almost certainly cracks, not noise.** Of the
   5,758 RDD test images, ~3,968 contain non-pothole damage labels
   (D00/D10/D20). Visual inspection of `failures/rtdetr-l/fp/` should
   confirm: most FPs are likely longitudinal/transverse/alligator cracks
   that the single-class model is forced to either ignore or call a
   pothole — and RT-DETR-L's lower precision says it picks "pothole" more
   often. A multi-class retrain (D00–D40) would let the model say "this
   is a crack, not a pothole" and almost certainly cut the FP rate
   substantially.

5. **YOLOv8m's 571 FNs (60 % of pothole GT) is its biggest weakness, and
   `conf=0.3` is probably not its F1-optimal operating point.** Lowering
   `conf` to ~0.15-0.20 would likely claw back recall while paying only
   moderate precision cost. A `conf` sweep is the obvious next experiment.

6. **951 pothole GT / 5,758 images = 16.5 % prevalence.** Pothole imagery
   is a minority of the RDD test split. This is consistent with the
   first-run finding that only 267 / 5,758 images had any detection at
   all when running plain inference.

---

## 5. How to reproduce

```bash
# 1. Symlink the trained weights into runs/compare/{name}/weights/best.pt
#    (final_test.py hardcodes those paths).
mkdir -p runs/detect_second_run/runs/compare
ln -sfn /home/shivsing/pothole/runs/detect_first_run/runs/compare/yolov8m \
        runs/detect_second_run/runs/compare/yolov8m
ln -sfn /home/shivsing/pothole/runs/detect/runs/compare/rtdetr-l \
        runs/detect_second_run/runs/compare/rtdetr-l

# 2. Run final_test.py from the second-run dir so failures/ lands there.
docker exec shiv_temp bash -c "
cd /home/shivsing/pothole/runs/detect_second_run && \
python /home/shivsing/pothole/final_test.py \
  --model both \
  --test-images /home/shivsing/pothole/RDD2022/RDD_SPLIT/test/images \
  --test-labels /home/shivsing/pothole/RDD2022/RDD_SPLIT/test/labels \
  --conf 0.3 --iou 0.5 --gt-classes 4 \
  2>&1 | tee /home/shivsing/pothole/runs/detect_second_run/final_test_d40.log
"
```

---

## 6. Suggested next steps

- [ ] Sweep `conf ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}` for both
      models and pick the F1-optimal operating point per model. Likely
      flips the F1 winner.
- [ ] Sample-inspect 30-50 of `failures/rtdetr-l/fp/` to confirm the
      "FPs are mostly cracks" hypothesis, and tabulate by RDD class.
- [ ] Re-train RT-DETR-L with proper early stopping (`patience>0` or stop
      at epoch ~50) — its `best.pt` was already overfitting by epoch 41.
- [ ] Multi-class retrain (D00, D10, D20, D40) to give both models a
      "this is a crack, not a pothole" option. Expected: massive RT-DETR-L
      FP reduction at minimal recall cost.
- [ ] Build a 2×N visual grid pairing yolov8m vs rtdetr-l predictions on
      the same test image, so the precision-vs-recall character difference
      is obvious at a glance.
- [ ] Run `--gt-classes 4,3` ("pothole + other corruption") to test
      whether the metric gap closes when D30 is also accepted as pothole.

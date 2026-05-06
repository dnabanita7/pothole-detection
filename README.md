# Pothole Detection on Indian and International Road Scenes

A comparative study of four object detectors for pothole detection: a
pretrained Roboflow YOLOv8 (baseline behind a Streamlit demo), a
custom-trained **YOLOv8m**, a custom-trained **RT-DETR-L**, and a
fine-tuned **DETR** (`facebook/detr-resnet-50`). All trained models share
a single domain-specific dataset that we curated by combining
**RDD2022** with the **Normal vs Potholes** set and collapsing five
RDD damage classes to a single `pothole` class.

The full technical write-up is in [`Pothole_Detection_Report.pdf`](Pothole_Detection_Report.pdf)
(also `Pothole_Detection_Report.docx`, with the animated baseline GIF
embedded). One-line summary of the held-out test results:

| Model | Track | Precision | Recall | F1 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Roboflow YOLOv8 (pretrained) | Baseline | 0.7201 | 0.4559 | 0.5583 | Visible FPs on clean roads |
| Custom YOLOv8m (ours) | One-stage CNN | 0.6884 | 0.3996 | 0.5057 | Highest precision among trained models |
| RT-DETR-L (ours) | Real-time transformer | 0.3854 | 0.5552 | 0.4550 | Best mAP@0.5, 2.5 ms / image |
| DETR (fine-tuned, ours) | Transformer | 0.4755 | **0.7595** | **0.5851** | Best recall and mIoU 0.7470 |

Test split: the unmodified RDD2022 test set, 5,758 images / 951 pothole
GT boxes (after filtering RDD class id 4 only via `--gt-classes 4`).

## Team

| Name | Roll number |
| --- | --- |
| Nabanita Dash | 2024701034 |
| Astik Srivastava | 2024701029 |

## Repository layout

```
pothole-detection/
├── README.md                       this file
├── requirements.txt                Python dependencies (CUDA / ROCm notes inside)
├── data.yaml                       Ultralytics dataset descriptor
├── make_better_dataset.py          curates RDD2022 + Normal-vs-Potholes
├── train.py                        custom YOLOv8m training entry point
├── test.py                         single-model inference (test-set predictions)
├── final_train.py                  YOLOv8m + RT-DETR-L joint training driver
├── final_test.py                   patched evaluation: TP/FP/FN/P/R/F1/mAP, --gt-classes filter
├── notebooks/
│   └── detr.ipynb                  DETR fine-tuning notebook (HuggingFace transformers)
├── app/
│   └── streamlit_app.py            Streamlit prototype ("Indian Road Pothole Detector")
├── report/
│   ├── Pothole_Detection_Report.pdf
│   └── Pothole_Detection_Report.docx        (animated GIF embedded)
└── media/                          screenshots, GIFs, W&B exports referenced in the report
    ├── roboflow-pretrained-yolov8.gif
    ├── yolo-metrics-graphs.png
    ├── metrics-yolov8.png
    ├── yolov8-logs.png
    ├── rt-detr-metrics-graphs.png
    ├── rt-detr-recall-precision-graphs.png
    ├── rt-detr-metrics-values-logs.png
    ├── predict_detections_grid.jpg
    ├── Screenshot from 2026-05-06 17-12-56.png    (baseline terminal metrics)
    ├── Screenshot from 2026-05-06 17-19-17.png    (Streamlit Indian road)
    ├── Screenshot from 2026-05-06 17-19-42.png    (Streamlit sunset FP)
    ├── Screenshot from 2026-05-06 17-20-06.png    (Streamlit forest TN)
    └── WhatsApp Image 2026-05-06 at 3.16.53 PM*.jpeg   (DETR qualitative outputs)
```

## Environment

We trained and evaluated the codebase on CUDA without changes.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you are on CUDA replace the torch line in `requirements.txt` with
the wheel that matches your CUDA version (see
<https://pytorch.org/get-started/locally/>). Tested on Python 3.10.

## Dataset preparation

Download RDD2022 from the official challenge page and the
*Normal vs Potholes* set from Kaggle, then run our curation script:

```bash
python make_better_dataset.py
```

This produces `rdd_and_normal_vs_potholes/{train,val}/{images,labels}/`
with single-class labels (`pothole = 0`) and writes `data.yaml`. We
hold out the unmodified RDD test split for evaluation.

## Training

### Custom YOLOv8m

```bash
# 100 epochs, 768x768, batch 16, SGD, cosine LR=1e-2, W&B logging
# Checkpoints: runs/detect/train/weights/{best,last}.pt
```

### YOLOv8m + RT-DETR-L joint training

```bash
python final_train.py
# trains both models back-to-back under matched conditions
# YOLOv8m: SGD lr=1e-2, RT-DETR-L: AdamW lr=1e-4 + 5 warmup epochs
# Logs to W&B project "pothole-comparison"
```

### DETR fine-tuning

Open `notebooks/detr.ipynb`. The notebook uses HuggingFace
`transformers` to fine-tune `facebook/detr-resnet-50` with
`num_labels = 1` for 30 epochs (AdamW lr=1e-5,
`ReduceLROnPlateau` on val loss).

## Evaluation

### Single-model qualitative predictions

```bash
# runs runs/detect/train/weights/best.pt on test/images, saves annotated
# images in runs/detect/predict/
```

### Two-model quantitative comparison

```bash
python final_test.py --model both \
  --test-images RDD2022/RDD_SPLIT/test/images \
  --test-labels RDD2022/RDD_SPLIT/test/labels \
  --conf 0.3 --iou 0.5 --gt-classes 4
```

The `--gt-classes 4` flag is important: RDD's test labels mix five
damage classes; without filtering, every crack box would count as an
unmatched FN against a single-class pothole model and inflate FN by
≈ 6× (recall collapses from 0.40 → 0.07 — see the Ablation in
Section 5 of the report).

The script prints a per-model TP / FP / FN / Precision / Recall / F1 /
mAP@0.5 table and dumps the failure images to
`runs/detect/<model>/{fp,fn}/` for inspection.

## Streamlit demo

```bash
streamlit run app.py
# opens http://localhost:8501
# default model: runs/detect/train/weights/best.pt, conf threshold 0.30
```

The demo accepts both still images and short video clips, draws
detections side-by-side with the input, and aggregates a "Total Unique
Potholes Found" counter across video frames. The GIF in
`media/roboflow-pretrained-yolov8.gif` shows the Roboflow baseline
behind the same UI on a stock dirt-road clip.

## Citing

If this codebase is useful in your work, please cite the underlying
papers (RDD2022, YOLOv8, RT-DETR, DETR — full list in
Section 7 of the report).

## License

Code: MIT. Datasets: their original licenses (RDD2022 — research use;
Roboflow Universe — see project page; Normal vs Potholes — Kaggle
license).

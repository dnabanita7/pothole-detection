import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from roboflow import Roboflow
import supervision as sv

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "MozIyzBJ7nMRkP0x2EmF"
WORKSPACE_ID = "nabanitas-workspace-urt3p"
PROJECT_ID = "pothole-detection-using-yolov8-fsf2h"
VERSION = 1

TEST_IMAGES = "/home/nabs/pothole-detection/pothole-dataset/test/images"
TEST_LABELS = "/home/nabs/pothole-detection/pothole-dataset/test/labels"

CONF_THRESHOLD = 70
IOU_THRESHOLD = 0.5

os.makedirs("failures/fp", exist_ok=True)
os.makedirs("failures/fn", exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
model = project.version(VERSION).model

# -----------------------------
# IOU
# -----------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# -----------------------------
# FIND LABEL FILE (ROBUST)
# -----------------------------
def find_label_file(img_name):
    base = os.path.splitext(img_name)[0]

    # direct match
    candidate = os.path.join(TEST_LABELS, base + ".txt")
    if os.path.exists(candidate):
        return candidate

    # fallback: match prefix (handles .rf hashes)
    for f in os.listdir(TEST_LABELS):
        if f.startswith(base.split(".")[0]) and f.endswith(".txt"):
            return os.path.join(TEST_LABELS, f)

    return None

# -----------------------------
# EVALUATION
# -----------------------------
y_true = []
y_pred = []

total_images = 0

for img_name in os.listdir(TEST_IMAGES):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    total_images += 1

    img_path = os.path.join(TEST_IMAGES, img_name)
    label_path = find_label_file(img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape

    # -----------------------------
    # LOAD GT
    # -----------------------------
    gt_boxes = []

    if label_path and os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, x, y, bw, bh = map(float, parts)

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                gt_boxes.append([x1, y1, x2, y2])

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    result = model.predict(image, confidence=CONF_THRESHOLD).json()
    detections = sv.Detections.from_inference(result)

    pred_boxes = detections.xyxy if len(detections) > 0 else []

    matched_gt = set()

    # -----------------------------
    # MATCH
    # -----------------------------
    for pb in pred_boxes:
        matched = False

        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            if compute_iou(pb, gb) > IOU_THRESHOLD:
                matched = True
                matched_gt.add(i)
                break

        if matched:
            y_true.append(1)
            y_pred.append(1)
        else:
            y_true.append(0)
            y_pred.append(1)
            cv2.imwrite(f"failures/fp/{img_name}", image)

    # -----------------------------
    # MISSED GT
    # -----------------------------
    for i in range(len(gt_boxes)):
        if i not in matched_gt:
            y_true.append(1)
            y_pred.append(0)
            cv2.imwrite(f"failures/fn/{img_name}", image)

# -----------------------------
# METRICS
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# -----------------------------
# OUTPUT
# -----------------------------
print("\nTotal Images:", total_images)

print("\nConfusion Matrix:")
print(cm)

print("\nMetrics:")
print("Precision:", round(precision, 4))
print("Recall:   ", round(recall, 4))
print("F1 Score: ", round(f1, 4))

print("\nFailure folders:")
print("failures/fp ----- false positives")
print("failures/fn ----- false negatives")

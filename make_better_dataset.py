import os
import shutil
import random
from glob import glob

# -----------------------
# PATHS (CHANGE IF NEEDED)
# -----------------------
RDD_IMG_DIR = "RDD2022/RDD_SPLIT/train/images"
RDD_LBL_DIR = "RDD2022/RDD_SPLIT/train/labels"

NORMAL_DIR = "normal_vs_pothole/normal"
POTHOLE_DIR = "normal_vs_pothole/potholes"  # optional

OUTPUT_DIR = "rdd_and_normal_vs_potholes"

# temp staging
IMG_OUT = os.path.join(OUTPUT_DIR, "images_all")
LBL_OUT = os.path.join(OUTPUT_DIR, "labels_all")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# -----------------------
# CONFIG
# -----------------------
# If your RDD uses class 0 = pothole, keep as is
# If not, adjust this
POTHOLE_CLASS_ID = 0

# -----------------------
# 1. COPY RDD (FILTER LABELS)
# -----------------------
print("Processing RDD YOLO dataset...")

for img_path in glob(os.path.join(RDD_IMG_DIR, "*")):
    name = os.path.basename(img_path)
    label_path = os.path.join(RDD_LBL_DIR, name.replace(".jpg", ".txt"))

    new_name = f"rdd_{name}"

    shutil.copy(img_path, os.path.join(IMG_OUT, new_name))

    # process label
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        filtered = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])

            # KEEP ONLY POTHOLES
            if cls == POTHOLE_CLASS_ID:
                filtered.append(line.strip())

        with open(os.path.join(LBL_OUT, new_name.replace(".jpg", ".txt")), "w") as f:
            for l in filtered:
                f.write(l + "\n")
    else:
        # no label → negative
        open(os.path.join(LBL_OUT, new_name.replace(".jpg", ".txt")), "w").close()

# -----------------------
# 2. ADD NORMAL (NEGATIVES)
# -----------------------
print("Adding normal images (negatives)...")

for img_path in glob(os.path.join(NORMAL_DIR, "*")):
    name = os.path.basename(img_path)
    new_name = f"neg_{name}"

    shutil.copy(img_path, os.path.join(IMG_OUT, new_name))

    # empty label
    open(os.path.join(LBL_OUT, new_name.replace(".jpg", ".txt")), "w").close()

# -----------------------
# 3. OPTIONAL: WEAK POTHOLES
# -----------------------
print("Adding weak pothole images (optional)...")

for img_path in glob(os.path.join(POTHOLE_DIR, "*")):
    name = os.path.basename(img_path)
    new_name = f"weak_{name}"

    shutil.copy(img_path, os.path.join(IMG_OUT, new_name))

    # no labels → treated as negative (not ideal)
    open(os.path.join(LBL_OUT, new_name.replace(".jpg", ".txt")), "w").close()

# -----------------------
# 4. TRAIN/VAL SPLIT
# -----------------------
print("Splitting dataset...")

all_images = os.listdir(IMG_OUT)
random.shuffle(all_images)

split = int(0.8 * len(all_images))
train_imgs = all_images[:split]
val_imgs = all_images[split:]

for split_name, split_list in [("train", train_imgs), ("val", val_imgs)]:
    img_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    lbl_dir = os.path.join(OUTPUT_DIR, split_name, "labels")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img in split_list:
        shutil.move(os.path.join(IMG_OUT, img), os.path.join(img_dir, img))

        lbl = img.rsplit(".", 1)[0] + ".txt"
        shutil.move(os.path.join(LBL_OUT, lbl), os.path.join(lbl_dir, lbl))

# cleanup temp
os.rmdir(IMG_OUT)
os.rmdir(LBL_OUT)

print("Dataset ready at:", OUTPUT_DIR)

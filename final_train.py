from ultralytics import YOLO
import wandb
import os

DATA    = "data.yaml"
EPOCHS  = 100
IMGSZ   = 640      #  RT-DETR requires 640 (not 768)
BATCH   = 8
WANDB_PROJECT = "pothole-comparison"

MODELS = [
    ("yolov8m.pt",  "yolov8m"),
    ("rtdetr-l.pt", "rtdetr-l"),
]

def train_model(model_name, run_name):
    is_rtdetr = "rtdetr" in model_name.lower()

    os.environ["WANDB_NAME"] = run_name
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
        config={
            "model":  model_name,
            "epochs": EPOCHS,
            "imgsz":  IMGSZ,
            "batch":  BATCH,
        }
    )

    model = YOLO(model_name)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=640 if is_rtdetr else IMGSZ,  # RT-DETR locked to 640
        batch=BATCH,
        project="runs/compare",
        name=run_name,
        exist_ok=False,
        save=True,
        save_period=5,
        device=0,
        workers=4,
        patience=0 if is_rtdetr else 20,    # RT-DETR ignores patience, set 0
        plots=True,
        # RT-DETR specific: encoder uses longer warmup
        warmup_epochs=5 if is_rtdetr else 3,
        lr0=1e-4 if is_rtdetr else 1e-2,    # RT-DETR prefers lower LR
        optimizer="AdamW" if is_rtdetr else "SGD",
    )

    wandb.finish()


for model_name, run_name in MODELS:
    print(f"\n{'='*50}")
    print(f"  Starting: {run_name}")
    print(f"{'='*50}\n")
    train_model(model_name, run_name)

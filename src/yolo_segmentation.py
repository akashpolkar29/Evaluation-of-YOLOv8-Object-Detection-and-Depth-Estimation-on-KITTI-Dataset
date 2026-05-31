import os
import glob
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # headless — no display needed for batch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
KITTI_ROOT = "../../KITTI-Selection_incl_LiDAR_2026"
IMAGE_DIR  = os.path.join(KITTI_ROOT,
                          "data_object_image_2", "training", "image_2")
OUT_DIR    = ("/home/akash-polkar/Desktop/LRS_Project/"
              "Task 2/Results/Segmentation")

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────
model = YOLO("yolov8n-seg.pt")   # downloads once (~6 MB)
CAR_CLASS_ID = 2                  # COCO class 2 = car
COLORS = [(255,0,0),(0,200,255),(255,165,0),(200,0,200),(0,255,128)]

# ─── Collect all frames ───────────────────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
print(f"Found {len(image_paths)} frames in {IMAGE_DIR}\n")

if len(image_paths) == 0:
    raise FileNotFoundError(f"No PNG files found in: {IMAGE_DIR}")

# ─── Summary log ──────────────────────────────────────────────────────────────
summary_rows = []   # (frame_id, cars_detected)

# ─── Main batch loop ──────────────────────────────────────────────────────────
def process_frame(img_path: str, model, out_dir: str) -> dict:
    """
    Run YOLO segmentation on one image.
    Returns dict: {frame_id, cars_detected, out_path}
    """
    frame_id = os.path.splitext(os.path.basename(img_path))[0]

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [WARN] Cannot read {img_path} — skipping.")
        return {"frame_id": frame_id, "cars_detected": -1, "out_path": ""}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]

    # Inference
    results = model(img_path, verbose=False)[0]

    # Extract car detections
    car_masks, car_boxes, car_scores = [], [], []

    if results.masks is not None:
        for i, cls_id in enumerate(results.boxes.cls.cpu().numpy()):
            if int(cls_id) != CAR_CLASS_ID:
                continue
            mask = results.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            car_masks.append((mask > 0.5).astype(np.uint8))
            car_boxes.append(results.boxes.xyxy[i].cpu().numpy())
            car_scores.append(float(results.boxes.conf[i].cpu()))

    # Build overlay
    overlay = img_rgb.copy()
    for i, mask in enumerate(car_masks):
        color   = COLORS[i % len(COLORS)]
        colored = np.zeros_like(img_rgb)
        colored[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)
        x1, y1, x2, y2 = car_boxes[i].astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"car {car_scores[i]:.2f}", (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save
    out_path = os.path.join(out_dir, f"yolo_{frame_id}.png")
    fig, ax  = plt.subplots(figsize=(16, 5))
    ax.imshow(overlay)
    ax.set_title(f"Frame {frame_id} — YOLOv8 segmentation "
                 f"({len(car_masks)} car{'s' if len(car_masks) != 1 else ''} detected)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # critical: prevents memory leak in long batches

    return {"frame_id": frame_id,
            "cars_detected": len(car_masks),
            "out_path": out_path}


for idx, img_path in enumerate(image_paths):
    frame_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"[{idx+1:>3}/{len(image_paths)}] Processing {frame_id} ...", end=" ")

    result = process_frame(img_path, model, OUT_DIR)
    n      = result["cars_detected"]

    if n >= 0:
        print(f"{n} car(s) → {os.path.basename(result['out_path'])}")
    else:
        print("SKIPPED (read error)")

    summary_rows.append((result["frame_id"], n))

# ─── Write summary CSV ────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "detection_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_id", "cars_detected"])
    writer.writerows(summary_rows)

print(f"\nDone. Processed {len(image_paths)} frames.")
print(f"Results → {OUT_DIR}")
print(f"Summary → {csv_path}")
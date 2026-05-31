import os
import glob
import csv
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from ultralytics import YOLO

from kitti_utils import (load_calib, load_labels,
                         project_lidar_to_image, filter_points_in_image)

# ─── Config ───────────────────────────────────────────────────────────────────
KITTI_ROOT = "../../KITTI-Selection_incl_LiDAR_2026"
OUT_DIR    = "/home/akash-polkar/Desktop/LRS_Project/Task 2/Results/BEV"

os.makedirs(OUT_DIR, exist_ok=True)

CAR_CLASS_ID    = 2
GROUND_Z_THRESH = -1.4

BEV_X_RANGE = (0.0,   60.0)   # forward  [m]
BEV_Y_RANGE = (-15.0, 15.0)   # lateral  [m] — narrowed: car boxes are ~1.6m wide, need zoom

_PALETTE = [
    (30,  144, 255), (255, 140,   0), (180,   0, 200), (0,   210, 180),
    (255, 215,   0), (255,  80, 160), (80,   200,  80), (255,  60,  60),
    (0,  160, 200),  (200, 120,  40), (140, 100, 255),  (60,  220, 180),
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_frame(frame_id):
    calib  = load_calib(
        f"{KITTI_ROOT}/data_object_calib/training/calib/{frame_id}.txt")
    labels = load_labels(
        f"{KITTI_ROOT}/data_object_label_2/training/label_2/{frame_id}.txt",
        class_filter=None)
    pts    = np.fromfile(
        f"{KITTI_ROOT}/data_object_velodyne/training/velodyne/{frame_id}.bin",
        dtype=np.float32).reshape(-1, 4)
    img_path = (f"{KITTI_ROOT}/data_object_image_2/"
                f"training/image_2/{frame_id}.png")
    return calib, labels, pts, img_path


# ─── YOLO ─────────────────────────────────────────────────────────────────────

def get_yolo_car_detections(img_path, model):
    img  = cv2.imread(img_path)
    H, W = img.shape[:2]
    res  = model(img_path, verbose=False)[0]
    masks, boxes = [], []
    if res.masks is not None:
        for i, cls_id in enumerate(res.boxes.cls.cpu().numpy()):
            if int(cls_id) != CAR_CLASS_ID:
                continue
            m = res.masks.data[i].cpu().numpy()
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append((m > 0.5).astype(np.uint8))
            boxes.append(res.boxes.xyxy[i].cpu().numpy())
    return masks, boxes, (H, W)


# ─── Geometry ─────────────────────────────────────────────────────────────────

def box_iou_2d(a, b):
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter  = max(0, xi2-xi1) * max(0, yi2-yi1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def gt_box_to_bev_corners(label):
    h, w, l  = label["dims"]
    loc      = label["loc"]      # [X_cam, Y_cam, Z_cam]
    rot_y    = label["rot_y"]

    # Skip boxes behind camera
    if loc[2] <= 0:
        return None

    cos_r = np.cos(rot_y)
    sin_r = np.sin(rot_y)

    # 4 bottom-face corners in camera object frame (Y=0 = top of box in cam)
    # matches fusion_improved corners_obj rows 0-3
    corners_obj = np.array([
        [ l/2, 0, -w/2],
        [ l/2, 0,  w/2],
        [-l/2, 0,  w/2],
        [-l/2, 0, -w/2],
    ])

    # Rotate: Ry @ corner  (camera frame rotation)
    Ry = np.array([[cos_r, 0, sin_r],
                   [0,     1, 0    ],
                   [-sin_r,0, cos_r]])
    corners_cam = (Ry @ corners_obj.T).T + loc  # (4,3) in camera frame

    # Project to BEV:  forward=Z_cam, lateral=-X_cam
    bev_fwd = corners_cam[:, 2]     # Z_cam   → plot Y-axis
    bev_lat = -corners_cam[:, 0]    # -X_cam  → plot X-axis

    # Return as (lat, fwd) = (x, y) to match matplotlib Polygon/annotate convention.
    # Plot uses xlim=lateral, ylim=forward, so column 0 must be lateral.
    return np.stack([bev_lat, bev_fwd], axis=1)  # (4,2)  [x=lat, y=fwd]


def draw_gt_box_bev(ax, label, color="lime", lw=1.5):
    if label.get("type", "") not in ("Car", "Van"):
        return

    corners = gt_box_to_bev_corners(label)
    if corners is None:
        return

    # Close the polygon
    # Filled box with strong border — boxes are ~1.6m wide, need high contrast
    poly = plt.Polygon(corners, closed=True,
                       edgecolor=color, facecolor=color,
                       linewidth=2.5, alpha=0.35, zorder=4)
    ax.add_patch(poly)
    border = plt.Polygon(corners, closed=True,
                         edgecolor=color, facecolor="none",
                         linewidth=2.5, zorder=6)
    ax.add_patch(border)

    # Heading arrow: box centre → front-face midpoint
    loc   = label["loc"]
    rot_y = label["rot_y"]
    cos_r, sin_r = np.cos(rot_y), np.sin(rot_y)
    Ry    = np.array([[cos_r, 0, sin_r], [0,1,0], [-sin_r,0, cos_r]])

    l = label["dims"][2]
    front_offset_cam = Ry @ np.array([l/2, 0, 0])
    front_cam        = loc + front_offset_cam
    # (x, y) = (lateral, forward) to match plot axes / annotate convention
    centre_bev       = np.array([-loc[0],        loc[2]])
    front_bev        = np.array([-front_cam[0],  front_cam[2]])

    ax.annotate(
        "", xy=(front_bev[0], front_bev[1]),
        xytext=(centre_bev[0], centre_bev[1]),
        arrowprops=dict(arrowstyle="->", color=color,
                        lw=2.5, mutation_scale=12),
        zorder=5
    )


# ─── Per-frame BEV plot ───────────────────────────────────────────────────────

def plot_bev_frame(frame_id, calib, labels, pts,
                   yolo_masks, yolo_boxes, img_shape, out_dir):

    H_img, W_img = img_shape
    u, v, front_mask = project_lidar_to_image(pts, calib)
    valid     = filter_points_in_image(u, v, front_mask, (H_img, W_img, 3))
    u_v       = u[valid].astype(int)
    v_v       = v[valid].astype(int)
    pts_valid = pts[valid]

    car_labels = [l for l in labels if l["type"] in ("Car", "Van")]

    # Build detections
    detections = []
    for yi, (mask, ybox) in enumerate(zip(yolo_masks, yolo_boxes)):
        in_mask = mask[v_v, u_v] == 1
        ypts    = pts_valid[in_mask]
        ypts_f  = ypts[ypts[:, 2] > GROUND_Z_THRESH]

        best_iou, best_label = 0.0, None
        for label in car_labels:
            iou = box_iou_2d(ybox, label["bbox2d"])
            if iou > best_iou:
                best_iou, best_label = iou, label

        dist_est = None
        if len(ypts_f) > 0:
            dist_est = float(np.min(np.linalg.norm(ypts_f[:, :3], axis=1)))

        dist_gt  = float(best_label["loc"][2]) if best_label else None
        dist_err = (abs(dist_est - dist_gt)
                    if dist_est is not None and dist_gt is not None else None)

        detections.append({
            "yolo_idx":  yi,
            "ypts_filt": ypts_f,
            "gt_label":  best_label,
            "iou_2d":    best_iou,
            "dist_est":  dist_est,
            "dist_gt":   dist_gt,
            "dist_err":  dist_err,
        })

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 14), facecolor="black")
    ax.set_facecolor("black")

    # Layer 1 — lidar point cloud coloured by Z height
    in_range = (
        (pts[:, 0] >= BEV_X_RANGE[0]) & (pts[:, 0] <= BEV_X_RANGE[1]) &
        (pts[:, 1] >= BEV_Y_RANGE[0]) & (pts[:, 1] <= BEV_Y_RANGE[1])
    )
    pf = pts[in_range]
    sc = ax.scatter(
        pf[:, 1], pf[:, 0],
        s=0.8, c=pf[:, 2],
        cmap="plasma", vmin=-2.0, vmax=1.5,
        linewidths=0, alpha=0.5, zorder=1
    )

    # Layer 2 — GT boxes (matplotlib clips to axis limits automatically)
    for label in car_labels:
        draw_gt_box_bev(ax, label, color="lime", lw=1.5)

    # Layer 3 — YOLO detection clusters + distance labels
    legend_patches = [
        mpatches.Patch(edgecolor="lime", facecolor="lime",
                       linewidth=2.0, alpha=0.4, label="GT box"),
    ]

    for det in detections:
        yi   = det["yolo_idx"]
        dc   = tuple(c/255.0 for c in _PALETTE[yi % len(_PALETTE)])
        ypts = det["ypts_filt"]
        if len(ypts) == 0:
            continue

        in_r = (
            (ypts[:, 0] >= BEV_X_RANGE[0]) & (ypts[:, 0] <= BEV_X_RANGE[1]) &
            (ypts[:, 1] >= BEV_Y_RANGE[0]) & (ypts[:, 1] <= BEV_Y_RANGE[1])
        )
        yf = ypts[in_r]
        if len(yf) == 0:
            continue

        ax.scatter(yf[:, 1], yf[:, 0], s=5, color=dc, zorder=3, alpha=0.9)

        cx       = float(np.median(yf[:, 0]))
        cy       = float(np.median(yf[:, 1]))
        dist_str = (f"D{det['dist_est']:.1f}m"
                    if det["dist_est"] is not None else "?m")
        ax.text(cy, cx, dist_str, color="white", fontsize=7, zorder=6,
                bbox=dict(facecolor="black", alpha=0.5, pad=1,
                          edgecolor="none"))
        legend_patches.append(
            mpatches.Patch(color=dc, label=f"Det {yi}  {dist_str}")
        )

    # Layer 4 — ego vehicle
    ax.plot(0, 0, marker="^", color="cyan", markersize=10,
            zorder=10, linestyle="none")
    legend_patches.append(mpatches.Patch(color="cyan", label="Ego vehicle"))

    # Axes
    ax.set_xlim(BEV_Y_RANGE[0], BEV_Y_RANGE[1])
    ax.set_ylim(BEV_X_RANGE[0], BEV_X_RANGE[1])
    ax.set_xlabel("Lateral  [m]  (left ←  → right)", color="white", fontsize=9)
    ax.set_ylabel("Forward distance  [m]",            color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.020, pad=0.02)
    cbar.set_label("Height Z [m]", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=7, framealpha=0.75,
              facecolor="#111111", labelcolor="white", edgecolor="none")

    ax.set_title(
        f"Bird's Eye View — Scene {frame_id}\n"
        f"Green boxes = GT ({len(car_labels)})  |  "
        f"Coloured dots = detections ({len(detections)})",
        color="white", fontsize=10, pad=8
    )

    out_path = os.path.join(out_dir, f"bev_{frame_id}.png")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out_path, detections


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_bev_batch(out_dir, model):
    image_paths = sorted(glob.glob(
        os.path.join(KITTI_ROOT,
                     "data_object_image_2", "training", "image_2", "*.png")))
    if not image_paths:
        raise FileNotFoundError(
            f"No images under {KITTI_ROOT}/data_object_image_2/training/image_2/")

    print(f"Found {len(image_paths)} frames.\n")

    csv_path   = os.path.join(out_dir, "bev_summary.csv")
    csv_fields = ["frame_id", "yolo_idx", "gt_type",
                  "dist_estimated_m", "dist_gt_m", "dist_error_m", "iou_2d"]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for idx, img_path in enumerate(image_paths):
            frame_id = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[{idx+1:>3}/{len(image_paths)}] {frame_id} ...",
                  end=" ", flush=True)
            try:
                calib, labels, pts, img_p = load_frame(frame_id)
                yolo_masks, yolo_boxes, (H, W) = \
                    get_yolo_car_detections(img_p, model)

                out_path, detections = plot_bev_frame(
                    frame_id, calib, labels, pts,
                    yolo_masks, yolo_boxes,
                    img_shape=(H, W),
                    out_dir=out_dir
                )
                n_gt = len([l for l in labels if l["type"] in ("Car","Van")])
                print(f"GT={n_gt} | det={len(yolo_masks)} → saved")

                for det in detections:
                    writer.writerow({
                        "frame_id":         frame_id,
                        "yolo_idx":         det["yolo_idx"],
                        "gt_type":          (det["gt_label"]["type"]
                                             if det["gt_label"] else ""),
                        "dist_estimated_m": (f"{det['dist_est']:.3f}"
                                             if det["dist_est"] else ""),
                        "dist_gt_m":        (f"{det['dist_gt']:.3f}"
                                             if det["dist_gt"] else ""),
                        "dist_error_m":     (f"{det['dist_err']:.3f}"
                                             if det["dist_err"] else ""),
                        "iou_2d":           f"{det['iou_2d']:.4f}",
                    })

            except Exception as e:
                print(f"ERROR — {e}")
                traceback.print_exc()

    print(f"\nDone. CSV → {csv_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")
    run_bev_batch(OUT_DIR, model)
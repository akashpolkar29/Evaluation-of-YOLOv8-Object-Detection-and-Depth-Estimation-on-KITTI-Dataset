import os
import glob
import csv
import traceback

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

from kitti_utils import (load_calib, load_labels,
                         project_lidar_to_image, filter_points_in_image)

# ─── Paths ────────────────────────────────────────────────────────────────────
KITTI_ROOT = "../../KITTI-Selection_incl_LiDAR_2026"
IMAGE_DIR  = os.path.join(KITTI_ROOT,
                          "data_object_image_2", "training", "image_2")
OUT_DIR    = ("/home/akash-polkar/Desktop/LRS_Project/"
              "Task 2/Results/Fusion_3D")

os.makedirs(OUT_DIR, exist_ok=True)

CAR_CLASS_ID = 2

_PALETTE = [
    (30,  144, 255),
    (255, 140,   0),
    (180,   0, 200),
    (0,   210, 180),
    (255, 215,   0),
    (255,  80, 160),
    (80,  200,  80),
    (255,  60,  60),
    (0,   160, 200),
    (200, 120,  40),
    (140, 100, 255),
    (60,  220, 180),
]


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_frame(frame_id):
    calib  = load_calib(
        f"{KITTI_ROOT}/data_object_calib/training/calib/{frame_id}.txt")
    labels = load_labels(
        f"{KITTI_ROOT}/data_object_label_2/training/label_2/{frame_id}.txt")
    pts    = np.fromfile(
        f"{KITTI_ROOT}/data_object_velodyne/training/velodyne/{frame_id}.bin",
        dtype=np.float32).reshape(-1, 4)
    img    = cv2.cvtColor(
        cv2.imread(
            f"{KITTI_ROOT}/data_object_image_2/training/image_2/{frame_id}.png"),
        cv2.COLOR_BGR2RGB)
    return calib, labels, pts, img


# ─── Geometry ─────────────────────────────────────────────────────────────────

def points_in_3d_box(pts_lidar, label, calib):
    h, w, l  = label["dims"]
    loc      = label["loc"]
    rot_y    = label["rot_y"]

    N       = pts_lidar.shape[0]
    pts_h   = np.hstack([pts_lidar[:, :3], np.ones((N, 1))])
    pts_cam = (calib["Tr_velo_to_cam"] @ pts_h.T)
    pts_cam = (calib["R0_rect"] @ pts_cam)[:3, :].T

    center       = loc - np.array([0, h / 2, 0])
    pts_centered = pts_cam - center

    cos_r, sin_r = np.cos(-rot_y), np.sin(-rot_y)
    Ry_inv = np.array([[cos_r, 0, sin_r],
                       [0,     1, 0    ],
                       [-sin_r,0, cos_r]])
    pts_obj = (Ry_inv @ pts_centered.T).T

    return (
        (pts_obj[:, 0] >= -l / 2) & (pts_obj[:, 0] <= l / 2) &
        (pts_obj[:, 1] >= -h / 2) & (pts_obj[:, 1] <= h / 2) &
        (pts_obj[:, 2] >= -w / 2) & (pts_obj[:, 2] <= w / 2)
    )


def box_iou_2d(box_a, box_b):
    xi1 = max(box_a[0], box_b[0]); yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2]); yi2 = min(box_a[3], box_b[3])
    inter  = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def project_3d_box_to_image(label, calib):
    h, w, l = label["dims"]
    loc     = label["loc"]
    rot_y   = label["rot_y"]

    corners_obj = np.array([
        [ l/2,    0, -w/2],
        [ l/2,    0,  w/2],
        [-l/2,    0,  w/2],
        [-l/2,    0, -w/2],
        [ l/2,   -h, -w/2],
        [ l/2,   -h,  w/2],
        [-l/2,   -h,  w/2],
        [-l/2,   -h, -w/2],
    ])

    cos_r, sin_r = np.cos(rot_y), np.sin(rot_y)
    Ry = np.array([[cos_r, 0, sin_r],
                   [0,     1, 0    ],
                   [-sin_r,0, cos_r]])
    corners_cam = (Ry @ corners_obj.T).T + loc

    if np.any(corners_cam[:, 2] <= 0):
        return None

    corners_h = np.hstack([corners_cam, np.ones((8, 1))])
    P2        = calib["P2"]
    uvw       = (P2 @ corners_h.T).T
    uv        = uvw[:, :2] / uvw[:, 2:3]
    return uv


def _clip_line_to_image(x0, y0, x1, y1, img_w, img_h):
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < 0:        c |= LEFT
        elif x > img_w:  c |= RIGHT
        if y < 0:        c |= TOP
        elif y > img_h:  c |= BOTTOM
        return c

    c0, c1 = code(x0, y0), code(x1, y1)

    while True:
        if not (c0 | c1):
            return x0, y0, x1, y1
        if c0 & c1:
            return None
        c_out = c0 if c0 else c1
        if c_out & BOTTOM:
            x = x0 + (x1-x0) * (img_h - y0) / (y1 - y0 + 1e-9)
            y = float(img_h)
        elif c_out & TOP:
            x = x0 + (x1-x0) * (0 - y0) / (y1 - y0 + 1e-9)
            y = 0.0
        elif c_out & RIGHT:
            y = y0 + (y1-y0) * (img_w - x0) / (x1 - x0 + 1e-9)
            x = float(img_w)
        else:
            y = y0 + (y1-y0) * (0 - x0) / (x1 - x0 + 1e-9)
            x = 0.0
        if c_out == c0:
            x0, y0, c0 = x, y, code(x, y)
        else:
            x1, y1, c1 = x, y, code(x, y)


def draw_3d_box(ax, corners_uv, color_f, img_w, img_h, lw=1.5):
    if corners_uv is None:
        return
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        x0, y0 = float(corners_uv[i, 0]), float(corners_uv[i, 1])
        x1, y1 = float(corners_uv[j, 0]), float(corners_uv[j, 1])
        clipped = _clip_line_to_image(x0, y0, x1, y1, img_w, img_h)
        if clipped is not None:
            cx0, cy0, cx1, cy1 = clipped
            ax.plot([cx0, cx1], [cy0, cy1],
                    color=color_f, linewidth=lw, alpha=0.9, zorder=4)


# ─── YOLO ─────────────────────────────────────────────────────────────────────

def get_yolo_car_masks(img_path, img_shape, model):
    H, W    = img_shape[:2]
    results = model(img_path, verbose=False)[0]
    masks, boxes, scores = [], [], []
    if results.masks is not None:
        for i, cls_id in enumerate(results.boxes.cls.cpu().numpy()):
            if int(cls_id) != CAR_CLASS_ID:
                continue
            mask = results.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append((mask > 0.5).astype(np.uint8))
            boxes.append(results.boxes.xyxy[i].cpu().numpy())
            scores.append(float(results.boxes.conf[i].cpu()))
    return masks, boxes, scores


# ─── Distance calculation ─────────────────────────────────────────────────────

def estimate_distance(pts_lidar_inside):
    if len(pts_lidar_inside) == 0:
        return None
    dists = np.linalg.norm(pts_lidar_inside[:, :3], axis=1)
    return float(np.min(dists))   # closest point = nearest surface of car


def gt_distance(label):
    return float(label["loc"][2])   # z in camera frame = depth = distance


# ─── Fusion + evaluation ──────────────────────────────────────────────────────
# CHANGE: now returns yolo_boxes + dist_estimated + dist_gt added to each result

def fuse_and_evaluate(frame_id, model, verbose=False):
    img_path = (f"{KITTI_ROOT}/data_object_image_2/"
                f"training/image_2/{frame_id}.png")
    calib, labels, pts, img = load_frame(frame_id)

    u, v, front_mask = project_lidar_to_image(pts, calib)
    valid     = filter_points_in_image(u, v, front_mask, img.shape)
    u_v       = u[valid].astype(int)
    v_v       = v[valid].astype(int)
    pts_valid = pts[valid]

    yolo_masks, yolo_boxes, yolo_scores = \
        get_yolo_car_masks(img_path, img.shape, model)

    if verbose:
        print(f"  Lidar pts in image : {valid.sum()}")
        print(f"  YOLO cars detected : {len(yolo_masks)}")
        print(f"  GT cars            : {len(labels)}")

    yolo_lidar_pts = []
    for mask in yolo_masks:
        in_mask = mask[v_v, u_v] == 1
        yolo_lidar_pts.append(pts_valid[in_mask])

    results = []
    for yi, (ybox, ypts) in enumerate(zip(yolo_boxes, yolo_lidar_pts)):
        best_iou, best_label = 0, None
        for label in labels:
            iou = box_iou_2d(ybox, label["bbox2d"])
            if iou > best_iou:
                best_iou, best_label = iou, label

        n_masked = len(ypts)

        if best_label is None or n_masked == 0:
            results.append({
                "yolo_idx": yi, "matched_gt": False,
                "n_masked_pts": n_masked, "n_inside_3d": 0,
                "precision_3d": 0.0, "iou_2d": 0.0,
                "dist_estimated": None, "dist_gt": None,
                "dist_error": None
            })
            continue

        inside_3d  = points_in_3d_box(ypts, best_label, calib)
        n_inside   = inside_3d.sum()
        precision  = n_inside / n_masked if n_masked > 0 else 0.0

        # ── Distance: estimated from inside-box lidar points vs GT ────────────
        pts_inside    = ypts[inside_3d]
        dist_est      = estimate_distance(pts_inside)   # from sub-pointcloud
        dist_gt_val   = gt_distance(best_label)         # from label loc[2]
        dist_err      = (abs(dist_est - dist_gt_val)
                         if dist_est is not None else None)

        results.append({
            "yolo_idx":      yi,
            "matched_gt":    True,
            "n_masked_pts":  n_masked,
            "n_inside_3d":   int(n_inside),
            "precision_3d":  float(precision),
            "iou_2d":        float(best_iou),
            "gt_label":      best_label,
            "dist_estimated": float(dist_est) if dist_est is not None else None,
            "dist_gt":        float(dist_gt_val),
            "dist_error":     float(dist_err) if dist_err is not None else None,
        })

    # CHANGE: yolo_boxes now returned so visualizer can draw them
    return results, yolo_masks, yolo_boxes, yolo_lidar_pts, \
           calib, pts_valid, u_v, v_v, img


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_fusion(frame_id, results, yolo_masks, yolo_boxes,
                     yolo_lidar_pts, calib, pts_valid, u_v, v_v, img, out_dir):

    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(18, 6), dpi=130)
    ax.imshow(img)

    legend_patches = []
    legend_patches = []
    used_label_y   = []

    for yi, (mask, ybox, ypts) in enumerate(
            zip(yolo_masks, yolo_boxes, yolo_lidar_pts)):

        res  = results[yi]
        mc   = _PALETTE[yi % len(_PALETTE)]
        mc_f = tuple(v / 255.0 for v in mc)
        prec = res["precision_3d"]

        # ── 1. Semi-transparent mask fill ────────────────────────────────────
        mask_rgba = np.zeros((H, W, 4), dtype=np.float32)
        mask_rgba[mask == 1] = [mc_f[0], mc_f[1], mc_f[2], 0.28]
        ax.imshow(mask_rgba, interpolation="nearest", zorder=2)

        # ── 2. Mask contour ───────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim == 2 and len(cnt) > 1:
                ax.plot(cnt[:, 0], cnt[:, 1],
                        color=mc_f, linewidth=1.2, alpha=0.85, zorder=3)

        # ── 3. YOLO 2D bounding box — solid colored rectangle ────────────────
        # This is the actual detector output box
        bx1, by1, bx2, by2 = float(ybox[0]), float(ybox[1]), \
                               float(ybox[2]), float(ybox[3])
        yolo_rect = mpatches.FancyBboxPatch(
            (bx1, by1), bx2 - bx1, by2 - by1,
            boxstyle="square,pad=0",
            linewidth=1.0,
            edgecolor="black",     # ← black, always visible
            facecolor="none",
            linestyle="--",
            zorder=6,
            alpha=1.0
        )
        ax.add_patch(yolo_rect)

        # ── 4. GT 2D bounding box — white dashed rectangle ───────────────────
        if res["matched_gt"]:
            gt2d = res["gt_label"]["bbox2d"]
            gx1, gy1, gx2, gy2 = float(gt2d[0]), float(gt2d[1]), \
                                   float(gt2d[2]), float(gt2d[3])
            gt_rect = mpatches.FancyBboxPatch(
                (gx1, gy1), gx2 - gx1, gy2 - gy1,
                boxstyle="square,pad=0",
                linewidth=1.2,
                edgecolor="white",
                facecolor="none",
                linestyle=":",    # dashed = ground truth
                zorder=4,
                alpha=0.80
            )
            ax.add_patch(gt_rect)

        # ── 5. Lidar points — green=inside GT 3D box / red=outside ───────────
        if res["matched_gt"] and len(ypts) > 0:
            inside_3d = points_in_3d_box(ypts, res["gt_label"], calib)
            u_y, v_y, fm = project_lidar_to_image(ypts, calib)
            valid_y  = filter_points_in_image(u_y, v_y, fm, img.shape)
            u_y      = u_y[valid_y].astype(int)
            v_y      = v_y[valid_y].astype(int)
            inside_y = inside_3d[valid_y]

            if inside_y.sum() > 0:
                ax.scatter(u_y[inside_y], v_y[inside_y],
                           s=3, c="#00e676", linewidths=0,
                           alpha=0.85, zorder=5)
            if (~inside_y).sum() > 0:
                ax.scatter(u_y[~inside_y], v_y[~inside_y],
                           s=3, c="#ff1744", linewidths=0,
                           alpha=0.65, zorder=5)

            # ── 6. GT 3D box wireframe ────────────────────────────────────────
            corners_uv = project_3d_box_to_image(res["gt_label"], calib)
            draw_3d_box(ax, corners_uv, mc_f, img_w=W, img_h=H, lw=1.8)
# REPLACE WITH:
        # ── 7. Label — stepped above YOLO box, collision-aware ───────────────
        yolo_top_y = float(ybox[1])
        yolo_mid_x = float((ybox[0] + ybox[2]) / 2.0)

        if res["dist_estimated"] is not None and res["dist_gt"] is not None:
            dist_txt = (f"d={res['dist_estimated']:.1f}m "
                        f"GT={res['dist_gt']:.1f}m "
                        f"err={res['dist_error']:.1f}m")
        else:
            dist_txt = "d=n/a"

        label_txt = (f"Det {yi} | P={prec:.2f} | "
                     f"pts={res['n_masked_pts']} | {dist_txt}")
        # Max 6 steps upward (= 6 * 20px = 120px above box top at most).
        STEP   = 20    # px per step
        MARGIN = 16    # minimum distance to consider "clear"
        
        candidate_y = max(yolo_top_y - 28, 10)   # start 28px above box

        for _ in range(6):
            conflict = any(
                abs(candidate_y - sy) < MARGIN and
                abs(yolo_mid_x - sx)  < 160        # only conflict if also nearby in x
                for sx, sy in used_label_y
            )
            if not conflict:
                break
            candidate_y = max(candidate_y - STEP, 6)

        used_label_y.append((yolo_mid_x, candidate_y))

        ax.annotate(
            label_txt,
            xy=(yolo_mid_x, yolo_top_y),        # arrow tip: YOLO box top
            xytext=(yolo_mid_x, candidate_y),    # label: stepped above
            fontsize=6, fontweight="bold",
            color="white", ha="center", va="bottom",
            zorder=9,
            arrowprops=dict(
                arrowstyle="-",
                color=mc_f,
                lw=0.8,
                alpha=0.85
            ),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=mc_f,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.93
            )
        )

        legend_patches.append(
            mpatches.Patch(facecolor=mc_f, alpha=0.8,
                           label=f"Det {yi}  P={prec:.2f}"))

    # ── 8. Legend ─────────────────────────────────────────────────────────────
    pt_patches = [
        mpatches.Patch(color="#00e676", label="inside GT 3D box"),
        mpatches.Patch(color="#ff1744", label="outside GT 3D box"),
        mpatches.Patch(facecolor="none", edgecolor="black",
                       linestyle="--", linewidth=1.0,
                       label="YOLO 2D box (detector)"),
        mpatches.Patch(facecolor="none", edgecolor="white",
                       linestyle=":", linewidth=1.2,
                       label="GT 2D box (ground truth)"),
    ]
    ax.legend(
        handles=pt_patches + legend_patches,
        loc="upper right", fontsize=7,
        framealpha=0.80,
        facecolor="#111111",
        labelcolor="white",
        edgecolor="none",
        ncol=1
    )

    # ── 9. Title stats bar ────────────────────────────────────────────────────
    matched  = [r for r in results if r["matched_gt"]]
    mean_p   = np.mean([r["precision_3d"] for r in matched]) if matched else 0.0
    total_in = sum(r["n_inside_3d"]  for r in results)
    total_pt = sum(r["n_masked_pts"] for r in results)
    # Mean distance error across detections that have it
    dist_errs = [r["dist_error"] for r in matched
                 if r.get("dist_error") is not None]
    mean_de  = np.mean(dist_errs) if dist_errs else float("nan")

    ax.set_title(
        f"Frame {frame_id}   |   detections: {len(yolo_masks)}   "
        f"matched: {len(matched)}   "
        f"mean precision: {mean_p:.2f}   "
        f"pts in GT: {total_in}/{total_pt}   "
        f"mean dist error: {mean_de:.2f}m",
        fontsize=9, pad=6, color="black"
    )
    ax.axis("off")

    out_path = os.path.join(out_dir, f"fusion_{frame_id}.png")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(kitti_root, out_dir, model):
    image_paths = sorted(glob.glob(
        os.path.join(kitti_root,
                     "data_object_image_2", "training", "image_2", "*.png")))

    print(f"Found {len(image_paths)} frames.\n")
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    csv_path   = os.path.join(out_dir, "fusion_summary.csv")

    # CHANGE: added dist_estimated, dist_gt, dist_error columns
    csv_fields = ["frame_id", "yolo_idx", "matched_gt",
                  "n_masked_pts", "n_inside_3d", "precision_3d", "iou_2d",
                  "dist_estimated_m", "dist_gt_m", "dist_error_m"]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for idx, img_path in enumerate(image_paths):
            frame_id = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[{idx+1:>3}/{len(image_paths)}] {frame_id} ...", end=" ")

            try:
                # CHANGE: unpack yolo_boxes from fuse_and_evaluate
                res, ymasks, yboxes, ypts, calib, pts_v, u_v, v_v, img = \
                    fuse_and_evaluate(frame_id, model, verbose=False)

                # CHANGE: pass yboxes to visualize_fusion
                out_path = visualize_fusion(
                    frame_id, res, ymasks, yboxes, ypts,
                    calib, pts_v, u_v, v_v, img, out_dir)

                matched = [r for r in res if r["matched_gt"]]
                mean_p  = (np.mean([r["precision_3d"] for r in matched])
                           if matched else float("nan"))
                dist_errs = [r["dist_error"] for r in matched
                             if r.get("dist_error") is not None]
                mean_de = np.mean(dist_errs) if dist_errs else float("nan")

                print(f"{len(ymasks)} cars | "
                      f"prec={mean_p:.3f} | "
                      f"dist_err={mean_de:.2f}m | "
                      f"→ {os.path.basename(out_path)}")

                for r in res:
                    writer.writerow({
                        "frame_id":       frame_id,
                        "yolo_idx":       r["yolo_idx"],
                        "matched_gt":     r["matched_gt"],
                        "n_masked_pts":   r["n_masked_pts"],
                        "n_inside_3d":    r["n_inside_3d"],
                        "precision_3d":   f"{r['precision_3d']:.4f}",
                        "iou_2d":         f"{r['iou_2d']:.4f}",
                        "dist_estimated_m": (f"{r['dist_estimated']:.3f}"
                                             if r["dist_estimated"] is not None
                                             else ""),
                        "dist_gt_m":      (f"{r['dist_gt']:.3f}"
                                           if r.get("dist_gt") is not None
                                           else ""),
                        "dist_error_m":   (f"{r['dist_error']:.3f}"
                                           if r.get("dist_error") is not None
                                           else ""),
                    })

            except Exception as e:
                print(f"ERROR — {e}")
                traceback.print_exc()

    print(f"\nBatch complete.")
    print(f"Visualizations → {out_dir}/fusion_<frame_id>.png")
    print(f"Summary        → {csv_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")
    run_batch(KITTI_ROOT, OUT_DIR, model)
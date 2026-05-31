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
OUT_DIR    = ("/home/akash-polkar/Desktop/LRS_Project/"
              "Task 2/Results/Fusion_Improved")

os.makedirs(OUT_DIR, exist_ok=True)

CAR_CLASS_ID = 2

GROUND_Z_THRESHOLD = -1.4

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
        [ l/2,    0, -w/2], [ l/2,    0,  w/2],
        [-l/2,    0,  w/2], [-l/2,    0, -w/2],
        [ l/2,   -h, -w/2], [ l/2,   -h,  w/2],
        [-l/2,   -h,  w/2], [-l/2,   -h, -w/2],
    ])
    cos_r, sin_r = np.cos(rot_y), np.sin(rot_y)
    Ry = np.array([[cos_r, 0, sin_r],
                   [0,     1, 0    ],
                   [-sin_r,0, cos_r]])
    corners_cam = (Ry @ corners_obj.T).T + loc

    if np.any(corners_cam[:, 2] <= 0):
        return None

    corners_h = np.hstack([corners_cam, np.ones((8, 1))])
    uvw       = (calib["P2"] @ corners_h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def _clip_line_to_image(x0, y0, x1, y1, img_w, img_h):
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x, y):
        c = INSIDE
        if x < 0:       c |= LEFT
        elif x > img_w: c |= RIGHT
        if y < 0:       c |= TOP
        elif y > img_h: c |= BOTTOM
        return c

    c0, c1 = code(x0, y0), code(x1, y1)
    while True:
        if not (c0 | c1): return x0, y0, x1, y1
        if c0 & c1:       return None
        c_out = c0 if c0 else c1
        if c_out & BOTTOM:
            x = x0 + (x1-x0)*(img_h-y0)/(y1-y0+1e-9); y = float(img_h)
        elif c_out & TOP:
            x = x0 + (x1-x0)*(0-y0)/(y1-y0+1e-9);     y = 0.0
        elif c_out & RIGHT:
            y = y0 + (y1-y0)*(img_w-x0)/(x1-x0+1e-9); x = float(img_w)
        else:
            y = y0 + (y1-y0)*(0-x0)/(x1-x0+1e-9);     x = 0.0
        if c_out == c0: x0, y0, c0 = x, y, code(x, y)
        else:           x1, y1, c1 = x, y, code(x, y)


def draw_3d_box(ax, corners_uv, color_f, img_w, img_h, lw=1.5):
    if corners_uv is None:
        return
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        clipped = _clip_line_to_image(
            float(corners_uv[i,0]), float(corners_uv[i,1]),
            float(corners_uv[j,0]), float(corners_uv[j,1]),
            img_w, img_h)
        if clipped:
            ax.plot([clipped[0], clipped[2]], [clipped[1], clipped[3]],
                    color=color_f, linewidth=lw, alpha=0.9, zorder=4)


# ─── Ground filter — the core improvement ─────────────────────────────────────

def filter_ground_points(pts_lidar, z_thresh=GROUND_Z_THRESHOLD):
    return pts_lidar[pts_lidar[:, 2] > z_thresh]


def estimate_distance(pts_lidar_inside):
    if len(pts_lidar_inside) == 0:
        return None
    dists = np.linalg.norm(pts_lidar_inside[:, :3], axis=1)
    return float(np.min(dists))


def gt_distance(label):
    return float(label["loc"][2])


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


# ─── Fusion + evaluation with before/after filter ─────────────────────────────

def fuse_and_evaluate_improved(frame_id, model, verbose=False):
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
                "n_masked_raw": n_masked, "n_inside_raw": 0,
                "precision_raw": 0.0,
                "n_masked_filt": 0, "n_inside_filt": 0,
                "precision_filt": 0.0,
                "precision_improvement": 0.0,
                "iou_2d": 0.0,
                "dist_estimated": None, "dist_gt": None, "dist_error": None
            })
            continue

        # ── BEFORE filter ─────────────────────────────────────────────────────
        inside_raw  = points_in_3d_box(ypts, best_label, calib)
        n_in_raw    = int(inside_raw.sum())
        prec_raw    = n_in_raw / n_masked if n_masked > 0 else 0.0

        # ── AFTER ground filter ───────────────────────────────────────────────
        ypts_filt   = filter_ground_points(ypts)
        n_filt      = len(ypts_filt)

        if n_filt > 0:
            inside_filt = points_in_3d_box(ypts_filt, best_label, calib)
            n_in_filt   = int(inside_filt.sum())
            prec_filt   = n_in_filt / n_filt
        else:
            inside_filt = np.array([], dtype=bool)
            n_in_filt   = 0
            prec_filt   = 0.0

        improvement = prec_filt - prec_raw

        # ── Distance from filtered inside points ──────────────────────────────
        pts_inside_filt = ypts_filt[inside_filt] if n_filt > 0 else np.empty((0,4))
        dist_est        = estimate_distance(pts_inside_filt)
        dist_gt_val     = gt_distance(best_label)
        dist_err        = (abs(dist_est - dist_gt_val)
                           if dist_est is not None else None)

        if verbose:
            print(f"  Det [{yi}]: "
                  f"P_raw={prec_raw:.3f} → P_filt={prec_filt:.3f} "
                  f"(+{improvement:.3f}) | "
                  f"pts: {n_masked}→{n_filt} after filter")

        results.append({
            "yolo_idx":          yi,
            "matched_gt":        True,
            "n_masked_raw":      n_masked,
            "n_inside_raw":      n_in_raw,
            "precision_raw":     float(prec_raw),
            "n_masked_filt":     n_filt,
            "n_inside_filt":     n_in_filt,
            "precision_filt":    float(prec_filt),
            "precision_improvement": float(improvement),
            "iou_2d":            float(best_iou),
            "gt_label":          best_label,
            "dist_estimated":    float(dist_est) if dist_est else None,
            "dist_gt":           float(dist_gt_val),
            "dist_error":        float(dist_err) if dist_err else None,
            # store for visualization
            "_ypts_filt":        ypts_filt,
            "_inside_filt":      inside_filt,
        })

    return results, yolo_masks, yolo_boxes, yolo_lidar_pts, \
           calib, pts_valid, u_v, v_v, img


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_improved(frame_id, results, yolo_masks, yolo_boxes,
                       yolo_lidar_pts, calib, pts_valid,
                       u_v, v_v, img, out_dir):

    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(18, 6), dpi=130)
    ax.imshow(img)

    legend_patches = []
    used_label_y   = []

    for yi, (mask, ybox, ypts) in enumerate(
            zip(yolo_masks, yolo_boxes, yolo_lidar_pts)):

        res  = results[yi]
        mc   = _PALETTE[yi % len(_PALETTE)]
        mc_f = tuple(v / 255.0 for v in mc)

        p_raw  = res["precision_raw"]
        p_filt = res["precision_filt"]

        # ── 1. Mask fill ──────────────────────────────────────────────────────
        mask_rgba = np.zeros((H, W, 4), dtype=np.float32)
        mask_rgba[mask == 1] = [mc_f[0], mc_f[1], mc_f[2], 0.25]
        ax.imshow(mask_rgba, interpolation="nearest", zorder=2)

        # ── 2. Mask contour ───────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim == 2 and len(cnt) > 1:
                ax.plot(cnt[:, 0], cnt[:, 1],
                        color=mc_f, linewidth=1.2, alpha=0.85, zorder=3)

        # ── 3. YOLO 2D box — black dashed ─────────────────────────────────────
        bx1,by1,bx2,by2 = (float(ybox[0]), float(ybox[1]),
                            float(ybox[2]), float(ybox[3]))
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx1, by1), bx2-bx1, by2-by1,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor="black", facecolor="none",
            linestyle="--", zorder=6, alpha=0.55))

        # ── 4. GT 2D box — white dotted ───────────────────────────────────────
        if res["matched_gt"]:
            gt2d = res["gt_label"]["bbox2d"]
            gx1,gy1,gx2,gy2 = (float(gt2d[0]), float(gt2d[1]),
                                float(gt2d[2]), float(gt2d[3]))
            ax.add_patch(mpatches.FancyBboxPatch(
                (gx1, gy1), gx2-gx1, gy2-gy1,
                boxstyle="square,pad=0", linewidth=1.2,
                edgecolor="white", facecolor="none",
                linestyle=":", zorder=4, alpha=0.80))

        # ── 5. Lidar points AFTER filter ──────────────────────────────────────
        # green = inside GT 3D box (filtered), red = outside (filtered)
        if res["matched_gt"] and len(res["_ypts_filt"]) > 0:
            ypts_f    = res["_ypts_filt"]
            inside_f  = res["_inside_filt"]

            u_f, v_f, fm = project_lidar_to_image(ypts_f, calib)
            val_f = filter_points_in_image(u_f, v_f, fm, img.shape)
            u_f   = u_f[val_f].astype(int)
            v_f   = v_f[val_f].astype(int)
            in_f  = inside_f[val_f]

            if in_f.sum() > 0:
                ax.scatter(u_f[in_f],  v_f[in_f],
                           s=3, c="#00e676", linewidths=0,
                           alpha=0.85, zorder=5)
            if (~in_f).sum() > 0:
                ax.scatter(u_f[~in_f], v_f[~in_f],
                           s=3, c="#ff1744", linewidths=0,
                           alpha=0.55, zorder=5)

            # ── 6. GT 3D box wireframe ────────────────────────────────────────
            corners_uv = project_3d_box_to_image(res["gt_label"], calib)
            draw_3d_box(ax, corners_uv, mc_f, W, H, lw=1.8)

        # ── 7. Label: show BOTH precisions + improvement ──────────────────────
        yolo_top_y = float(ybox[1])
        yolo_mid_x = float((ybox[0] + ybox[2]) / 2.0)

        imp   = res["precision_improvement"]
        sign  = "+" if imp >= 0 else ""

        if res["dist_estimated"] is not None:
            dist_txt = f"d={res['dist_estimated']:.1f}m GT={res['dist_gt']:.1f}m"
        else:
            dist_gt_val = res.get('dist_gt') or 0.0
            dist_txt = f"lidar occluded | GT={dist_gt_val:.1f}m"

        label_txt = (f"Det {yi} | "
                     f"P_raw={p_raw:.2f} → P_filt={p_filt:.2f} "
                     f"({sign}{imp:.2f}) | {dist_txt}")

        # Collision-aware placement
        STEP, MARGIN = 20, 16
        candidate_y  = max(yolo_top_y - 28, 10)
        for _ in range(6):
            conflict = any(
                abs(candidate_y - sy) < MARGIN and
                abs(yolo_mid_x - sx)  < 160
                for sx, sy in used_label_y)
            if not conflict:
                break
            candidate_y = max(candidate_y - STEP, 6)
        used_label_y.append((yolo_mid_x, candidate_y))

        # Label color: green if improved, orange if same, red if worse
        if imp > 0.01:
            badge_color = (0.0, 0.6, 0.3)
        elif imp < -0.01:
            badge_color = (0.8, 0.1, 0.1)
        else:
            badge_color = mc_f

        ax.annotate(
            label_txt,
            xy=(yolo_mid_x, yolo_top_y),
            xytext=(yolo_mid_x, candidate_y),
            fontsize=6, fontweight="bold",
            color="white", ha="center", va="bottom",
            zorder=9,
            arrowprops=dict(arrowstyle="-", color=mc_f,
                            lw=0.8, alpha=0.85),
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=badge_color,
                      edgecolor="white", linewidth=0.5, alpha=0.93)
        )

        legend_patches.append(
            mpatches.Patch(facecolor=mc_f, alpha=0.8,
                           label=f"Det {yi}  {p_raw:.2f}→{p_filt:.2f}"))

    # ── 8. Legend ─────────────────────────────────────────────────────────────
    pt_patches = [
        mpatches.Patch(color="#00e676", label="inside GT 3D box (filtered)"),
        mpatches.Patch(color="#ff1744", label="outside GT 3D box (filtered)"),
        mpatches.Patch(facecolor="none", edgecolor="black",
                       linestyle="--", linewidth=1.5,
                       label="YOLO 2D box"),
        mpatches.Patch(facecolor="none", edgecolor="white",
                       linestyle=":", linewidth=1.2,
                       label="GT 2D box"),
        mpatches.Patch(color=(0.0,0.6,0.3), label="label = improved"),
        mpatches.Patch(color=(0.8,0.1,0.1), label="label = worse"),
    ]
    ax.legend(handles=pt_patches + legend_patches,
              loc="upper right", fontsize=7,
              framealpha=0.80, facecolor="#111111",
              labelcolor="white", edgecolor="none", ncol=1)

    # ── 9. Title ──────────────────────────────────────────────────────────────
    matched = [r for r in results if r["matched_gt"]]
    mean_raw  = np.mean([r["precision_raw"]  for r in matched]) if matched else 0
    mean_filt = np.mean([r["precision_filt"] for r in matched]) if matched else 0
    mean_imp  = np.mean([r["precision_improvement"] for r in matched]) if matched else 0

    ax.set_title(
        f"Frame {frame_id}   |   detections: {len(yolo_masks)}   "
        f"mean P_raw={mean_raw:.2f}  →  "
        f"mean P_filt={mean_filt:.2f}  "
        f"(Δ={mean_imp:+.2f})   "
        f"ground filter Z > {GROUND_Z_THRESHOLD}m",
        fontsize=9, pad=6, color="black"
    )
    ax.axis("off")

    out_path = os.path.join(out_dir, f"fusion_improved_{frame_id}.png")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── Dataset-level comparison plot ────────────────────────────────────────────

def plot_comparison(all_results_flat, out_dir):
    """
    Two plots:
    1. Histogram: precision before vs after (overlaid)
    2. Scatter:   per-detection P_raw vs P_filt
    """
    matched = [r for r in all_results_flat if r["matched_gt"]]
    if not matched:
        print("No matched detections — skipping comparison plot.")
        return

    p_raw  = np.array([r["precision_raw"]  for r in matched])
    p_filt = np.array([r["precision_filt"] for r in matched])
    imp    = p_filt - p_raw

    # ── Plot A: overlaid histograms ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)

    bins = np.linspace(0, 1, 21)
    ax.hist(p_raw,  bins=bins, alpha=0.6, color="#ff1744",
            label=f"Before filter  mean={p_raw.mean():.3f}",
            edgecolor="white", linewidth=0.5)
    ax.hist(p_filt, bins=bins, alpha=0.6, color="#00e676",
            label=f"After filter   mean={p_filt.mean():.3f}",
            edgecolor="white", linewidth=0.5)

    ax.axvline(p_raw.mean(),  color="#ff1744", linewidth=2,
               linestyle="--", alpha=0.9)
    ax.axvline(p_filt.mean(), color="#00e676", linewidth=2,
               linestyle="--", alpha=0.9)

    ax.set_xlabel("Precision  (pts inside GT 3D box / pts in mask)",
                  fontsize=10)
    ax.set_ylabel("Number of detections", fontsize=10)
    ax.set_title(
        f"Precision before vs after ground filter  "
        f"(Z > {GROUND_Z_THRESHOLD}m)\n"
        f"Mean improvement: {imp.mean():+.3f}   "
        f"Detections improved: {(imp > 0.01).sum()}/{len(matched)}",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "improvement_histogram.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved: improvement_histogram.png")

    # ── Plot B: per-detection scatter P_raw vs P_filt ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)

    colors = np.where(imp > 0.01, "#00e676",
              np.where(imp < -0.01, "#ff1744", "#aaaaaa"))

    ax.scatter(p_raw, p_filt, c=colors, s=35, alpha=0.85,
               edgecolors="none", zorder=3)
    ax.plot([0, 1], [0, 1], color="white", linewidth=1.2,
            linestyle="--", alpha=0.5, label="no change line")

    ax.set_xlabel("Precision BEFORE filter", fontsize=10)
    ax.set_ylabel("Precision AFTER filter",  fontsize=10)
    ax.set_title(
        "Per-detection precision: before vs after ground filter\n"
        "green = improved   red = worse   gray = unchanged",
        fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#333333", edgecolor="none")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "improvement_scatter.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved: improvement_scatter.png")

    # ── Text summary ──────────────────────────────────────────────────────────
    lines = [
        "=" * 58,
        "  GROUND FILTER IMPROVEMENT SUMMARY",
        f"  Filter threshold: Z > {GROUND_Z_THRESHOLD} m (lidar frame)",
        "=" * 58,
        f"  Matched detections          : {len(matched)}",
        f"  Detections improved (Δ>0.01): {(imp > 0.01).sum()}",
        f"  Detections unchanged        : {((imp >= -0.01) & (imp <= 0.01)).sum()}",
        f"  Detections worse   (Δ<-0.01): {(imp < -0.01).sum()}",
        "",
        "  ── Precision BEFORE filter ──────────────────────",
        f"  Mean   : {p_raw.mean():.3f}",
        f"  Median : {np.median(p_raw):.3f}",
        f"  Std    : {p_raw.std():.3f}",
        "",
        "  ── Precision AFTER filter ───────────────────────",
        f"  Mean   : {p_filt.mean():.3f}",
        f"  Median : {np.median(p_filt):.3f}",
        f"  Std    : {p_filt.std():.3f}",
        "",
        "  ── Improvement ──────────────────────────────────",
        f"  Mean Δ precision : {imp.mean():+.3f}",
        f"  Max  Δ precision : {imp.max():+.3f}",
        f"  Min  Δ precision : {imp.min():+.3f}",
        "=" * 58,
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out_dir, "improvement_stats.txt"), "w") as f:
        f.write(txt + "\n")
    print("Saved: improvement_stats.txt")


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(out_dir, model):
    image_paths = sorted(glob.glob(
        os.path.join(KITTI_ROOT,
                     "data_object_image_2", "training", "image_2", "*.png")))

    print(f"Found {len(image_paths)} frames.\n")

    csv_path   = os.path.join(out_dir, "fusion_improved_summary.csv")
    csv_fields = ["frame_id", "yolo_idx", "matched_gt",
                  "n_masked_raw", "n_inside_raw", "precision_raw",
                  "n_masked_filt", "n_inside_filt", "precision_filt",
                  "precision_improvement", "iou_2d",
                  "dist_estimated_m", "dist_gt_m", "dist_error_m"]

    all_results_flat = []

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for idx, img_path in enumerate(image_paths):
            frame_id = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[{idx+1:>3}/{len(image_paths)}] {frame_id} ...", end=" ")

            try:
                res, ymasks, yboxes, ypts, calib, pts_v, u_v, v_v, img = \
                    fuse_and_evaluate_improved(frame_id, model, verbose=False)

                out_path = visualize_improved(
                    frame_id, res, ymasks, yboxes, ypts,
                    calib, pts_v, u_v, v_v, img, out_dir)

                matched = [r for r in res if r["matched_gt"]]
                mean_raw  = (np.mean([r["precision_raw"]  for r in matched])
                             if matched else float("nan"))
                mean_filt = (np.mean([r["precision_filt"] for r in matched])
                             if matched else float("nan"))
                mean_imp  = (np.mean([r["precision_improvement"] for r in matched])
                             if matched else float("nan"))

                print(f"{len(ymasks)} cars | "
                      f"P_raw={mean_raw:.3f} → P_filt={mean_filt:.3f} "
                      f"(Δ={mean_imp:+.3f})")

                for r in res:
                    all_results_flat.append(r)
                    # strip internal numpy arrays before writing CSV
                    writer.writerow({
                        "frame_id":            frame_id,
                        "yolo_idx":            r["yolo_idx"],
                        "matched_gt":          r["matched_gt"],
                        "n_masked_raw":        r["n_masked_raw"],
                        "n_inside_raw":        r["n_inside_raw"],
                        "precision_raw":       f"{r['precision_raw']:.4f}",
                        "n_masked_filt":       r["n_masked_filt"],
                        "n_inside_filt":       r["n_inside_filt"],
                        "precision_filt":      f"{r['precision_filt']:.4f}",
                        "precision_improvement": f"{r['precision_improvement']:.4f}",
                        "iou_2d":              f"{r['iou_2d']:.4f}",
                        "dist_estimated_m":    (f"{r['dist_estimated']:.3f}"
                                                if r["dist_estimated"] else ""),
                        "dist_gt_m":           (f"{r['dist_gt']:.3f}"
                                                if r.get("dist_gt") else ""),
                        "dist_error_m":        (f"{r['dist_error']:.3f}"
                                                if r.get("dist_error") else ""),
                    })

            except Exception as e:
                print(f"ERROR — {e}")
                traceback.print_exc()

    # Dataset-level comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_results_flat, out_dir)

    print(f"\nBatch complete.")
    print(f"Visualizations → {out_dir}/fusion_improved_<frame_id>.png")
    print(f"CSV            → {csv_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")
    run_batch(OUT_DIR, model)
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = ("/home/akash-polkar/Desktop/LRS_Project/"
            "Task 2/Results/Fusion_3D/fusion_summary.csv")
OUT_DIR  = ("/home/akash-polkar/Desktop/LRS_Project/"
            "Task 2/Results/Summary")

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Keep only matched detections with valid data
matched   = df[df["matched_gt"] == True].copy()
with_dist = matched.dropna(subset=["dist_estimated_m",
                                   "dist_gt_m", "dist_error_m"]).copy()

print(f"Total detections      : {len(df)}")
print(f"Matched to GT         : {len(matched)}")
print(f"With distance estimate: {len(with_dist)}")
print(f"Lidar occluded (n/a)  : {len(matched) - len(with_dist)}")
print(f"Unmatched             : {len(df[df['matched_gt'] == False])}")

# ─── Plot 1 — Precision histogram ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=130)

ax.hist(matched["precision_3d"], bins=20, range=(0, 1),
        color="#1e88e5", edgecolor="white", linewidth=0.6, alpha=0.9)

mean_p  = matched["precision_3d"].mean()
median_p = matched["precision_3d"].median()

ax.axvline(mean_p,   color="#ff1744", linewidth=1.5,
           linestyle="--", label=f"mean = {mean_p:.2f}")
ax.axvline(median_p, color="#00e676", linewidth=1.5,
           linestyle=":",  label=f"median = {median_p:.2f}")

ax.set_xlabel("Precision  (pts inside GT 3D box / pts in YOLO mask)",
              fontsize=10)
ax.set_ylabel("Number of detections", fontsize=10)
ax.set_title("3D precision distribution — all matched detections", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "precision_histogram.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("\nSaved: precision_histogram.png")

# ─── Plot 2 — Distance error bar per frame ────────────────────────────────────
frame_dist = (with_dist.groupby("frame_id")["dist_error_m"]
              .mean().reset_index())
frame_dist = frame_dist.sort_values("dist_error_m")

fig, ax = plt.subplots(figsize=(12, 5), dpi=130)

bars = ax.bar(range(len(frame_dist)),
              frame_dist["dist_error_m"],
              color="#1e88e5", edgecolor="white", linewidth=0.5, alpha=0.9)

# Color bars above 2m error in red
for bar, val in zip(bars, frame_dist["dist_error_m"]):
    if val > 2.0:
        bar.set_color("#ff1744")
        bar.set_alpha(0.85)

ax.set_xticks(range(len(frame_dist)))
ax.set_xticklabels(frame_dist["frame_id"].astype(str),
                   rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Frame ID", fontsize=10)
ax.set_ylabel("Mean distance error (m)", fontsize=10)
ax.set_title("Mean distance error per frame  "
             "(red = error > 2 m)", fontsize=11)
ax.axhline(with_dist["dist_error_m"].mean(), color="#ff9800",
           linewidth=1.5, linestyle="--",
           label=f"dataset mean = {with_dist['dist_error_m'].mean():.2f}m")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "distance_error_per_frame.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved: distance_error_per_frame.png")

# ─── Plot 3 — Estimated vs GT distance scatter ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), dpi=130)

sc = ax.scatter(with_dist["dist_gt_m"], with_dist["dist_estimated_m"],
                c=with_dist["dist_error_m"], cmap="RdYlGn_r",
                s=40, alpha=0.85, edgecolors="none",
                vmin=0, vmax=5)

# Perfect prediction line
max_d = max(with_dist["dist_gt_m"].max(),
            with_dist["dist_estimated_m"].max()) + 2
ax.plot([0, max_d], [0, max_d], color="white",
        linewidth=1.2, linestyle="--", alpha=0.7,
        label="perfect estimate")

cbar = fig.colorbar(sc, ax=ax, label="Distance error (m)")
ax.set_xlabel("GT distance (m)", fontsize=10)
ax.set_ylabel("Estimated distance from lidar (m)", fontsize=10)
ax.set_title("Estimated vs GT distance per detection\n"
             "color = error magnitude", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, max_d)
ax.set_ylim(0, max_d)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "distance_scatter.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved: distance_scatter.png")

# ─── Text summary ─────────────────────────────────────────────────────────────
lines = [
    "=" * 55,
    "  DATASET-LEVEL EVALUATION SUMMARY",
    "=" * 55,
    f"  Total YOLO detections       : {len(df)}",
    f"  Matched to GT (IoU>0)       : {len(matched)}",
    f"  Lidar occluded (no dist)    : {len(matched) - len(with_dist)}",
    f"  Unmatched detections        : {len(df[df['matched_gt'] == False])}",
    "",
    "  ── Precision (pts inside GT 3D box / mask pts) ──",
    f"  Mean precision              : {matched['precision_3d'].mean():.3f}",
    f"  Median precision            : {matched['precision_3d'].median():.3f}",
    f"  Std deviation               : {matched['precision_3d'].std():.3f}",
    f"  Detections with P > 0.5     : "
    f"{(matched['precision_3d'] > 0.5).sum()} / {len(matched)}",
    f"  Detections with P > 0.7     : "
    f"{(matched['precision_3d'] > 0.7).sum()} / {len(matched)}",
    "",
    "  ── Distance estimation ──────────────────────────",
    f"  Detections with distance    : {len(with_dist)}",
    f"  Mean distance error         : {with_dist['dist_error_m'].mean():.3f} m",
    f"  Median distance error       : {with_dist['dist_error_m'].median():.3f} m",
    f"  Std distance error          : {with_dist['dist_error_m'].std():.3f} m",
    f"  Max distance error          : {with_dist['dist_error_m'].max():.3f} m",
    f"  Errors < 1 m                : "
    f"{(with_dist['dist_error_m'] < 1.0).sum()} / {len(with_dist)}",
    f"  Errors < 2 m                : "
    f"{(with_dist['dist_error_m'] < 2.0).sum()} / {len(with_dist)}",
    "=" * 55,
]

summary_txt = "\n".join(lines)
print("\n" + summary_txt)

txt_path = os.path.join(OUT_DIR, "summary_stats.txt")
with open(txt_path, "w") as f:
    f.write(summary_txt + "\n")
print(f"\nSaved: summary_stats.txt")
print(f"All outputs → {OUT_DIR}")
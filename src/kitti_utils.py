import numpy as np


# ─── Calibration ──────────────────────────────────────────────────────────────

def load_calib(calib_path: str) -> dict:
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])

    P2 = data["P2"].reshape(3, 4)

    R0 = np.eye(4)
    R0[:3, :3] = data["R0_rect"].reshape(3, 3)

    Tr = np.eye(4)
    Tr[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)

    return {"P2": P2, "R0_rect": R0, "Tr_velo_to_cam": Tr}


# ─── Projection ───────────────────────────────────────────────────────────────

def project_lidar_to_image(points_xyz: np.ndarray, calib: dict) -> tuple:
    pts   = points_xyz[:, :3]
    N     = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N, 1))])

    pts_cam  = calib["Tr_velo_to_cam"] @ pts_h.T
    pts_rect = calib["R0_rect"] @ pts_cam

    front_mask = pts_rect[2, :] > 0

    pts_img = calib["P2"] @ pts_rect
    pts_img[:2, :] /= pts_img[2, :]

    return pts_img[0, :], pts_img[1, :], front_mask


def filter_points_in_image(u, v, front_mask, img_shape: tuple) -> np.ndarray:
    H, W      = img_shape[:2]
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return front_mask & in_bounds


# ─── Label parsing ────────────────────────────────────────────────────────────

def load_labels(label_path: str, class_filter=None) -> list:
    """
    Parse a KITTI label .txt file.

    CHANGED from original:
      class_filter now defaults to None (returns ALL object types).
      Pass class_filter="Car" to replicate old behaviour.

    Each dict contains:
      type        : str
      truncation  : float
      occlusion   : int
      alpha       : float
      bbox2d      : np.array [x_min, y_min, x_max, y_max]  image pixels
      dims        : np.array [height, width, length]         metres
      loc         : np.array [x, y, z]                      camera frame metres
                    x=right, y=down, z=forward
                    loc[2] = forward distance from camera
      rot_y       : float  yaw around Y_cam axis, radians
    """
    objects = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            if class_filter and obj_type != class_filter:
                continue
            objects.append({
                "type":       obj_type,
                "truncation": float(parts[1]),
                "occlusion":  int(parts[2]),
                "alpha":      float(parts[3]),
                "bbox2d":     np.array([float(x) for x in parts[4:8]]),
                "dims":       np.array([float(x) for x in parts[8:11]]),
                "loc":        np.array([float(x) for x in parts[11:14]]),
                "rot_y":      float(parts[14]),
            })
    return objects
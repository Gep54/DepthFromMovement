from __future__ import annotations

import numpy as np

from pipeline.geometry import invert_se3


def reprojection_errors(
    X_world_h: np.ndarray,
    pts: np.ndarray,
    K: np.ndarray,
    world_T_camera: np.ndarray,
) -> np.ndarray:
    """Per-point RMS pixel error for one camera. X_world_h is (4,N), pts (N,2).

    ``world_T_camera`` is **camera→world** (``motion.json`` convention).
    """
    Tcw = invert_se3(world_T_camera)
    R = Tcw[:3, :3]
    t = Tcw[:3, 3].reshape(3, 1)
    X = X_world_h[:3, :]
    Xc = R @ X + t
    z = Xc[2, :]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * (Xc[0, :] / z) + K[0, 2]
    v = K[1, 1] * (Xc[1, :] / z) + K[1, 2]
    pred = np.stack([u, v], axis=1)
    d = pred - pts
    return np.sqrt(np.sum(d * d, axis=1))


def summarize_reprojection(err1: np.ndarray, err2: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "rmse_cam1": float(np.sqrt(np.mean(err1**2))) if len(err1) else 0.0,
        "mean_cam1": float(np.mean(err1)) if len(err1) else 0.0,
    }
    if err2 is not None and len(err2):
        out["rmse_cam2"] = float(np.sqrt(np.mean(err2**2)))
        out["mean_cam2"] = float(np.mean(err2))
        out["rmse_both"] = float(np.sqrt(np.mean(np.concatenate([err1**2, err2**2]))))
    return out

from __future__ import annotations

import cv2
import numpy as np

from pipeline.geometry import invert_se3


def _projection_k_rt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = K[R|t] with X_cam = R @ X_world + t (column t)."""
    tr = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R.astype(np.float64), tr])


def world_points_projection_matrix(K: np.ndarray, cam_to_world_T: np.ndarray) -> np.ndarray:
    """
    Build P projecting **world** homogeneous points to image pixels.

    ``cam_to_world_T`` maps camera→world: ``X_w = R_wc @ X_c + t_wc``.
    """
    Tcw = invert_se3(cam_to_world_T)
    return _projection_k_rt(K, Tcw[:3, :3], Tcw[:3, 3])


def triangulate_cam1_frame(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R_c1_c2: np.ndarray,
    t_c1_c2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate in camera-1 frame: P1=K[I|0], P2=K[R|t] with X_c2 expressed from X_c1
    as in OpenCV ``recoverPose`` / ``triangulatePoints`` convention.

    Returns homogeneous (4,N) in camera-1 coordinates + cheirality mask (positive Z in cam1).
    """
    P1 = _projection_k_rt(K, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    P2 = _projection_k_rt(K, R_c1_c2, t_c1_c2.ravel())
    pts1_h = pts1.T.astype(np.float64)
    pts2_h = pts2.T.astype(np.float64)
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X_h = X_h / (X_h[3:4, :] + 1e-12)
    Xc = X_h[:3, :]
    mask = Xc[2, :] > 1e-6
    return X_h, mask


def cam1_to_world_points(X_cam1_h: np.ndarray, world_T_c1: np.ndarray) -> np.ndarray:
    """Map homogeneous cam1-frame points to world homogeneous (4,N), cam→world pose."""
    R = world_T_c1[:3, :3]
    t = world_T_c1[:3, 3].reshape(3, 1)
    Xc = X_cam1_h[:3, :]
    Xw = R @ Xc + t
    out = np.ones((4, Xw.shape[1]), dtype=np.float64)
    out[:3, :] = Xw
    return out


def triangulate_world_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    world_T_c1: np.ndarray,
    world_T_c2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate homogeneous world points (4,N) from two views.

    Poses are **camera→world** (``motion.json`` convention).
    Returns ``(Xw_h, depth_positive_mask)`` where mask uses positive depth in cam1.
    """
    P1 = world_points_projection_matrix(K, world_T_c1)
    P2 = world_points_projection_matrix(K, world_T_c2)
    pts1_h = pts1.T.astype(np.float64)
    pts2_h = pts2.T.astype(np.float64)
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X_h = X_h / (X_h[3:4, :] + 1e-12)
    Xw = X_h[:3, :]
    Tcw = invert_se3(world_T_c1)
    R1 = Tcw[:3, :3]
    t1 = Tcw[:3, 3].reshape(3, 1)
    z1 = (R1 @ Xw + t1)[2, :]
    mask = z1 > 1e-6
    return X_h, mask

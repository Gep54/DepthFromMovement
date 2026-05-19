from __future__ import annotations

import cv2
import numpy as np

from pipeline.geometry import invert_se3

# Minimum positive depth (metres) along +Z in each camera frame; points at or behind fail cheirality.
CHEIRAL_MIN_Z = 1e-6  # 0.000001


def cheiral_mask_cam_frames(
    X_cam1: np.ndarray,
    R_c1_c2: np.ndarray,
    t_c1_c2: np.ndarray,
    *,
    min_z: float = CHEIRAL_MIN_Z,
) -> np.ndarray:
    """
    True where the 3D point lies in front of both cameras (positive Z in each frame).

    ``X_cam1`` is (3, N) in camera-1 coordinates. ``R_c1_c2``, ``t_c1_c2`` follow OpenCV
    ``recoverPose`` / ``triangulatePoints``: ``X_c1 = R @ X_c2 + t``, so
    ``X_c2 = R.T @ (X_c1 - t)``.
    """
    Xc = np.asarray(X_cam1, dtype=np.float64)
    if Xc.ndim != 2 or Xc.shape[0] != 3:
        raise ValueError(f"X_cam1 must be (3, N), got {Xc.shape}")
    R = np.asarray(R_c1_c2, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_c1_c2, dtype=np.float64).reshape(3, 1)
    z1 = Xc[2, :]
    X_cam2 = R.T @ (Xc - t)
    z2 = X_cam2[2, :]
    return (z1 > min_z) & (z2 > min_z)


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
    *,
    check_cheiral: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate in camera-1 frame: P1=K[I|0], P2=K[R|t] with X_c2 expressed from X_c1
    as in OpenCV ``recoverPose`` / ``triangulatePoints`` convention.

    Returns homogeneous (4,N) in camera-1 coordinates + cheirality mask (Z > CHEIRAL_MIN_Z in cam1 and cam2).
    """
    P1 = _projection_k_rt(K, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    P2 = _projection_k_rt(K, R_c1_c2, t_c1_c2.ravel())
    pts1_h = pts1.T.astype(np.float64)
    pts2_h = pts2.T.astype(np.float64)
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X_h = X_h / (X_h[3:4, :] + 1e-12)
    Xc = X_h[:3, :]
    n = Xc.shape[1]
    if not check_cheiral or n == 0:
        mask = np.ones(n, dtype=bool) if n else np.zeros(0, dtype=bool)
    else:
        mask = cheiral_mask_cam_frames(Xc, R_c1_c2, t_c1_c2)
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
    *,
    check_cheiral: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate homogeneous world points (4,N) from two views.

    Poses are **camera→world** (``motion.json`` convention).
    Returns ``(Xw_h, cheiral_mask)`` where mask requires Z > CHEIRAL_MIN_Z in both cameras.
    """
    P1 = world_points_projection_matrix(K, world_T_c1)
    P2 = world_points_projection_matrix(K, world_T_c2)
    pts1_h = pts1.T.astype(np.float64)
    pts2_h = pts2.T.astype(np.float64)
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X_h = X_h / (X_h[3:4, :] + 1e-12)
    Xw = X_h[:3, :]
    Tcw1 = invert_se3(world_T_c1)
    Tcw2 = invert_se3(world_T_c2)
    n = Xw.shape[1]
    if not check_cheiral or n == 0:
        mask = np.ones(n, dtype=bool) if n else np.zeros(0, dtype=bool)
    else:
        z1 = (Tcw1[:3, :3] @ Xw + Tcw1[:3, 3].reshape(3, 1))[2, :]
        z2 = (Tcw2[:3, :3] @ Xw + Tcw2[:3, 3].reshape(3, 1))[2, :]
        mask = (z1 > CHEIRAL_MIN_Z) & (z2 > CHEIRAL_MIN_Z)
    return X_h, mask

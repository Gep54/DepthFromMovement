from __future__ import annotations

import cv2
import numpy as np


def relative_motion_from_world_poses(world_T_c1: np.ndarray, world_T_c2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t such that X_c1 = R @ X_c2 + t (OpenCV recoverPose convention)."""
    c1_T_w = _inv_se3(world_T_c1)
    c1_T_c2 = c1_T_w @ world_T_c2
    R = c1_T_c2[:3, :3].astype(np.float64)
    t = c1_T_c2[:3, 3].astype(np.float64).reshape(3, 1)
    return R, t


def essential_from_world_poses(
    world_T_c1: np.ndarray,
    world_T_c2: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    R, t = relative_motion_from_world_poses(world_T_c1, world_T_c2)
    t_skew = _skew_symmetric(t.ravel())
    E = t_skew @ R
    return E


def _skew_symmetric(t: np.ndarray) -> np.ndarray:
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) matrix (e.g. cam→world ↔ world→cam)."""
    return _inv_se3(T)


def estimate_essential_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    prob: float = 0.999,
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns E (3,3) and inlier mask (N,1) uint8."""
    if len(pts1) < 8:
        raise ValueError("need at least 8 points for essential matrix")
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=prob,
        threshold=threshold,
    )
    return E, mask


def recover_pose_from_essential(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns R, t (unit translation), inlier mask from recoverPose."""
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, pose_mask


def scale_from_odometry(t_vis: np.ndarray, t_odom: np.ndarray, eps: float = 1e-9) -> tuple[float, bool]:
    """
    s = ||t_odom|| / ||t_vis|| with t_vis from recoverPose (unit-norm up to sign).
    Returns (scale, ok). ok is False if ||t_vis|| is tiny (degenerate translation).
    """
    nv = float(np.linalg.norm(t_vis))
    no = float(np.linalg.norm(t_odom))
    if nv < eps:
        return 1.0, False
    dot = float(np.dot(t_vis.ravel(), t_odom.ravel()))
    sign = -1.0 if dot < 0 else 1.0
    s = sign * (no / nv)
    return float(s), True


def align_translation_direction(t_vis: np.ndarray, t_odom: np.ndarray) -> np.ndarray:
    """Flip t_vis if it points opposite to t_odom (same line, ambiguous sign)."""
    tv = t_vis.reshape(3, 1)
    to = t_odom.reshape(3, 1)
    if np.dot(tv.ravel(), to.ravel()) < 0:
        return -tv
    return tv

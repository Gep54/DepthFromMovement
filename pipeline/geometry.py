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


def canonicalize_world_T_camera_to_first(world_T_camera: list[np.ndarray]) -> list[np.ndarray]:
    """Deprecated: left-multiply by ``inv(W[0])`` (camera-0-centric world). Not used by ``load_dataset``."""
    if not world_T_camera:
        raise ValueError("world_T_camera must be non-empty")
    W0 = np.asarray(world_T_camera[0], dtype=np.float64)
    L = _inv_se3(W0)
    return [L @ np.asarray(Wk, dtype=np.float64) for Wk in world_T_camera]


def estimate_essential_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    prob: float = 0.999,
    threshold: float = 3.0,
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


def essential_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Essential matrix E = [t]_× R for relative pose (t may be any non-zero scale)."""
    tv = np.asarray(t, dtype=np.float64).ravel()
    t_skew = _skew_symmetric(tv)
    return t_skew @ np.asarray(R, dtype=np.float64)


def fundamental_from_essential(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Fundamental matrix ``F = K^{-T} E K^{-1}`` for calibrated pinhole cameras."""
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    return Kinv.T @ np.asarray(E, dtype=np.float64) @ Kinv


def symmetric_epipolar_distances(
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Per-match symmetric point-to-epiline distance in pixels (mean of both images)."""
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if p1.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    d2 = np.abs(lines2[:, 0] * p2[:, 0] + lines2[:, 1] * p2[:, 1] + lines2[:, 2])
    n2 = np.hypot(lines2[:, 0], lines2[:, 1]) + 1e-12
    d1 = np.abs(lines1[:, 0] * p1[:, 0] + lines1[:, 1] * p1[:, 1] + lines1[:, 2])
    n1 = np.hypot(lines1[:, 0], lines1[:, 1]) + 1e-12
    return 0.5 * (d1 / n1 + d2 / n2)


def epipolar_inlier_mask_from_motion(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    *,
    distance_thresh_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify correspondences as epipolar inliers using motion ``E = [t]_× R`` (not vision RANSAC).

    Returns ``(E, mask)`` with ``mask`` shaped ``(N, 1)`` ``uint8`` (1 = inlier).
    """
    n = len(pts1)
    E = essential_from_R_t(R, t)
    if n == 0 or float(np.linalg.norm(t)) < 1e-12:
        return E, np.zeros((n, 1), dtype=np.uint8)
    F = fundamental_from_essential(E, K)
    dists = symmetric_epipolar_distances(pts1, pts2, F)
    inlier = (dists <= float(distance_thresh_px)).astype(np.uint8)
    return E, inlier.reshape(-1, 1)


def vision_rotation_odom_translation_scale(
    R_vis: np.ndarray,
    t_vis: np.ndarray,
    t_odom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """
    Use vision rotation and translation *direction*; set translation length from odometry.

    Returns ``(R_est, t_est, ok, s)`` with ``R_est = R_vis`` and ``t_est = s * t_vis'`` where
    ``t_vis'`` is ``t_vis`` flipped if needed to agree in sign with ``t_odom``, and ``s`` is
    from :func:`scale_from_odometry` so ``||t_est|| ≈ ||t_odom||`` when ``ok`` is True.
    """
    R_out = np.asarray(R_vis, dtype=np.float64)
    tv = align_translation_direction(np.asarray(t_vis, dtype=np.float64), np.asarray(t_odom, dtype=np.float64))
    s, ok = scale_from_odometry(tv, t_odom)
    t_out = (s * tv).astype(np.float64)
    return R_out, t_out.reshape(3, 1), ok, float(s)

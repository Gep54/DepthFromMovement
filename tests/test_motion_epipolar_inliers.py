"""Epipolar inliers from motion essential matrix (not findEssentialMat RANSAC)."""

from __future__ import annotations

import numpy as np

from pipeline.geometry import (
    epipolar_inlier_mask_from_motion,
    essential_from_R_t,
    fundamental_from_essential,
    relative_motion_from_world_poses,
    symmetric_epipolar_distances,
)


def test_symmetric_epipolar_zero_for_consistent_match() -> None:
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    E = essential_from_R_t(R, t)
    F = fundamental_from_essential(E, K)
    pts1 = np.array([[100.0, 120.0], [400.0, 300.0]], dtype=np.float64)
    lines2 = __import__("cv2").computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    pts2 = []
    for a, b, c in lines2:
        y = -c / b if abs(b) > 1e-6 else 120.0
        pts2.append([pts1[len(pts2), 0], float(y)])
    pts2 = np.array(pts2, dtype=np.float64)
    d = symmetric_epipolar_distances(pts1, pts2, F)
    assert np.all(d < 1e-3)


def test_motion_inlier_mask_rejects_large_epipolar_error() -> None:
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([[0.2], [0.0], [0.0]], dtype=np.float64)
    pts1 = np.array([[100.0, 120.0]], dtype=np.float32)
    pts2 = np.array([[500.0, 400.0]], dtype=np.float32)
    _, mask = epipolar_inlier_mask_from_motion(
        pts1, pts2, R, t, K, distance_thresh_px=3.0
    )
    assert mask.ravel()[0] == 0


def test_motion_inlier_mask_from_world_poses() -> None:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = 500.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0
    W1 = np.eye(4, dtype=np.float64)
    W2 = np.eye(4, dtype=np.float64)
    W2[:3, 3] = [0.15, 0.0, 0.0]
    R, t = relative_motion_from_world_poses(W1, W2)
    pts1 = np.array([[320.0, 240.0], [400.0, 300.0]], dtype=np.float32)
    pts2 = pts1 + np.array([1.0, 0.0], dtype=np.float32)
    E, mask = epipolar_inlier_mask_from_motion(
        pts1, pts2, R, t, K, distance_thresh_px=5.0
    )
    assert E.shape == (3, 3)
    assert mask.ravel().tolist() == [1, 1]

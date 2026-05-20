"""Odometry depth scale applies to camera-frame Z after triangulation, not to translation in P2."""

from __future__ import annotations

import numpy as np

from pipeline.map import IncrementalMap, MapConfig, TwoViewResult
from pipeline.triangulation import triangulate_cam1_frame


def test_depth_scale_only_affects_z_not_xy() -> None:
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t_unit = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    t_scaled = np.array([[5.0], [0.0], [0.0]], dtype=np.float64)
    pts1 = np.array([[320.0, 240.0]], dtype=np.float32)
    pts2 = np.array([[360.0, 240.0]], dtype=np.float32)

    X_u, mask_u = triangulate_cam1_frame(pts1, pts2, K, R, t_unit)
    X_s, mask_s = triangulate_cam1_frame(pts1, pts2, K, R, t_scaled)
    assert bool(mask_u[0]) and bool(mask_s[0])
    np.testing.assert_allclose(X_u[:2, 0], X_s[:2, 0], rtol=1e-6, atol=1e-6)
    assert abs(X_s[2, 0] / X_u[2, 0] - 5.0) < 0.05

    X_u[2, mask_u] *= 5.0
    np.testing.assert_allclose(X_u[:3, 0], X_s[:3, 0], rtol=1e-5, atol=1e-5)


def test_two_view_result_carries_cam1_points() -> None:
    X = np.zeros((4, 1), dtype=np.float64)
    X[:3, 0] = [0.1, 0.2, 3.0]
    tw = TwoViewResult(
        frame_i=0,
        frame_j=1,
        pts1=np.zeros((1, 2), np.float32),
        pts2=np.zeros((1, 2), np.float32),
        inlier_mask=np.ones((1, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=5.0,
        scale_ok=True,
        X_cam1_h=X,
        X_world_h=X,
        cheiral_mask=np.array([True]),
        reproj={},
    )
    np.testing.assert_allclose(tw.X_cam1_h[:3, 0], [0.1, 0.2, 3.0])

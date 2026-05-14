"""Tests for vision rotation + odometry translation magnitude composition."""

from __future__ import annotations

import numpy as np

from pipeline.geometry import vision_rotation_odom_translation_scale


def test_vision_rotation_odom_translation_scale_matches_odom_norm() -> None:
    R_vis = np.eye(3, dtype=np.float64)
    t_vis = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    t_odom = np.array([[5.0], [0.0], [0.0]], dtype=np.float64)
    R_est, t_est, ok, s = vision_rotation_odom_translation_scale(R_vis, t_vis, t_odom)
    assert ok
    assert np.allclose(R_est, R_vis)
    assert np.allclose(t_est.ravel(), [5.0, 0.0, 0.0])
    assert abs(s - 5.0) < 1e-9
    assert abs(float(np.linalg.norm(t_est)) - 5.0) < 1e-9


def test_vision_rotation_odom_translation_scale_flips_sign_with_odom() -> None:
    R_vis = np.eye(3, dtype=np.float64)
    t_vis = np.array([[-1.0], [0.0], [0.0]], dtype=np.float64)
    t_odom = np.array([[2.0], [0.0], [0.0]], dtype=np.float64)
    R_est, t_est, ok, _s = vision_rotation_odom_translation_scale(R_vis, t_vis, t_odom)
    assert ok
    assert np.allclose(R_est, R_vis)
    assert np.allclose(t_est.ravel(), [2.0, 0.0, 0.0])


def test_vision_rotation_odom_translation_scale_degenerate_returns_not_ok() -> None:
    R_vis = np.eye(3, dtype=np.float64)
    t_vis = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    t_odom = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    R_est, t_est, ok, s = vision_rotation_odom_translation_scale(R_vis, t_vis, t_odom)
    assert not ok
    assert np.allclose(R_est, R_vis)
    assert s == 1.0
    assert np.allclose(t_est.ravel(), 0.0)

"""Fixed camera-axis remap for ROS world_T_camera composition."""

from __future__ import annotations

import numpy as np

from incremental_vo_ros2.camera_axes import apply_camera_axes_swap, camera_axes_swap_rotation


def test_opencv_to_body_race_maps_axes() -> None:
    R = camera_axes_swap_rotation("opencv_to_body_race")
    assert R is not None
    np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-12


def test_apply_swap_post_multiplies_on_world_T() -> None:
    R = camera_axes_swap_rotation("race")
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [10.0, 20.0, 30.0]
    out = apply_camera_axes_swap(T, R)
    np.testing.assert_allclose(out[:3, 3], T[:3, 3])
    np.testing.assert_allclose(out[:3, :3], R, atol=1e-12)

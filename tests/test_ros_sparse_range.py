from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.range_gate import (  # noqa: E402
    consecutive_keyframe_baseline_m,
    max_sparse_range_m,
)
from incremental_vo_ros2.frame_axes import (  # noqa: E402
    R_OPENCV_CAM_TO_BODY,
    rotate_points_opencv_cam_to_body,
    transform_points_opencv_cam0_to_world,
)


def test_consecutive_keyframe_baseline_m() -> None:
    Wi = np.eye(4, dtype=np.float64)
    Wj = np.eye(4, dtype=np.float64)
    Wj[:3, 3] = [3.0, 4.0, 0.0]
    assert consecutive_keyframe_baseline_m(Wi, Wj) == 5.0


def test_max_sparse_range_m_factor_and_floor() -> None:
    assert max_sparse_range_m(0.5, 100.0) == 50.0
    assert max_sparse_range_m(1e-6, 100.0, min_baseline_m=1e-3) == 0.1
    assert max_sparse_range_m(0.5, 0.0) is None
    assert max_sparse_range_m(0.5, -1.0) is None


def test_opencv_cam_to_body_rotation_axes() -> None:
    ex, ey, ez = np.eye(3, dtype=np.float64)
    body_x = rotate_points_opencv_cam_to_body(ez.reshape(1, 3))[0]
    body_y = rotate_points_opencv_cam_to_body(ex.reshape(1, 3))[0]
    body_z = rotate_points_opencv_cam_to_body(ey.reshape(1, 3))[0]
    np.testing.assert_allclose(body_x, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(body_y, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(body_z, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(R_OPENCV_CAM_TO_BODY @ ez, [1.0, 0.0, 0.0], atol=1e-12)


def test_transform_opencv_cam0_to_world_composes_rotation() -> None:
    pts_cam = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [10.0, 20.0, 30.0]
    pts_w = transform_points_opencv_cam0_to_world(pts_cam, T)
    np.testing.assert_allclose(pts_w, [[12.0, 20.0, 30.0]], atol=1e-12)

from __future__ import annotations

import numpy as np

from pipeline.frame_axes import (
    R_OPENCV_CAM_TO_BODY,
    opencv_cam_point_to_cam0,
    rotate_opencv_cam_to_body,
    step_one_opencv_cam_depth_axis,
    world_T_body_to_world_T_opencv_cam,
)
from pipeline.geometry import relative_motion_from_world_poses


def test_opencv_cam_to_body_rotation_axes() -> None:
    ex, ey, ez = np.eye(3, dtype=np.float64)
    np.testing.assert_allclose(rotate_opencv_cam_to_body(ez), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotate_opencv_cam_to_body(ex), [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotate_opencv_cam_to_body(ey), [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(R_OPENCV_CAM_TO_BODY @ ez, [1.0, 0.0, 0.0], atol=1e-12)


def test_step_one_only_remaps_depth_not_lateral() -> None:
    np.testing.assert_allclose(
        step_one_opencv_cam_depth_axis(np.array([1.0, 2.0, 3.0])),
        [3.0, 1.0, 2.0],
        atol=1e-12,
    )
    full = rotate_opencv_cam_to_body(np.array([1.0, 2.0, 3.0]))
    partial = step_one_opencv_cam_depth_axis(np.array([1.0, 2.0, 3.0]))
    assert not np.allclose(full, partial)


def test_opencv_cam_point_to_cam0_identity_poses() -> None:
    W = np.eye(4, dtype=np.float64)
    np.testing.assert_allclose(
        opencv_cam_point_to_cam0(np.array([0.0, 0.0, 2.0]), W, W),
        [2.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_world_T_body_to_opencv_changes_relative_translation() -> None:
    W0 = np.eye(4, dtype=np.float64)
    W1 = np.eye(4, dtype=np.float64)
    W1[:3, 3] = [1.0, 0.0, 0.0]
    _, t_body = relative_motion_from_world_poses(W0, W1)
    _, t_ocv = relative_motion_from_world_poses(
        world_T_body_to_world_T_opencv_cam(W0),
        world_T_body_to_world_T_opencv_cam(W1),
    )
    assert not np.allclose(t_body.ravel(), t_ocv.ravel(), atol=1e-9)

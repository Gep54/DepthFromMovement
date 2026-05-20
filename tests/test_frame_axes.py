from __future__ import annotations

import numpy as np

from pipeline.frame_axes import (
    R_OPENCV_CAM_TO_BODY,
    opencv_cam_point_to_cam0,
    rotate_opencv_cam_to_body,
)


def test_opencv_cam_to_body_rotation_axes() -> None:
    ex, ey, ez = np.eye(3, dtype=np.float64)
    np.testing.assert_allclose(rotate_opencv_cam_to_body(ez), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotate_opencv_cam_to_body(ex), [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotate_opencv_cam_to_body(ey), [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(R_OPENCV_CAM_TO_BODY @ ez, [1.0, 0.0, 0.0], atol=1e-12)


def test_opencv_cam_point_to_cam0_identity_poses() -> None:
    W = np.eye(4, dtype=np.float64)
    np.testing.assert_allclose(
        opencv_cam_point_to_cam0(np.array([0.0, 0.0, 2.0]), W, W),
        [2.0, 0.0, 0.0],
        atol=1e-12,
    )

"""OpenCV camera vs body/FCU axis conventions."""

from __future__ import annotations

import numpy as np

from pipeline.geometry import invert_se3

# OpenCV camera (Z forward, X right, Y down) -> body/odom child (X forward, Y left, Z up).
# Matches static TF fcu -> rgb in race rosbag (quaternion 0.5, -0.5, 0.5, -0.5).
R_OPENCV_CAM_TO_BODY = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def rotate_opencv_cam_to_body(x: np.ndarray) -> np.ndarray:
    """Map a 3-vector from OpenCV camera axes to body/FCU axes."""
    v = np.asarray(x, dtype=np.float64).reshape(3)
    return (R_OPENCV_CAM_TO_BODY @ v).astype(np.float64)


def opencv_cam_point_to_cam0(
    X_w: np.ndarray,
    world_T_camera_0: np.ndarray,
    world_T_camera_j: np.ndarray,
) -> np.ndarray:
    """
    Map a world point into camera-0 storage coordinates.

    Pipeline: world -> OpenCV cam-j -> body axes -> cam-0 (body-aligned at keyframe 0).
    ``world_T_camera_*`` are camera/body->world poses (same convention as ``motion.json``).
    """
    Xw = np.asarray(X_w, dtype=np.float64).reshape(3)
    W0 = np.asarray(world_T_camera_0, dtype=np.float64)
    Wj = np.asarray(world_T_camera_j, dtype=np.float64)
    X_cj = (invert_se3(Wj) @ np.array([Xw[0], Xw[1], Xw[2], 1.0], dtype=np.float64))[:3]
    X_cj_body = rotate_opencv_cam_to_body(X_cj)
    T_c0_cj = invert_se3(W0) @ Wj
    return (T_c0_cj @ np.array([X_cj_body[0], X_cj_body[1], X_cj_body[2], 1.0], dtype=np.float64))[:3]

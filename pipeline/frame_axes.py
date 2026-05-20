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


def body_T_opencv_cam() -> np.ndarray:
    """Fixed ``body_T_cam`` (OpenCV optical -> body/FCU child); rotation only, zero translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_OPENCV_CAM_TO_BODY
    return T


def world_T_body_to_world_T_opencv_cam(world_T_body: np.ndarray) -> np.ndarray:
    """
    Convert an odometry child (body/FCU) ``world_T_*`` pose to OpenCV optical ``world_T_camera``.

    ``X_world = world_T_body @ body_T_opencv_cam @ X_cam``.
    """
    W = np.asarray(world_T_body, dtype=np.float64)
    return W @ body_T_opencv_cam()


def rotate_opencv_cam_to_body(x: np.ndarray) -> np.ndarray:
    """Map a 3-vector from OpenCV camera axes to body/FCU axes."""
    v = np.asarray(x, dtype=np.float64).reshape(3)
    return (R_OPENCV_CAM_TO_BODY @ v).astype(np.float64)


def step_one_opencv_cam_depth_axis(x: np.ndarray) -> np.ndarray:
    """
    Apply only the forward-depth part of OpenCV→body (step one): cam **Z** → storage **X**.

    Image-plane ``x,y`` stay in OpenCV lateral coordinates; this is not a full ``R @ X``.
    """
    v = np.asarray(x, dtype=np.float64).reshape(3)
    forward = float((R_OPENCV_CAM_TO_BODY @ v)[0])
    return np.array([forward, v[0], v[1]], dtype=np.float64)


def opencv_cam_point_to_cam0(
    X_opencv_cam_i: np.ndarray,
    world_T_camera_0: np.ndarray,
    world_T_camera_i: np.ndarray,
) -> np.ndarray:
    """
    Map a triangulated point in OpenCV camera-``i`` into camera-0 storage coordinates.

    Step one (depth axis only) runs on the camera-frame point; ``world_T_camera_0`` is not
    modified—only ``T_c0_ci = inv(W0) @ Wi`` maps the adjusted point into keyframe-0.
    """
    X_adj = step_one_opencv_cam_depth_axis(X_opencv_cam_i)
    W0 = np.asarray(world_T_camera_0, dtype=np.float64)
    Wi = np.asarray(world_T_camera_i, dtype=np.float64)
    T_c0_ci = invert_se3(W0) @ Wi
    return (T_c0_ci @ np.array([X_adj[0], X_adj[1], X_adj[2], 1.0], dtype=np.float64))[:3]

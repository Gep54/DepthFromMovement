"""OpenCV camera vs body/FCU axis conventions (no ROS imports)."""

from __future__ import annotations

import numpy as np

# OpenCV camera (Z forward, X right, Y down) -> body/odom child (X forward, Y left, Z up).
# Matches static TF fcu -> rgb in race rosbag (quaternion 0.5, -0.5, 0.5, -0.5).
R_OPENCV_CAM_TO_BODY = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def rotate_points_opencv_cam_to_body(points_nx3: np.ndarray) -> np.ndarray:
    """Apply fixed ``R_OPENCV_CAM_TO_BODY`` to N×3 OpenCV camera-frame points (row vectors)."""
    p = np.asarray(points_nx3, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return p @ R_OPENCV_CAM_TO_BODY.T


def transform_points_rowvectors(
    points_nx3: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """``X_out = X_in @ R.T + t`` for N×3 row vectors."""
    p = np.asarray(points_nx3, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return (p @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64).reshape(1, 3)


def transform_points_opencv_cam0_to_world(
    points_nx3: np.ndarray, world_T_odom_child: np.ndarray
) -> np.ndarray:
    """
    Map OpenCV cam-0 landmarks to metric world.

    ``world_T_odom_child`` is body→world from odometry; landmarks use OpenCV optical axes.
    """
    T = np.asarray(world_T_odom_child, dtype=np.float64)
    return transform_points_rowvectors(
        rotate_points_opencv_cam_to_body(points_nx3),
        T[:3, :3],
        T[:3, 3],
    )

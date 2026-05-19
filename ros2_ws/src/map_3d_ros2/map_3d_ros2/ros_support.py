"""Helpers: pipeline path, ROS Image → gray, Odometry/TF → SE(3), pose messages."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion, Transform, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from map_3d_ros2.image_buffer import (
    copy_image_msg,
    image_msg_to_gray_undistorted,
    ros_image_to_gray,
    should_buffer_image,
)

__all__ = [
    "BufferedFrame",
    "camera_T_world_to_pose_stamped",
    "camera_info_to_calibration",
    "copy_image_msg",
    "effective_K_from_calibration",
    "ensure_pipeline_on_path",
    "image_msg_to_gray_undistorted",
    "odom_to_cam_to_world_T",
    "quat_msg_to_mat",
    "ros_image_to_gray",
    "should_buffer_image",
    "transform_msg_to_world_T",
    "transform_stamped_to_world_T",
    "transform_points_world_T_camera",
    "world_T_camera_to_quaternion_xyzw",
    "world_T_camera_to_pose_stamped",
]


def ensure_pipeline_on_path() -> Path | None:
    """Insert DepthFromMovement repo root on ``sys.path`` so ``pipeline.*`` imports work."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pipeline" / "map.py").is_file():
            root = str(p)
            if root not in sys.path:
                sys.path.insert(0, root)
            return p
    return None


def quat_msg_to_mat(q: Quaternion) -> np.ndarray:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    n = math.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_msg_to_world_T(tr: Transform) -> np.ndarray:
    t = tr.translation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_msg_to_mat(tr.rotation)
    T[:3, 3] = (t.x, t.y, t.z)
    return T


def transform_stamped_to_world_T(msg: TransformStamped) -> np.ndarray:
    return transform_msg_to_world_T(msg.transform)


def odom_to_cam_to_world_T(msg: Odometry) -> np.ndarray:
    """4×4 camera→world (``world_T_camera``): X_world = T @ X_cam."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_msg_to_mat(q)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def world_T_camera_to_quaternion_xyzw(T: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    tr = float(np.trace(R))
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    return (x / n, y / n, z / n, w / n)


def world_T_camera_to_pose_stamped(T: np.ndarray, *, header: Header) -> PoseStamped:
    """4×4 ``world_T_camera`` (cam→world) → ``PoseStamped`` in ``header.frame_id``."""
    T = np.asarray(T, dtype=np.float64)
    qx, qy, qz, qw = world_T_camera_to_quaternion_xyzw(T)
    msg = PoseStamped()
    msg.header = header
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def camera_T_world_to_pose_stamped(world_T_camera: np.ndarray, *, header: Header) -> PoseStamped:
    """Publish ``camera_T_world = invert(world_T_camera)`` as a ``PoseStamped``.

    Points are lifted with ``world_T_camera``; this pose maps world coordinates back into
    the camera frame (inverse of the point transform).
    """
    ensure_pipeline_on_path()
    from pipeline.geometry import invert_se3

    T_inv = invert_se3(np.asarray(world_T_camera, dtype=np.float64))
    return world_T_camera_to_pose_stamped(T_inv, header=header)


def camera_info_to_calibration(msg: CameraInfo):
    ensure_pipeline_on_path()
    from data.camera_calibration import calibration_from_intrinsics

    return calibration_from_intrinsics(
        K_flat=msg.k,
        D=msg.d,
        width=int(msg.width),
        height=int(msg.height),
        distortion_model=msg.distortion_model,
    )


def effective_K_from_calibration(cal) -> np.ndarray:
    if cal.dist_coeffs is None or np.linalg.norm(cal.dist_coeffs) < 1e-9:
        return np.asarray(cal.K, dtype=np.float64)
    if cal.image_size is None:
        return np.asarray(cal.K, dtype=np.float64)
    w, h = cal.image_size
    new_K, _ = cv2.getOptimalNewCameraMatrix(cal.K, cal.dist_coeffs, (w, h), alpha=0)
    return np.asarray(new_K, dtype=np.float64)


def transform_points_world_T_camera(points_nx3: np.ndarray, world_T_camera: np.ndarray) -> np.ndarray:
    """Map N×3 points in camera frame to world: ``X_w = world_T_camera @ X_c``."""
    T = np.asarray(world_T_camera, dtype=np.float64)
    p = np.asarray(points_nx3, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ones = np.ones((p.shape[0], 1), dtype=np.float64)
    ph = np.hstack([p, ones])
    return (ph @ T.T)[:, :3]


@dataclass
class BufferedFrame:
    stamp_sec: int
    stamp_nsec: int
    image_msg: Image
    pos_odom: np.ndarray
    cam_to_world: np.ndarray
    qx: float
    qy: float
    qz: float
    qw: float

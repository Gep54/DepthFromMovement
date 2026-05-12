"""Helpers: locate repo for ``pipeline`` imports, ROS Image → gray, Odometry → SE(3), disk output."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


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
    """Unit quaternion (x,y,z,w) → rotation matrix; no SciPy (matches ROS ``tf`` convention)."""
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


def odom_to_cam_to_world_T(msg: Odometry) -> np.ndarray:
    """
    Build 4×4 ``cam_to_world`` (maps camera frame into ``msg.header.frame_id``), matching
    ``pipeline``'s ``world_T_camera`` naming in datasets: X_world = T @ X_cam columns.
    """
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_msg_to_mat(q)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def odom_position_xyz(msg: Odometry) -> np.ndarray:
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def world_T_camera_to_quaternion_xyzw(T: np.ndarray) -> tuple[float, float, float, float]:
    """
    Rotation part of 4×4 ``world_T_camera`` → unit quaternion ``(x, y, z, w)``
    (``geometry_msgs/Quaternion`` order).
    """
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


def pose_stamped_to_world_T_camera(msg) -> np.ndarray:
    """``geometry_msgs/PoseStamped`` pose → 4×4 homogeneous camera→world."""
    p = msg.pose.position
    q = msg.pose.orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_msg_to_mat(q)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def ros_image_to_gray(msg: Image) -> np.ndarray | None:
    """Decode ``sensor_msgs/Image`` to single-channel uint8; returns None if encoding unsupported."""
    h, w = int(msg.height), int(msg.width)
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("mono8", "8UC1"):
        if arr.size != h * w:
            return None
        return arr.reshape((h, w))
    if msg.encoding in ("bgr8", "8UC3"):
        if arr.size != h * w * 3:
            return None
        bgr = arr.reshape((h, w, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if msg.encoding in ("rgb8", "rgba8"):
        step = 4 if msg.encoding == "rgba8" else 3
        if arr.size != h * w * step:
            return None
        rgb = arr.reshape((h, w, step))[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return None


@dataclass
class BufferedFrame:
    stamp_sec: int
    stamp_nsec: int
    gray: np.ndarray
    pos_odom: np.ndarray  # (3,) translation used for keyframe distance (fused when fusion active)
    cam_to_world: np.ndarray  # 4×4 fused camera→world for triangulation
    qx: float
    qy: float
    qz: float
    qw: float


def save_keyframe_manifest(
    path: Path,
    *,
    frame_id: str,
    keyframe_distance_m: float,
    odom_child_frame: str,
    odom_header_frame: str,
    records: list[dict],
    fusion_method: str | None = None,
    provided_pose_topic: str | None = None,
) -> None:
    payload = {
        "odom_header_frame_id": odom_header_frame,
        "odom_child_frame_id": odom_child_frame,
        "image_header_frame_id": frame_id,
        "keyframe_distance_m": keyframe_distance_m,
        "keyframes": records,
    }
    if fusion_method is not None:
        payload["fusion_method"] = fusion_method
    if provided_pose_topic is not None:
        payload["provided_pose_topic"] = provided_pose_topic
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_sparse_map_npz(path: Path, points_xyz: np.ndarray) -> None:
    if points_xyz.size == 0:
        np.savez_compressed(str(path), points=np.zeros((0, 3), dtype=np.float64))
    else:
        np.savez_compressed(str(path), points=np.asarray(points_xyz, dtype=np.float64))

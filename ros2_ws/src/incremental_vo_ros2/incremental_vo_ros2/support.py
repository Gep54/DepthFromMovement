"""Helpers: locate repo for ``pipeline`` imports, ROS Image → gray, Odometry → SE(3), disk output."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image

from incremental_vo_ros2.se3 import (
    rigid_transform_to_matrix4,
    transform_points_world_T_camera,
    world_T_camera_from_odom_extrinsic,
)
from incremental_vo_ros2.image_buffer import (
    copy_image_msg,
    image_msg_to_gray_undistorted,
    ros_image_to_gray,
    should_buffer_image,
)
from incremental_vo_ros2.offline_dataset import offline_dataset_image_basename
from incremental_vo_ros2.range_gate import (
    consecutive_keyframe_baseline_m,
    max_sparse_range_m,
)

__all__ = [
    "BufferedFrame",
    "camera_info_to_calibration",
    "consecutive_keyframe_baseline_m",
    "copy_image_msg",
    "ensure_pipeline_on_path",
    "eval_world_T_camera0_from_parameter",
    "image_msg_to_gray_undistorted",
    "max_sparse_range_m",
    "odom_position_xyz",
    "odom_to_cam_to_world_T",
    "rigid_transform_to_matrix4",
    "world_T_camera_from_odom_extrinsic",
    "offline_dataset_image_basename",
    "pose_stamped_to_world_T_camera",
    "quat_msg_to_mat",
    "ros_image_to_gray",
    "save_keyframe_manifest",
    "save_sparse_map_eval_world_npz",
    "save_sparse_map_npz",
    "should_buffer_image",
    "transform_points_world_T_camera",
    "effective_K_from_calibration",
    "undistort_gray_if_needed",
    "world_T_camera_to_quaternion_xyzw",
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
    """Unit quaternion (x,y,z,w) → rotation matrix; no SciPy (matches ROS ``tf`` convention)."""
    from incremental_vo_ros2.se3 import quat_xyzw_to_mat

    return quat_xyzw_to_mat(float(q.x), float(q.y), float(q.z), float(q.w))


def odom_to_cam_to_world_T(msg: Odometry) -> np.ndarray:
    """
    Build 4×4 ``world_T_odom_child`` from ``nav_msgs/Odometry`` (pose of ``child_frame_id`` in
    ``header.frame_id``). This is **not** the optical camera frame unless ``child_frame_id`` is optical.
    """
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return rigid_transform_to_matrix4(p, q)


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


def camera_info_to_calibration(msg: CameraInfo):
    """``sensor_msgs/CameraInfo`` → :class:`data.schema.Calibration``."""
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
    """Intrinsics matrix used for triangulation after optional undistortion."""
    if cal.dist_coeffs is None or np.linalg.norm(cal.dist_coeffs) < 1e-9:
        return np.asarray(cal.K, dtype=np.float64)
    if cal.image_size is None:
        return np.asarray(cal.K, dtype=np.float64)
    w, h = cal.image_size
    new_K, _ = cv2.getOptimalNewCameraMatrix(cal.K, cal.dist_coeffs, (w, h), alpha=0)
    return np.asarray(new_K, dtype=np.float64)


def undistort_gray_if_needed(gray: np.ndarray, cal) -> tuple[np.ndarray, np.ndarray]:
    """Return (grayscale, K) for feature detection / triangulation.

    When distortion is negligible, returns the input and ``cal.K``. Otherwise
    undistorts and returns ``new_K`` from ``cv2.getOptimalNewCameraMatrix``.
    """
    if cal.dist_coeffs is None or np.linalg.norm(cal.dist_coeffs) < 1e-9:
        return gray, np.asarray(cal.K, dtype=np.float64)
    new_K = effective_K_from_calibration(cal)
    und = cv2.undistort(gray, cal.K, cal.dist_coeffs, None, newK=new_K)
    return und, new_K


@dataclass
class BufferedFrame:
    stamp_sec: int
    stamp_nsec: int
    image_msg: Image
    pos_odom: np.ndarray  # (3,) translation used for keyframe distance (fused when fusion active)
    cam_to_world: np.ndarray  # 4×4 fused camera→world for triangulation
    odom_cam_to_world: np.ndarray  # 4×4 raw odom_main camera→world at buffer time
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
    pair_lookback: int | None = None,
    fusion_method: str | None = None,
    provided_pose_topic: str | None = None,
    feature_method: str | None = None,
    feature_n_features: int | None = None,
    descriptor_merge_beta: float | None = None,
    descriptor_max_match_distance: float | None = None,
    descriptor_ratio_second_best: float | None = None,
    landmarks_reference_frame: str | None = None,
    map_coordinate_frame: str | None = None,
    eval_world_T_camera0_flat16: Sequence[float] | None = None,
) -> None:
    payload = {
        "odom_header_frame_id": odom_header_frame,
        "odom_child_frame_id": odom_child_frame,
        "image_header_frame_id": frame_id,
        "keyframe_distance_m": keyframe_distance_m,
        "keyframes": records,
    }
    if pair_lookback is not None:
        payload["pair_lookback"] = int(pair_lookback)
    if fusion_method is not None:
        payload["fusion_method"] = fusion_method
    if provided_pose_topic is not None:
        payload["provided_pose_topic"] = provided_pose_topic
    if feature_method is not None:
        payload["feature_method"] = feature_method
    if feature_n_features is not None:
        payload["feature_n_features"] = int(feature_n_features)
    # ``merge_beta``/``ratio_second_best`` are legitimately ``None`` (mean-equivalent / disabled);
    # only the top-level keys are gated on caller intent, not on the value itself.
    if descriptor_merge_beta is not None or feature_method is not None:
        payload["descriptor_merge_beta"] = (
            None if descriptor_merge_beta is None else float(descriptor_merge_beta)
        )
    if descriptor_max_match_distance is not None:
        payload["descriptor_max_match_distance"] = float(descriptor_max_match_distance)
    if descriptor_ratio_second_best is not None or feature_method is not None:
        payload["descriptor_ratio_second_best"] = (
            None if descriptor_ratio_second_best is None else float(descriptor_ratio_second_best)
        )
    if landmarks_reference_frame is not None:
        payload["landmarks_reference_frame"] = landmarks_reference_frame
    if map_coordinate_frame is not None:
        payload["map_coordinate_frame"] = map_coordinate_frame
    if eval_world_T_camera0_flat16 is not None:
        payload["eval_world_T_camera0"] = [float(x) for x in eval_world_T_camera0_flat16]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def eval_world_T_camera0_from_parameter(values: Sequence[float]) -> np.ndarray | None:
    """Parse sixteen row-major floats into ``4×4`` **camera~0 → evaluation world**; ``None`` if invalid or all-zero."""
    a = np.asarray(list(values), dtype=np.float64).ravel()
    if a.size != 16:
        return None
    T = a.reshape((4, 4), order="C")
    if not np.all(np.isfinite(T)):
        return None
    if np.allclose(T, 0.0):
        return None
    R = T[:3, :3]
    if abs(np.linalg.det(R)) < 1e-12:
        return None
    return T.astype(np.float64)


def save_sparse_map_eval_world_npz(
    path: Path, points_cam0_xyz: np.ndarray, eval_world_T_camera0: np.ndarray
) -> None:
    """Save ``points`` in evaluation world: ``X_eval = T @ homog(X_cam0)`` (``T`` = ``eval_world_T_camera0``)."""
    X = np.asarray(points_cam0_xyz, dtype=np.float64)
    if X.size == 0:
        np.savez_compressed(str(path), points=np.zeros((0, 3), dtype=np.float64))
        return
    T = np.asarray(eval_world_T_camera0, dtype=np.float64)
    Xh = np.vstack([X.T, np.ones((1, X.shape[0]))])
    Xe = (T @ Xh)[:3].T.astype(np.float64)
    np.savez_compressed(str(path), points=Xe)


def save_sparse_map_npz(path: Path, points_xyz: np.ndarray) -> None:
    if points_xyz.size == 0:
        np.savez_compressed(str(path), points=np.zeros((0, 3), dtype=np.float64))
    else:
        np.savez_compressed(str(path), points=np.asarray(points_xyz, dtype=np.float64))

"""Pure-numpy SE(3) helpers (no ROS message imports)."""

from __future__ import annotations

import math

import numpy as np


def quat_xyzw_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unit quaternion (x,y,z,w) → rotation matrix."""
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


def rigid_transform_to_matrix4(translation, rotation) -> np.ndarray:
    """
    Build 4×4 **source→target** (``X_target = T @ X_source`` columns), matching ``tf2`` lookup.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_mat(
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
        float(rotation.w),
    )
    T[:3, 3] = (
        float(translation.x),
        float(translation.y),
        float(translation.z),
    )
    return T


def world_T_camera_from_odom_extrinsic(
    world_T_odom_child: np.ndarray,
    odom_child_T_optical: np.ndarray | None,
) -> np.ndarray:
    """``world_T_optical = world_T_odom_child @ odom_child_T_optical``."""
    W = np.asarray(world_T_odom_child, dtype=np.float64).reshape(4, 4)
    if odom_child_T_optical is None:
        return W.copy()
    E = np.asarray(odom_child_T_optical, dtype=np.float64).reshape(4, 4)
    return W @ E


def camera_z_arrow_endpoints_world(
    world_T_camera: np.ndarray, length_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tail and head of an arrow along camera **+Z** (viewing axis) expressed in world.

    ``world_T_camera`` maps camera → world: ``X_w = R @ X_c + t``.
    """
    L = float(length_m)
    if L <= 0.0:
        raise ValueError("length_m must be positive")
    T = np.asarray(world_T_camera, dtype=np.float64).reshape(4, 4)
    tail = T[:3, 3].copy()
    head = tail + L * T[:3, 2]
    return tail, head


def transform_points_world_T_camera(points_nx3: np.ndarray, world_T_camera: np.ndarray) -> np.ndarray:
    """Map N×3 points in camera frame to world: ``X_w = R @ X_c + t``."""
    T = np.asarray(world_T_camera, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    p = np.asarray(points_nx3, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return (p @ R.T) + t

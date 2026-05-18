"""Unit tests for odom + TF extrinsic composition (no ROS runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.se3 import (  # noqa: E402
    rigid_transform_to_matrix4,
    transform_points_world_T_camera,
    world_T_camera_from_odom_extrinsic,
)


class _Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x, self.y, self.z, self.w = x, y, z, w


def test_world_T_camera_from_odom_extrinsic_none_is_identity_chain() -> None:
    W = np.eye(4, dtype=np.float64)
    W[:3, 3] = [1.0, 2.0, 3.0]
    out = world_T_camera_from_odom_extrinsic(W, None)
    assert np.allclose(out, W)


def test_optical_z_maps_to_parent_x_when_extrinsic_rotates_z_to_x() -> None:
    """Simulate fcu (+X forward) vs optical (+Z forward): parent_T_optical maps Z_cam to X_parent."""
    parent_T_optical = np.eye(4, dtype=np.float64)
    parent_T_optical[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    world_T_parent = np.eye(4, dtype=np.float64)
    world_T_optical = world_T_camera_from_odom_extrinsic(world_T_parent, parent_T_optical)
    depth_optical = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    p_world = transform_points_world_T_camera(depth_optical, world_T_optical)
    assert np.allclose(p_world[0], [5.0, 0.0, 0.0], atol=1e-9)


def test_rigid_transform_to_matrix4_matches_manual() -> None:
    t = _Vec3(1.0, 0.0, 0.0)
    q = _Quat(0.0, 0.0, 0.0, 1.0)
    T = rigid_transform_to_matrix4(t, q)
    assert np.allclose(T[:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(T[:3, :3], np.eye(3))

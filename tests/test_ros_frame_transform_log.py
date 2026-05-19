"""Tests for ROS frame-transform debug formatting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
for p in (str(_REPO), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from incremental_vo_ros2.frame_transform_debug import (  # noqa: E402
    drone_T_camera_from_world_poses,
    format_se3_4x4,
)


def test_format_se3_4x4_identity() -> None:
    out = format_se3_4x4(np.eye(4))
    assert "|" in out
    assert "+1.0000" in out
    assert out.count("\n") == 3


def test_format_se3_4x4_bad_shape() -> None:
    with pytest.raises(ValueError, match="4x4"):
        format_se3_4x4(np.eye(3))


def test_drone_T_camera_round_trip() -> None:
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    world_T_drone = np.eye(4, dtype=np.float64)
    world_T_drone[:3, :3] = R
    world_T_drone[:3, 3] = [1.0, 2.0, 3.0]
    drone_T_cam = np.eye(4, dtype=np.float64)
    drone_T_cam[:3, 3] = [0.1, 0.2, 0.3]
    world_T_camera = world_T_drone @ drone_T_cam
    recovered = drone_T_camera_from_world_poses(world_T_drone, world_T_camera)
    assert np.allclose(recovered, drone_T_cam)
    assert np.allclose(world_T_drone @ recovered, world_T_camera)

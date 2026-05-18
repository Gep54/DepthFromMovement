"""Unit tests for keyframe camera +Z arrow geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.se3 import camera_z_arrow_endpoints_world  # noqa: E402


def test_camera_z_arrow_along_world_z_when_camera_identity() -> None:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1.0, 2.0, 3.0]
    tail, head = camera_z_arrow_endpoints_world(T, 0.5)
    assert np.allclose(tail, [1.0, 2.0, 3.0])
    assert np.allclose(head, [1.0, 2.0, 3.5])


def test_camera_z_arrow_rotated_with_camera() -> None:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    tail, head = camera_z_arrow_endpoints_world(T, 2.0)
    assert np.allclose(tail, 0.0)
    assert np.allclose(head, [2.0, 0.0, 0.0])

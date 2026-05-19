"""Unit tests for odometry history + SE(3) helpers (no ROS runtime)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_PKG = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.odom_history import (  # noqa: E402
    OdomHistory,
    compose_se3,
    odom_stamp_nsec,
)


def _odom(sec: int, nsec: int, x: float) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nsec), frame_id="world"),
        child_frame_id="body",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
    )


def test_odom_stamp_nsec() -> None:
    msg = _odom(1, 500_000_000, 0.0)
    assert odom_stamp_nsec(msg) == 1_500_000_000


def test_odom_history_nearest() -> None:
    hist = OdomHistory(maxlen=8)
    hist.push(_odom(0, 0, 0.0))
    hist.push(_odom(0, 300_000_000, 0.3))
    hist.push(_odom(0, 600_000_000, 0.6))
    near = hist.nearest(0, 290_000_000)
    assert near is not None
    assert float(near.pose.pose.position.x) == 0.3


def test_compose_se3_translation() -> None:
    Ta = np.eye(4, dtype=np.float64)
    Ta[:3, 3] = [1.0, 0.0, 0.0]
    Tb = np.eye(4, dtype=np.float64)
    Tb[:3, 3] = [0.0, 2.0, 0.0]
    T = compose_se3(Ta, Tb)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 0.0])


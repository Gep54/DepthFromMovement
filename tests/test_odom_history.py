"""Unit tests for incremental_vo_ros2.odom_history."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.odom_history import OdomHistory  # noqa: E402


class _FakeStamp:
    def __init__(self, sec: int, nanosec: int) -> None:
        self.sec = sec
        self.nanosec = nanosec


class _FakeOdom:
    def __init__(self, sec: int, nanosec: int, tag: str) -> None:
        self.header = type("H", (), {"stamp": _FakeStamp(sec, nanosec)})()
        self.tag = tag


def test_odom_history_nearest() -> None:
    hist = OdomHistory(maxlen=10)
    hist.push(_FakeOdom(1, 0, "a"))
    hist.push(_FakeOdom(1, 500_000_000, "b"))
    hist.push(_FakeOdom(2, 0, "c"))
    got = hist.nearest(1, 400_000_000)
    assert got is not None
    assert got.tag == "b"

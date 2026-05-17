"""Tests for offline dataset naming helpers (no ROS runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.offline_dataset import offline_dataset_image_basename  # noqa: E402


def test_offline_dataset_image_basename() -> None:
    assert offline_dataset_image_basename("frame", 0) == "frame_00000.png"
    assert offline_dataset_image_basename("frame", 42) == "frame_00042.png"
    assert offline_dataset_image_basename("", 1) == "frame_00001.png"
    assert offline_dataset_image_basename("  kf  ", 3) == "kf_00003.png"

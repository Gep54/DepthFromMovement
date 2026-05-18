from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.range_gate import should_reset_bag_replay  # noqa: E402


def test_should_reset_bag_replay_no_prior_stamp() -> None:
    assert not should_reset_bag_replay(None, 10.0, 1.0)


def test_should_reset_bag_replay_forward_ok() -> None:
    assert not should_reset_bag_replay(5.0, 6.0, 1.0)


def test_should_reset_bag_replay_backward_jump() -> None:
    assert should_reset_bag_replay(100.0, 98.0, 1.0)


def test_should_reset_bag_replay_small_backward_within_threshold() -> None:
    assert not should_reset_bag_replay(100.0, 99.5, 1.0)

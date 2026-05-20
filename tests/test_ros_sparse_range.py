from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.range_gate import (  # noqa: E402
    consecutive_keyframe_baseline_m,
    max_sparse_range_m,
)
def test_consecutive_keyframe_baseline_m() -> None:
    Wi = np.eye(4, dtype=np.float64)
    Wj = np.eye(4, dtype=np.float64)
    Wj[:3, 3] = [3.0, 4.0, 0.0]
    assert consecutive_keyframe_baseline_m(Wi, Wj) == 5.0


def test_max_sparse_range_m_factor_and_floor() -> None:
    assert max_sparse_range_m(0.5, 100.0) == 50.0
    assert max_sparse_range_m(1e-6, 100.0, min_baseline_m=1e-3) == 0.1
    assert max_sparse_range_m(0.5, 0.0) is None
    assert max_sparse_range_m(0.5, -1.0) is None

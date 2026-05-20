"""Triangulation max-range gate applied after cheirality in IncrementalMap."""

from __future__ import annotations

import numpy as np

from pipeline.map import MapConfig
from pipeline.range_gate import apply_max_range_gate_cam1, max_sparse_range_m


def test_max_sparse_range_m_floor() -> None:
    assert max_sparse_range_m(1e-6, 100.0) == 0.1


def test_apply_max_range_gate_cam1_squared() -> None:
    X = np.array(
        [
            [1.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cheiral = np.array([True, True, True], dtype=bool)
    # baseline 0.5 m, factor 10 -> max_range 5 m; keep (1,0,0) and (0,0,5), drop (10,0,0)
    out = apply_max_range_gate_cam1(X, cheiral, baseline_m=0.5, factor=10.0)
    assert out.tolist() == [True, False, True]


def test_map_config_range_factor_default_disabled() -> None:
    assert MapConfig().max_range_baseline_factor == 0.0

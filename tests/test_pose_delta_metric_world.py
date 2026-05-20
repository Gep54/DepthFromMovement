"""Poses stay in metric world (no camera-0 canonicalization)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.dataset import load_dataset
from viz.pose_delta import compute_pair_pose_delta


@pytest.mark.skipif(
    not Path("TestData/run_20260519_224032/motion.json").is_file(),
    reason="race offline export dataset not in tree",
)
def test_pair_5_6_motion_is_world_x() -> None:
    ds = load_dataset("TestData/run_20260519_224032")
    summary = compute_pair_pose_delta(
        ds.world_T_camera[5],
        ds.world_T_camera[6],
        frame_i=5,
        frame_j=6,
    )
    tw = np.asarray(summary["translation_world_m"], dtype=np.float64)
    assert abs(tw[0]) > 0.4
    assert abs(tw[0]) > abs(tw[2]) * 5
    assert summary["motion_vs_optical_axis_deg"] is not None
    assert summary["motion_vs_optical_axis_deg"] < 10.0

    assert not np.allclose(ds.world_T_camera[0], np.eye(4), atol=1e-3)

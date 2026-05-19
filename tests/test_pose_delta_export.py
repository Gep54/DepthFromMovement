"""Per-pair pose_delta.png / pose_delta.json in export output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.dataset import load_dataset
from viz.pose_delta import POSE_DELTA_JSON, POSE_DELTA_PNG, compute_pair_pose_delta
from viz.step_runner import export_all_stages, ensure_pair_pose_delta_exists


def test_compute_pair_pose_delta_baseline(mini_dataset_dir: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    s = compute_pair_pose_delta(ds.world_T_camera[0], ds.world_T_camera[1], frame_i=0, frame_j=1)
    assert s["frame_i"] == 0 and s["frame_j"] == 1
    assert s["baseline_m"] == pytest.approx(0.15, abs=1e-6)
    assert np.linalg.norm(np.asarray(s["translation_world_m"])) == pytest.approx(0.15, abs=1e-6)


def test_export_writes_pose_delta_files(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run"
    export_all_stages(ds, run_dir, i=0, j=1)
    png, js = ensure_pair_pose_delta_exists(run_dir)
    assert png.name == POSE_DELTA_PNG
    assert js.name == POSE_DELTA_JSON
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["baseline_m"] == pytest.approx(0.15, abs=1e-6)

from __future__ import annotations

from pathlib import Path

from data.dataset import load_dataset
from viz.recorder import STEP_ORDER
from viz.step_runner import (
    ensure_all_step_pngs_exist,
    ensure_sequence_outputs_exist,
    export_all_stages,
    export_sequence_consecutive_pairs,
)


def test_export_all_step_pngs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run1"
    export_all_stages(ds, run_dir, i=0, j=1, motion_mode="known_pose")
    paths = ensure_all_step_pngs_exist(run_dir)
    assert len(paths) == len(STEP_ORDER)

    export_all_stages(ds, tmp_path / "run2", i=0, j=1, motion_mode="estimate_essential")
    ensure_all_step_pngs_exist(tmp_path / "run2")


def test_export_sequence_consecutive_pairs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_root = tmp_path / "seq_run"
    export_sequence_consecutive_pairs(ds, run_root, motion_mode="known_pose")
    ensure_sequence_outputs_exist(run_root, len(ds.image_paths))

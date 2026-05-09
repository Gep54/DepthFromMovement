from __future__ import annotations

from pathlib import Path

from data.dataset import load_dataset
from viz.recorder import STEP_ORDER
from viz.step_runner import (
    ensure_all_step_pngs_exist,
    ensure_sequence_outputs_exist,
    export_all_stages,
    export_sequence_consecutive_pairs,
    iter_sequence_pairs,
)


def test_iter_sequence_pairs_counts() -> None:
    assert iter_sequence_pairs(3, 1) == [(0, 1), (1, 2)]
    assert iter_sequence_pairs(3, 10) == [(0, 1), (1, 2), (0, 2)]
    assert len(iter_sequence_pairs(5, 10)) == 10


def test_export_all_step_pngs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run1"
    export_all_stages(ds, run_dir, i=0, j=1, motion_confidence=1.0)
    paths = ensure_all_step_pngs_exist(run_dir)
    assert len(paths) == len(STEP_ORDER)

    for conf in (0.0, 0.5):
        export_all_stages(ds, tmp_path / f"run_{conf}", i=0, j=1, motion_confidence=conf)
        ensure_all_step_pngs_exist(tmp_path / f"run_{conf}")


def test_export_sequence_consecutive_pairs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_root = tmp_path / "seq_run"
    export_sequence_consecutive_pairs(ds, run_root, motion_confidence=1.0, pair_lookback=10)
    n = len(ds.image_paths)
    ensure_sequence_outputs_exist(run_root, n, pair_lookback=10)
    assert len(list((run_root / "pairs").iterdir())) == len(iter_sequence_pairs(n, 10))

    run_b = tmp_path / "seq_run_lb1"
    export_sequence_consecutive_pairs(ds, run_b, motion_confidence=1.0, pair_lookback=1)
    ensure_sequence_outputs_exist(run_b, n, pair_lookback=1)
    assert len(list((run_b / "pairs").iterdir())) == n - 1

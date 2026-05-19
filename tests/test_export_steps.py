from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.dataset import load_dataset
from viz.match_classification import classify_match_rejections
from viz.recorder import (
    DETAIL_PAIR_STEP_ORDER,
    GEOMETRY_STEP_ORDER,
    MINIMAL_GEOMETRY_STEP_ORDER,
    MINIMAL_PAIR_STEP_ORDER,
    PAIR_STEP_ORDER,
    SINGLE_STEP_ORDER,
    ensure_step_pngs_exist,
)
from viz.step_runner import (
    export_all_stages,
    export_sequence_consecutive_pairs,
    iter_sequence_pairs,
)


def test_iter_sequence_pairs_counts() -> None:
    assert iter_sequence_pairs(3, 1) == [(0, 1), (1, 2)]
    assert iter_sequence_pairs(3, 10) == [(0, 1), (1, 2), (0, 2)]
    assert len(iter_sequence_pairs(5, 10)) == 10


def test_export_minimal_step_pngs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run_minimal"
    export_all_stages(ds, run_dir, i=0, j=1)
    paths = ensure_step_pngs_exist(run_dir, include_geometry=True)
    expected = len(MINIMAL_PAIR_STEP_ORDER) + len(MINIMAL_GEOMETRY_STEP_ORDER) + 2  # pose_delta png+json
    assert len(paths) == expected

    audit = run_dir / "rejection_audit.jsonl"
    assert audit.is_file()

    export_all_stages(ds, tmp_path / "run_no_geom", i=0, j=1, include_geometry=False)
    ensure_step_pngs_exist(tmp_path / "run_no_geom", include_geometry=False)


def test_export_detail_log_adds_rejection_panels(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run_detail"
    export_all_stages(ds, run_dir, i=0, j=1, detail_log=True)
    paths = ensure_step_pngs_exist(run_dir, include_geometry=True, detail_log=True)
    expected = len(MINIMAL_PAIR_STEP_ORDER) + len(DETAIL_PAIR_STEP_ORDER) + len(MINIMAL_GEOMETRY_STEP_ORDER) + 2
    assert len(paths) == expected
    for slug in DETAIL_PAIR_STEP_ORDER:
        assert any(slug in str(p) for p in paths)


def test_export_full_steps(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run_full"
    export_all_stages(ds, run_dir, i=0, j=1, full_steps=True)
    paths = ensure_step_pngs_exist(run_dir, include_geometry=True, full_steps=True)
    expected = len(SINGLE_STEP_ORDER) + len(PAIR_STEP_ORDER) + len(GEOMETRY_STEP_ORDER) + 2
    assert len(paths) == expected


def test_no_cheiral_keeps_more_inliers_than_default(mini_dataset_dir: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    from pipeline.map import IncrementalMap, MapConfig
    import cv2

    g_i = cv2.imread(str(ds.image_paths[0]), cv2.IMREAD_GRAYSCALE)
    g_j = cv2.imread(str(ds.image_paths[1]), cv2.IMREAD_GRAYSCALE)
    K = ds.calibration.K
    wt = ds.world_T_camera

    m_on = IncrementalMap(cfg=MapConfig(check_cheiral=True), feat_cfg=ds.feature_config, K=K, world_T_camera=wt)
    tw_on = m_on.add_frame_pair(0, 1, g_i, g_j)
    m_off = IncrementalMap(cfg=MapConfig(check_cheiral=False), feat_cfg=ds.feature_config, K=K, world_T_camera=wt)
    tw_off = m_off.add_frame_pair(0, 1, g_i, g_j)

    cls_on = classify_match_rejections(tw_on, check_cheiral=True)
    cls_off = classify_match_rejections(tw_off, check_cheiral=False)
    assert int(cls_off.cheiral.sum()) == 0
    assert int(cls_off.inlier.sum()) >= int(cls_on.inlier.sum())
    if tw_on.X_world_h.shape[1] > 0:
        assert tw_off.cheiral_mask.all()
        assert np.sum(~np.isfinite(tw_on.X_world_h[:3, :])) >= 0


def test_classify_match_rejections_exclusive(mini_dataset_dir: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    from pipeline.map import IncrementalMap, MapConfig

    bgr_i_path = ds.image_paths[0]
    bgr_j_path = ds.image_paths[1]
    import cv2

    g_i = cv2.imread(str(bgr_i_path), cv2.IMREAD_GRAYSCALE)
    g_j = cv2.imread(str(bgr_j_path), cv2.IMREAD_GRAYSCALE)
    m = IncrementalMap(
        cfg=MapConfig(),
        feat_cfg=ds.feature_config,
        K=ds.calibration.K,
        world_T_camera=ds.world_T_camera,
    )
    tw = m.add_frame_pair(0, 1, g_i, g_j)
    cls = classify_match_rejections(tw)
    n = len(tw.pts1)
    if n > 0:
        total = cls.epipolar | cls.cheiral | cls.inlier
        assert total.sum() == n
        assert not (cls.epipolar & cls.cheiral).any()
        assert not (cls.inlier & cls.cheiral).any()


def test_export_epipolar_pdfs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_dir = tmp_path / "run_epi"
    export_all_stages(ds, run_dir, i=0, j=1, export_epipolar=True)
    epi_dir = run_dir / "epipolar"
    for name in (
        "all_pairs_epilines.pdf",
        "worst_epipolar_5.pdf",
        "best_epipolar_5.pdf",
    ):
        p = epi_dir / name
        assert p.is_file(), f"missing {p}"
        assert p.stat().st_size > 100


def test_export_epipolar_pdfs_sequence(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_root = tmp_path / "seq_epi"
    export_sequence_consecutive_pairs(ds, run_root, pair_lookback=10, export_epipolar=True)
    epi_dir = run_root / "epipolar"
    assert (epi_dir / "all_pairs_epilines.pdf").is_file()
    assert (epi_dir / "worst_epipolar_5.pdf").is_file()
    assert (epi_dir / "best_epipolar_5.pdf").is_file()


def test_export_sequence_consecutive_pairs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_root = tmp_path / "seq_run"
    export_sequence_consecutive_pairs(ds, run_root, pair_lookback=10)
    n = len(ds.image_paths)
    from viz.step_runner import ensure_sequence_outputs_exist

    ensure_sequence_outputs_exist(run_root, n, pair_lookback=10)
    assert len(list((run_root / "pairs").iterdir())) == len(iter_sequence_pairs(n, 10))
    pair0 = run_root / "pairs" / "000_001"
    assert (pair0 / "pose_delta.png").is_file()
    assert (pair0 / "pose_delta.json").is_file()
    assert (run_root / "rejection_audit.jsonl").is_file()

    run_b = tmp_path / "seq_run_lb1"
    export_sequence_consecutive_pairs(ds, run_b, pair_lookback=1)
    ensure_sequence_outputs_exist(run_b, n, pair_lookback=1)
    assert len(list((run_b / "pairs").iterdir())) == n - 1

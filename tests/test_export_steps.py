from __future__ import annotations

import json
from pathlib import Path

from data.dataset import load_dataset
from viz.match_classification import classify_match_rejections
from viz.recorder import GEOMETRY_STEP_ORDER, PAIR_STEP_ORDER, SINGLE_STEP_ORDER, ensure_step_pngs_exist
from viz.step_runner import (
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
    export_all_stages(ds, run_dir, i=0, j=1)
    paths = ensure_step_pngs_exist(run_dir, include_geometry=True)
    expected = len(SINGLE_STEP_ORDER) + len(PAIR_STEP_ORDER) + len(GEOMETRY_STEP_ORDER)
    assert len(paths) == expected

    audit = run_dir / "rejection_audit.jsonl"
    assert audit.is_file()
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["i"] == 0 and rec["j"] == 1
    assert "counts" in rec

    export_all_stages(ds, tmp_path / "run_no_geom", i=0, j=1, include_geometry=False)
    ensure_step_pngs_exist(tmp_path / "run_no_geom", include_geometry=False)


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

    cls_on = classify_match_rejections(
        tw_on, K, wt[0], wt[1], check_cheiral=True
    )
    cls_off = classify_match_rejections(
        tw_off, K, wt[0], wt[1], check_cheiral=False
    )
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
    cls = classify_match_rejections(
        tw, ds.calibration.K, ds.world_T_camera[0], ds.world_T_camera[1], reproj_thresh_px=3.0
    )
    n = len(tw.pts1)
    if n > 0:
        total = cls.epipolar | cls.cheiral | cls.reproj | cls.inlier
        assert total.sum() == n
        assert not (cls.epipolar & cls.cheiral).any()
        assert not (cls.inlier & cls.reproj).any()


def test_export_sequence_consecutive_pairs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    run_root = tmp_path / "seq_run"
    export_sequence_consecutive_pairs(ds, run_root, pair_lookback=10)
    n = len(ds.image_paths)
    from viz.step_runner import ensure_sequence_outputs_exist

    ensure_sequence_outputs_exist(run_root, n, pair_lookback=10)
    assert len(list((run_root / "pairs").iterdir())) == len(iter_sequence_pairs(n, 10))

    pairs_json = run_root / "summary" / "pairs_all_rejection_types.json"
    data = json.loads(pairs_json.read_text(encoding="utf-8"))
    assert "pairs" in data

    run_b = tmp_path / "seq_run_lb1"
    export_sequence_consecutive_pairs(ds, run_b, pair_lookback=1)
    ensure_sequence_outputs_exist(run_b, n, pair_lookback=1)
    assert len(list((run_b / "pairs").iterdir())) == n - 1

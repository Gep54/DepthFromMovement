from __future__ import annotations

from pathlib import Path

import numpy as np

from data.dataset import load_dataset, read_image_bgr
from pipeline.features import compute_frame_features_cache
from pipeline.map import IncrementalMap, MapConfig
from viz.step_runner import _undistort_if_needed


def test_compute_frame_features_cache_length_matches_sequence(mini_dataset_dir: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    grays: list[np.ndarray] = []
    for path in ds.image_paths:
        bgr = read_image_bgr(path)
        _, g = _undistort_if_needed(bgr, ds)
        grays.append(g)
    cache = compute_frame_features_cache(grays, ds.feature_config)
    assert len(cache) == len(ds.image_paths)
    assert all(isinstance(c.keypoints, (list, tuple)) for c in cache)


def test_add_frame_pair_with_cache_matches_without_cache(mini_dataset_dir: Path) -> None:
    """Same geometry when descriptors are cached vs recomputed (deterministic given same grays)."""
    ds = load_dataset(mini_dataset_dir)
    grays: list[np.ndarray] = []
    for path in ds.image_paths:
        bgr = read_image_bgr(path)
        _, g = _undistort_if_needed(bgr, ds)
        grays.append(g)
    fc = ds.feature_config
    cache = compute_frame_features_cache(grays, fc)
    cfg = MapConfig()
    K = ds.calibration.K
    W = ds.world_T_camera

    tw_cached = IncrementalMap(cfg=cfg, feat_cfg=fc, K=K, world_T_camera=W).add_frame_pair(
        0,
        1,
        grays[0],
        grays[1],
        features_i=cache[0],
        features_j=cache[1],
    )
    tw_fresh = IncrementalMap(cfg=cfg, feat_cfg=fc, K=K, world_T_camera=W).add_frame_pair(
        0,
        1,
        grays[0],
        grays[1],
    )
    assert tw_cached.pts1.shape == tw_fresh.pts1.shape
    assert np.allclose(tw_cached.pts1, tw_fresh.pts1)
    assert np.allclose(tw_cached.pts2, tw_fresh.pts2)

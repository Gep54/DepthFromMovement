"""Two-view triangulation uses R,t from provided world poses (motion.json), not recoverPose."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from data.dataset import load_dataset
from pipeline import map as map_mod
from pipeline.geometry import relative_motion_from_world_poses
from pipeline.map import IncrementalMap, MapConfig


def test_add_frame_pair_pose_from_motion_json(
    mini_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _synthetic_matches(*_args, **_kwargs) -> tuple[np.ndarray, np.ndarray, list]:
        pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0],
                        [50.0, 50.0], [60.0, 60.0], [70.0, 70.0], [80.0, 80.0],
                        [90.0, 90.0], [100.0, 100.0]], dtype=np.float32)
        return pts, pts + np.array([2.0, 0.0], dtype=np.float32), []

    def _all_inliers(*_args, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
        n = _args[0].shape[0] if _args else 0
        return np.eye(3, dtype=np.float64), np.ones((n, 1), np.uint8)

    monkeypatch.setattr(map_mod, "match_pair_points", _synthetic_matches)
    monkeypatch.setattr(map_mod, "epipolar_inlier_mask_from_motion", _all_inliers)

    ds = load_dataset(mini_dataset_dir)
    K = ds.calibration.K
    W = ds.world_T_camera
    g0 = cv2.imread(str(ds.image_paths[0]), cv2.IMREAD_GRAYSCALE)
    g1 = cv2.imread(str(ds.image_paths[1]), cv2.IMREAD_GRAYSCALE)
    R_exp, t_exp = relative_motion_from_world_poses(W[0], W[1])
    tw = IncrementalMap(
        cfg=MapConfig(),
        feat_cfg=ds.feature_config,
        K=K,
        world_T_camera=W,
    ).add_frame_pair(0, 1, g0, g1)
    assert tw.scale_ok
    assert tw.R_est is not None and tw.t_est is not None
    assert np.allclose(tw.R_est, R_exp)
    assert np.allclose(tw.t_est, t_exp)
    assert abs(tw.scale - float(np.linalg.norm(t_exp))) < 1e-9

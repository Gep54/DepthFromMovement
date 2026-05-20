"""Two-view triangulation motion source: vision_scale vs odometry_pose."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from data.dataset import load_dataset
from pipeline import map as map_mod
from pipeline.frame_axes import world_T_body_to_world_T_opencv_cam
from pipeline.geometry import relative_motion_from_world_poses
from pipeline.map import IncrementalMap, MapConfig


def _synthetic_pts() -> tuple[np.ndarray, np.ndarray, list]:
    pts = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
            [50.0, 50.0],
            [60.0, 60.0],
            [70.0, 70.0],
            [80.0, 80.0],
            [90.0, 90.0],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )
    return pts, pts + np.array([2.0, 0.0], dtype=np.float32), []


def test_odometry_pose_uses_motion_R_t(
    mini_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _all_inliers(*_args, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
        n = _args[0].shape[0] if _args else 0
        return np.eye(3, dtype=np.float64), np.ones((n, 1), np.uint8)

    monkeypatch.setattr(map_mod, "match_pair_points", lambda *_a, **_k: _synthetic_pts())
    monkeypatch.setattr(map_mod, "epipolar_inlier_mask_from_motion", _all_inliers)

    ds = load_dataset(mini_dataset_dir)
    W = ds.world_T_camera
    g0 = cv2.imread(str(ds.image_paths[0]), cv2.IMREAD_GRAYSCALE)
    g1 = cv2.imread(str(ds.image_paths[1]), cv2.IMREAD_GRAYSCALE)
    R_exp, t_exp = relative_motion_from_world_poses(
        world_T_body_to_world_T_opencv_cam(W[0]),
        world_T_body_to_world_T_opencv_cam(W[1]),
    )
    tw = IncrementalMap(
        cfg=MapConfig(triangulation_motion_source="odometry_pose"),
        feat_cfg=ds.feature_config,
        K=ds.calibration.K,
        world_T_camera=W,
    ).add_frame_pair(0, 1, g0, g1)
    assert tw.scale_ok
    assert tw.R_est is not None and tw.t_est is not None
    assert np.allclose(tw.R_est, R_exp)
    assert np.allclose(tw.t_est, t_exp)


def test_vision_scale_uses_recovered_rotation(
    mini_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    R_fake = np.array(
        [
            [0.996, -0.087, 0.0],
            [0.087, 0.996, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t_fake = np.array([[0.05], [0.0], [0.0]], dtype=np.float64)

    def _fake_recover(*_args, **_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return R_fake, t_fake, np.ones((10, 1), dtype=np.uint8)

    def _fake_essential(*_args, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
        return np.eye(3, dtype=np.float64), np.ones((10, 1), dtype=np.uint8)

    monkeypatch.setattr(map_mod, "match_pair_points", lambda *_a, **_k: _synthetic_pts())
    monkeypatch.setattr(map_mod.cv2, "findEssentialMat", _fake_essential)
    monkeypatch.setattr(map_mod, "recover_pose_from_essential", _fake_recover)

    ds = load_dataset(mini_dataset_dir)
    W = ds.world_T_camera
    g0 = cv2.imread(str(ds.image_paths[0]), cv2.IMREAD_GRAYSCALE)
    g1 = cv2.imread(str(ds.image_paths[1]), cv2.IMREAD_GRAYSCALE)
    R_motion, _ = relative_motion_from_world_poses(W[0], W[1])
    tw = IncrementalMap(
        cfg=MapConfig(triangulation_motion_source="vision_scale"),
        feat_cfg=ds.feature_config,
        K=ds.calibration.K,
        world_T_camera=W,
    ).add_frame_pair(0, 1, g0, g1)
    assert tw.scale_ok
    assert tw.R_est is not None
    assert np.allclose(tw.R_est, R_fake)
    assert not np.allclose(tw.R_est, R_motion)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.descriptor_map_json import load_descriptor_map_json
from data.dataset import load_dataset
from pipeline.descriptor_landmark_map import (
    DescriptorLandmarkMap,
    DescriptorMapConfig,
    descriptor_distance,
    export_landmarks_csv,
    point_camera_to_drone_to_world,
    within_merge_sphere,
    world_point_to_camera_frame,
)
from pipeline.map import TwoViewResult


def _identity_poses() -> tuple[np.ndarray, np.ndarray]:
    W = np.eye(4, dtype=np.float64)
    return W, W


def _tw_from_z(z: float, descriptor: np.ndarray, *, frame_i: int = 0) -> TwoViewResult:
    X_cam = np.zeros((4, 1), dtype=np.float64)
    X_cam[:3, 0] = [0.0, 0.0, z]
    X_cam[3, 0] = 1.0
    X_world = X_cam.copy()
    return TwoViewResult(
        frame_i=frame_i,
        frame_j=frame_i + 1,
        pts1=np.zeros((1, 2), np.float32),
        pts2=np.zeros((1, 2), np.float32),
        inlier_mask=np.ones((1, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=True,
        X_world_h=X_world,
        X_cam_h=X_cam,
        cheiral_mask=np.array([True]),
        reproj={},
        descriptors=descriptor.reshape(1, -1).copy(),
    )


def test_world_point_to_camera_frame_identity() -> None:
    W = np.eye(4, dtype=np.float64)
    Xw = np.array([1.0, -2.0, 5.0])
    np.testing.assert_allclose(world_point_to_camera_frame(Xw, W), Xw)


def test_world_point_to_camera_frame_translation() -> None:
    W = np.eye(4, dtype=np.float64)
    W[:3, 3] = [1.0, 0.0, 0.0]
    np.testing.assert_allclose(world_point_to_camera_frame(np.zeros(3), W), [-1.0, 0.0, 0.0])


def test_ema_default_matches_incremental_mean() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    d = np.zeros((1, 32), dtype=np.uint8)
    m.integrate(_tw_from_z(2.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    m.integrate(_tw_from_z(8.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    np.testing.assert_allclose(m.landmarks[0].position_world[2], 5.0)


def test_fixed_merge_beta_differs_from_mean() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=0.25, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    d = np.ones((1, 32), dtype=np.uint8)
    m.integrate(_tw_from_z(2.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    m.integrate(_tw_from_z(8.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    np.testing.assert_allclose(m.landmarks[0].position_world[2], 3.5)


def test_replace_if_better_updates_prototype() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=512.0)
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    proto = np.zeros((1, 32), dtype=np.uint8)
    m.integrate(_tw_from_z(2.0, proto), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    stored = m.landmarks[0].descriptor.copy()
    alt = np.ones((1, 32), dtype=np.uint8)
    m.integrate(
        _tw_from_z(3.0, alt, frame_i=1),
        world_T_camera_raw=W_cam,
        world_T_drone_raw=W_drone,
        world_T_camera_j_raw=W_cam,
    )
    assert np.any(stored != m.landmarks[0].descriptor)


def test_descriptor_distance_orb() -> None:
    a = np.zeros((32,), dtype=np.uint8)
    b = a.copy()
    assert descriptor_distance(a, b, "ORB") == 0.0


def test_within_merge_sphere() -> None:
    assert within_merge_sphere(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), 1.0)
    assert not within_merge_sphere(np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), 1.0)


def test_spatial_gate_rejects_distant_descriptor_match() -> None:
    cfg = DescriptorMapConfig(
        method="ORB",
        merge_beta=None,
        max_match_distance=256.0,
        spatial_merge_radius_m=1.0,
    )
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    d = np.zeros((32,), dtype=np.uint8)
    m.integrate(_tw_single_point(2.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    m.integrate(_tw_single_point(10.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    assert len(m.landmarks) == 2


def test_spatial_gate_allows_near_descriptor_match() -> None:
    cfg = DescriptorMapConfig(
        method="ORB",
        merge_beta=None,
        max_match_distance=256.0,
        spatial_merge_radius_m=1.0,
    )
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    d = np.zeros((32,), dtype=np.uint8)
    m.integrate(_tw_single_point(2.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    m.integrate(_tw_single_point(2.5, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    assert len(m.landmarks) == 1
    np.testing.assert_allclose(m.landmarks[0].position_world[2], 2.25)


def _tw_single_point(z: float, descriptor: np.ndarray) -> TwoViewResult:
    return _tw_from_z(z, descriptor)


def test_load_descriptor_map_json_defaults(tmp_path: Path) -> None:
    cfg = load_descriptor_map_json(tmp_path / "missing.json", "ORB")
    assert cfg.merge_beta is None
    assert cfg.max_match_distance == 64.0
    assert cfg.spatial_merge_radius_m is None


def test_load_descriptor_map_json_spatial_radius(tmp_path: Path) -> None:
    path = tmp_path / "descriptor_map.json"
    path.write_text('{"spatial_merge_radius_m": 0.5}', encoding="utf-8")
    cfg = load_descriptor_map_json(path, "ORB")
    assert cfg.spatial_merge_radius_m == 0.5


def test_integrate_max_range_world_skips_far_points() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    d = np.zeros((2, 32), dtype=np.uint8)

    def tw_points(z_near: float, z_far: float) -> TwoViewResult:
        X_cam = np.zeros((4, 2), dtype=np.float64)
        X_cam[:3, 0] = [0.0, 0.0, z_near]
        X_cam[:3, 1] = [0.0, 0.0, z_far]
        X_cam[3, :] = 1.0
        return TwoViewResult(
            frame_i=0,
            frame_j=1,
            pts1=np.zeros((2, 2), np.float32),
            pts2=np.zeros((2, 2), np.float32),
            inlier_mask=np.ones((2, 1), np.uint8),
            E=None,
            R_est=None,
            t_est=None,
            scale=1.0,
            scale_ok=True,
            X_world_h=X_cam.copy(),
            X_cam_h=X_cam,
            cheiral_mask=np.array([True, True]),
            reproj={},
            descriptors=d.copy(),
        )

    m.integrate(
        tw_points(2.0, 200.0),
        world_T_camera_raw=W_cam,
        world_T_drone_raw=W_drone,
        world_T_camera_j_raw=W_cam,
        max_range_world=50.0,
    )
    assert len(m.landmarks) == 1
    np.testing.assert_allclose(m.landmarks[0].position_world[2], 2.0)


def test_integrate_max_range_world_uses_frame_j_anchor() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W_cam_i, W_drone = _identity_poses()
    W_cam_j = np.eye(4, dtype=np.float64)
    W_cam_j[:3, 3] = [100.0, 0.0, 0.0]
    d = np.zeros((1, 32), dtype=np.uint8)
    # World point (2, 0, 0): close to camera i, far from camera j at x=100.
    tw_near_i = _tw_from_z(2.0, d)
    tw_near_i.X_cam_h[:3, 0] = [2.0, 0.0, 0.0]
    tw_near_i.X_world_h = tw_near_i.X_cam_h.copy()
    m.integrate(
        tw_near_i,
        world_T_camera_raw=W_cam_i,
        world_T_drone_raw=W_drone,
        world_T_camera_j_raw=W_cam_j,
        max_range_world=10.0,
    )
    assert len(m.landmarks) == 0
    # World point (101, 0, 0): ~1 m from camera j.
    tw_near_j = _tw_from_z(2.0, d)
    tw_near_j.X_cam_h[:3, 0] = [101.0, 0.0, 0.0]
    tw_near_j.X_world_h = tw_near_j.X_cam_h.copy()
    m.integrate(
        tw_near_j,
        world_T_camera_raw=W_cam_i,
        world_T_drone_raw=W_drone,
        world_T_camera_j_raw=W_cam_j,
        max_range_world=10.0,
    )
    assert len(m.landmarks) == 1
    np.testing.assert_allclose(m.landmarks[0].position_world[0], 101.0)


def test_point_camera_to_drone_to_world_with_offset_camera() -> None:
    W_drone = np.eye(4, dtype=np.float64)
    W_cam = np.eye(4, dtype=np.float64)
    W_cam[:3, 3] = [0.0, 0.0, 5.0]
    X_cam = np.array([0.0, 0.0, 2.0])
    Xw = point_camera_to_drone_to_world(X_cam, W_cam, W_drone)
    np.testing.assert_allclose(Xw, [0.0, 0.0, 7.0])


def test_export_landmarks_csv_roundtrip(tmp_path: Path) -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W_cam, W_drone = _identity_poses()
    d = np.full((1, 32), 7, dtype=np.uint8)
    m.integrate(_tw_from_z(3.0, d), world_T_camera_raw=W_cam, world_T_drone_raw=W_drone, world_T_camera_j_raw=W_cam)
    outp = tmp_path / "lm.csv"
    export_landmarks_csv(outp, m)
    text = outp.read_text(encoding="utf-8")
    assert "id,x_world,y_world,z_world,n_updates,descriptor_hex" in text
    assert str(m.landmarks[0].id) in text


def test_mini_dataset_descriptor_pipeline_runs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    try:
        import viz.match_classification  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("viz.match_classification not in tree (step_runner dependency)")
    ds = load_dataset(mini_dataset_dir)
    cfg = DescriptorMapConfig.defaults(ds.feature_config.method)
    cfg = DescriptorMapConfig(
        method=cfg.method,
        merge_beta=None,
        max_match_distance=500.0 if cfg.method == "ORB" else 800.0,
        ratio_second_best=None,
    )
    import importlib.util

    runner_path = Path(__file__).resolve().parents[1] / "viz" / "descriptor_map_runner.py"
    spec = importlib.util.spec_from_file_location("descriptor_map_runner_mod", runner_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    desc_map = runner.run_descriptor_landmark_pipeline(
        ds,
        tmp_path / "out",
        pair_lookback=10,
        desc_cfg=cfg,
        save_iter_viz=False,
    )
    assert isinstance(desc_map.landmarks, list)
    export_landmarks_csv(tmp_path / "lm.csv", desc_map)
    assert (tmp_path / "lm.csv").is_file()

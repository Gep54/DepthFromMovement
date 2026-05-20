from __future__ import annotations

from pathlib import Path

import numpy as np

from data.descriptor_map_json import load_descriptor_map_json
from data.dataset import load_dataset
from pipeline.descriptor_landmark_map import (
    DescriptorLandmarkMap,
    DescriptorMapConfig,
    descriptor_distance,
    export_landmarks_csv,
    within_merge_sphere,
    world_point_to_cam0,
)
from pipeline.map import TwoViewResult


def test_world_point_to_cam0_identity() -> None:
    W0 = np.eye(4, dtype=np.float64)
    Xw = np.array([1.0, -2.0, 5.0])
    np.testing.assert_allclose(world_point_to_cam0(Xw, W0), Xw)


def test_world_point_to_cam0_translation() -> None:
    """Camera 0 one unit ahead in world X → world point at origin is at (-1,0,0) in cam0."""
    W0 = np.eye(4, dtype=np.float64)
    W0[:3, 3] = [1.0, 0.0, 0.0]
    np.testing.assert_allclose(world_point_to_cam0(np.zeros(3), W0), [-1.0, 0.0, 0.0])


def test_ema_default_matches_incremental_mean() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W0 = np.eye(4, dtype=np.float64)
    d = np.zeros((1, 32), dtype=np.uint8)

    def tw_from_z(z: float) -> TwoViewResult:
        X = np.zeros((4, 1), dtype=np.float64)
        X[:3, 0] = [0.0, 0.0, z]
        X[3, 0] = 1.0
        return TwoViewResult(
            frame_i=0,
            frame_j=1,
            pts1=np.zeros((1, 2), np.float32),
            pts2=np.zeros((1, 2), np.float32),
            inlier_mask=np.ones((1, 1), np.uint8),
            E=None,
            R_est=None,
            t_est=None,
            scale=1.0,
            scale_ok=True,
            X_world_h=X,
            cheiral_mask=np.array([True]),
            reproj={},
            descriptors=d.copy(),
        )

    W1 = W0
    m.integrate(tw_from_z(2.0), W0, W1)
    m.integrate(tw_from_z(8.0), W0, W1)
    np.testing.assert_allclose(m.landmarks[0].position_cam0[0], 5.0)


def test_fixed_merge_beta_differs_from_mean() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=0.25, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W0 = np.eye(4, dtype=np.float64)
    d = np.ones((1, 32), dtype=np.uint8)

    def tw(z: float) -> TwoViewResult:
        X = np.zeros((4, 1), dtype=np.float64)
        X[:3, 0] = [0.0, 0.0, z]
        X[3, 0] = 1.0
        return TwoViewResult(
            frame_i=0,
            frame_j=1,
            pts1=np.zeros((1, 2), np.float32),
            pts2=np.zeros((1, 2), np.float32),
            inlier_mask=np.ones((1, 1), np.uint8),
            E=None,
            R_est=None,
            t_est=None,
            scale=1.0,
            scale_ok=True,
            X_world_h=X,
            cheiral_mask=np.array([True]),
            reproj={},
            descriptors=d.copy(),
        )

    W1 = W0
    m.integrate(tw(2.0), W0, W1)
    m.integrate(tw(8.0), W0, W1)
    # Mean would be 5.0; beta=0.25 gives 0.75*2 + 0.25*8 = 3.5
    np.testing.assert_allclose(m.landmarks[0].position_cam0[0], 3.5)


def test_replace_if_better_updates_prototype() -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=512.0)
    m = DescriptorLandmarkMap(cfg)
    W0 = np.eye(4, dtype=np.float64)
    proto = np.zeros((1, 32), dtype=np.uint8)
    proto_obs = np.zeros((4, 1), dtype=np.float64)
    proto_obs[:3, 0] = [0.0, 0.0, 2.0]
    proto_obs[3, 0] = 1.0
    tw1 = TwoViewResult(
        frame_i=0,
        frame_j=1,
        pts1=np.zeros((1, 2), np.float32),
        pts2=np.zeros((1, 2), np.float32),
        inlier_mask=np.ones((1, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=True,
        X_world_h=proto_obs,
        cheiral_mask=np.array([True]),
        reproj={},
        descriptors=proto.copy(),
    )
    m.integrate(tw1, W0, W0)
    stored = m.landmarks[0].descriptor.copy()

    alt = np.ones((1, 32), dtype=np.uint8)
    alt_obs = proto_obs.copy()
    alt_obs[:3, 0] = [0.0, 0.0, 3.0]
    tw2 = TwoViewResult(
        frame_i=1,
        frame_j=2,
        pts1=np.zeros((1, 2), np.float32),
        pts2=np.zeros((1, 2), np.float32),
        inlier_mask=np.ones((1, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=True,
        X_world_h=alt_obs,
        cheiral_mask=np.array([True]),
        reproj={},
        descriptors=alt.copy(),
    )
    m.integrate(tw2, W0, W0)
    assert np.any(stored != m.landmarks[0].descriptor)


def test_descriptor_distance_orb() -> None:
    a = np.zeros((32,), dtype=np.uint8)
    b = a.copy()
    assert descriptor_distance(a, b, "ORB") == 0.0


def test_within_merge_sphere() -> None:
    assert within_merge_sphere(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), 1.0)
    assert not within_merge_sphere(np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), 1.0)


def _tw_single_point(z: float, descriptor: np.ndarray) -> TwoViewResult:
    X = np.zeros((4, 1), dtype=np.float64)
    X[:3, 0] = [0.0, 0.0, z]
    X[3, 0] = 1.0
    return TwoViewResult(
        frame_i=0,
        frame_j=1,
        pts1=np.zeros((1, 2), np.float32),
        pts2=np.zeros((1, 2), np.float32),
        inlier_mask=np.ones((1, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=True,
        X_world_h=X,
        cheiral_mask=np.array([True]),
        reproj={},
        descriptors=descriptor.reshape(1, -1).copy(),
    )


def test_spatial_gate_rejects_distant_descriptor_match() -> None:
    cfg = DescriptorMapConfig(
        method="ORB",
        merge_beta=None,
        max_match_distance=256.0,
        spatial_merge_radius_m=1.0,
    )
    m = DescriptorLandmarkMap(cfg)
    W0 = np.eye(4, dtype=np.float64)
    d = np.zeros((32,), dtype=np.uint8)
    W1 = W0
    m.integrate(_tw_single_point(2.0, d), W0, W1)
    m.integrate(_tw_single_point(10.0, d), W0, W1)
    assert len(m.landmarks) == 2


def test_spatial_gate_allows_near_descriptor_match() -> None:
    cfg = DescriptorMapConfig(
        method="ORB",
        merge_beta=None,
        max_match_distance=256.0,
        spatial_merge_radius_m=1.0,
    )
    m = DescriptorLandmarkMap(cfg)
    W0 = np.eye(4, dtype=np.float64)
    d = np.zeros((32,), dtype=np.uint8)
    W1 = W0
    m.integrate(_tw_single_point(2.0, d), W0, W1)
    m.integrate(_tw_single_point(2.5, d), W0, W1)
    assert len(m.landmarks) == 1
    np.testing.assert_allclose(m.landmarks[0].position_cam0[0], 2.25)


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


def test_export_landmarks_csv_roundtrip(tmp_path: Path) -> None:
    cfg = DescriptorMapConfig(method="ORB", merge_beta=None, max_match_distance=256.0)
    m = DescriptorLandmarkMap(cfg)
    W0 = np.eye(4, dtype=np.float64)
    d = np.full((1, 32), 7, dtype=np.uint8)
    X = np.zeros((4, 1), dtype=np.float64)
    X[:3, 0] = [1.0, 2.0, 3.0]
    X[3, 0] = 1.0
    tw = TwoViewResult(
        frame_i=0,
        frame_j=1,
        pts1=np.zeros((1, 2), np.float32),
        pts2=np.zeros((1, 2), np.float32),
        inlier_mask=np.ones((1, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=True,
        X_world_h=X,
        cheiral_mask=np.array([True]),
        reproj={},
        descriptors=d,
    )
    m.integrate(tw, W0, W0)
    outp = tmp_path / "lm.csv"
    export_landmarks_csv(outp, m)
    text = outp.read_text(encoding="utf-8")
    assert "id,x_cam0,y_cam0,z_cam0,n_updates,descriptor_hex" in text
    assert str(m.landmarks[0].id) in text


def test_mini_dataset_descriptor_pipeline_runs(mini_dataset_dir: Path, tmp_path: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    cfg = DescriptorMapConfig.defaults(ds.feature_config.method)
    cfg = DescriptorMapConfig(
        method=cfg.method,
        merge_beta=None,
        max_match_distance=500.0 if cfg.method == "ORB" else 800.0,
        ratio_second_best=None,
    )
    from viz.descriptor_map_runner import run_descriptor_landmark_pipeline

    desc_map = run_descriptor_landmark_pipeline(
        ds,
        tmp_path / "out",
        pair_lookback=10,
        desc_cfg=cfg,
        save_iter_viz=False,
    )
    assert isinstance(desc_map.landmarks, list)
    export_landmarks_csv(tmp_path / "lm.csv", desc_map)
    assert (tmp_path / "lm.csv").is_file()

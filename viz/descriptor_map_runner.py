from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data.dataset import Dataset, read_image_bgr
from pipeline.descriptor_landmark_map import (
    DescriptorLandmarkMap,
    DescriptorMapConfig,
    world_point_to_camera_frame,
)
from pipeline.features import compute_frame_features_cache
from pipeline.map import IncrementalMap, MapConfig
from viz.overlays import estimated_depth_visualization, project_points_topdown


def _undistort_if_needed(bgr: np.ndarray, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if ds.calibration.dist_coeffs is None or np.linalg.norm(ds.calibration.dist_coeffs) < 1e-9:
        return bgr, gray
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        ds.calibration.K,
        ds.calibration.dist_coeffs,
        (bgr.shape[1], bgr.shape[0]),
        alpha=0,
    )
    und = cv2.undistort(bgr, ds.calibration.K, ds.calibration.dist_coeffs, None, newK=new_K)
    g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    return und, g


def iter_sequence_pairs(n_frames: int, pair_lookback: int) -> list[tuple[int, int]]:
    """Indices ``j`` paired with earlier frames ``j-off`` for ``off`` in ``1..min(pair_lookback, j)``."""
    wl = max(1, int(pair_lookback))
    out: list[tuple[int, int]] = []
    for j in range(1, n_frames):
        for off in range(1, min(wl, j) + 1):
            i = j - off
            out.append((i, j))
    return out


def project_cam0_to_uv_z(K: np.ndarray, X_cam0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points in camera-0 frame to pixels and depths (positive Z forward)."""
    X = np.asarray(X_cam0, dtype=np.float64).reshape(-1, 3)
    if X.shape[0] == 0:
        return np.zeros((0, 2), np.float64), np.zeros(0, np.float64)
    z = X[:, 2]
    uv = np.zeros((X.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * (X[:, 0] / np.maximum(z, 1e-9)) + K[0, 2]
    uv[:, 1] = K[1, 1] * (X[:, 1] / np.maximum(z, 1e-9)) + K[1, 2]
    return uv, z


def run_descriptor_landmark_pipeline(
    ds: Dataset,
    run_dir: str | Path,
    *,
    pair_lookback: int = 10,
    desc_cfg: DescriptorMapConfig,
    save_iter_viz: bool = False,
) -> DescriptorLandmarkMap:
    """
    Multi-baseline pairing (same as sequence export), landmark fusion in dataset world frame.

    Offline datasets have no separate body frame: ``world_T_drone`` is set equal to
    ``world_T_camera`` at each frame (camera→world only).

    Writes ``run_dir/descriptor_map/landmarks_topdown.png``,
    ``run_dir/descriptor_map/sparse_depth_cam0.png``, and optional ``iter/*.png``.
    """
    root = Path(run_dir)
    dm_root = root / "descriptor_map"
    dm_root.mkdir(parents=True, exist_ok=True)
    iter_root = dm_root / "iter"
    if save_iter_viz:
        iter_root.mkdir(parents=True, exist_ok=True)

    n = len(ds.image_paths)
    if n < 2:
        raise ValueError(f"need at least 2 images, got {n}")

    grays: list[np.ndarray] = []
    for path in ds.image_paths:
        bgr = read_image_bgr(path)
        _, g = _undistort_if_needed(bgr, ds)
        grays.append(g)

    fc = ds.feature_config
    frame_cache = compute_frame_features_cache(grays, fc)

    map_cfg = MapConfig()
    inc = IncrementalMap(
        cfg=map_cfg,
        feat_cfg=fc,
        K=ds.calibration.K,
        world_T_camera=ds.world_T_camera,
        window=10**9,
    )
    desc_map = DescriptorLandmarkMap(desc_cfg)

    wl = max(1, int(pair_lookback))
    pairs = iter_sequence_pairs(n, wl)
    for step_idx, (i, j) in enumerate(pairs):
        tw = inc.add_frame_pair(
            i,
            j,
            grays[i],
            grays[j],
            features_i=frame_cache[i],
            features_j=frame_cache[j],
        )
        W_i = ds.world_T_camera[i]
        W_j = ds.world_T_camera[j]
        desc_map.integrate(
            tw,
            world_T_camera_raw=W_i,
            world_T_drone_raw=W_i,
            world_T_camera_j_raw=W_j,
            spatial_merge_radius_m=desc_cfg.spatial_merge_radius_m,
        )
        if save_iter_viz:
            _write_snapshot(iter_root, ds, desc_map, step_idx, i, j)

    _write_final_plots(dm_root, ds, desc_map)

    return desc_map


def _world_positions_to_cam0(ds: Dataset, xyz_world: np.ndarray) -> np.ndarray:
    W0 = ds.world_T_camera[0]
    if xyz_world.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(
        [world_point_to_camera_frame(xyz_world[k], W0) for k in range(xyz_world.shape[0])],
        axis=0,
    )


def _write_final_plots(dm_root: Path, ds: Dataset, desc_map: DescriptorLandmarkMap) -> None:
    K = ds.calibration.K
    xyz_world = desc_map.positions_world()
    top = project_points_topdown(xyz_world if xyz_world.shape[0] else np.zeros((0, 3)))
    cv2.imwrite(str(dm_root / "landmarks_topdown.png"), top)

    xyz_cam0 = _world_positions_to_cam0(ds, xyz_world)
    bgr0 = read_image_bgr(ds.image_paths[0])
    und0, _ = _undistort_if_needed(bgr0, ds)
    uv, z = project_cam0_to_uv_z(K, xyz_cam0)
    mask = np.isfinite(uv[:, 0]) & (z > 1e-9)
    panel = estimated_depth_visualization(und0, uv[mask], z[mask])
    cv2.imwrite(str(dm_root / "sparse_depth_cam0.png"), panel)


def _write_snapshot(
    iter_root: Path,
    ds: Dataset,
    desc_map: DescriptorLandmarkMap,
    step_idx: int,
    i: int,
    j: int,
) -> None:
    K = ds.calibration.K
    xyz_world = desc_map.positions_world()
    xyz_cam0 = _world_positions_to_cam0(ds, xyz_world)
    stem = f"{step_idx:04d}_{i:03d}_{j:03d}"
    top = project_points_topdown(xyz_world if xyz_world.shape[0] else np.zeros((0, 3)))
    cv2.imwrite(str(iter_root / f"landmarks_topdown_{stem}.png"), top)

    bgr0 = read_image_bgr(ds.image_paths[0])
    und0, _ = _undistort_if_needed(bgr0, ds)
    uv, z = project_cam0_to_uv_z(K, xyz_cam0)
    mask = np.isfinite(uv[:, 0]) & (z > 1e-9)
    panel = estimated_depth_visualization(und0, uv[mask], z[mask])
    cv2.imwrite(str(iter_root / f"sparse_depth_cam0_{stem}.png"), panel)

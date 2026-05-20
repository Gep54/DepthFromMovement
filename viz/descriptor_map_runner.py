from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data.dataset import Dataset, read_image_bgr
from pipeline.descriptor_landmark_map import DescriptorLandmarkMap, DescriptorMapConfig
from pipeline.features import compute_frame_features_cache
from pipeline.map import IncrementalMap, MapConfig
from viz.overlays import estimated_depth_visualization, project_points_topdown
from viz.step_runner import _undistort_if_needed, iter_sequence_pairs


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
    Multi-baseline pairing (same as sequence export), landmark fusion in camera-0 frame.

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
    W0 = ds.world_T_camera[0]

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
        desc_map.integrate(
            tw,
            W0,
            ds.world_T_camera[j],
            spatial_merge_radius_m=desc_cfg.spatial_merge_radius_m,
        )
        if save_iter_viz:
            _write_snapshot(iter_root, ds, desc_map, step_idx, i, j)

    _write_final_plots(dm_root, ds, desc_map)

    return desc_map


def _write_final_plots(dm_root: Path, ds: Dataset, desc_map: DescriptorLandmarkMap) -> None:
    K = ds.calibration.K
    xyz = desc_map.positions_cam0()
    top = project_points_topdown(xyz if xyz.shape[0] else np.zeros((0, 3)))
    cv2.imwrite(str(dm_root / "landmarks_topdown.png"), top)

    bgr0 = read_image_bgr(ds.image_paths[0])
    und0, _ = _undistort_if_needed(bgr0, ds)
    uv, z = project_cam0_to_uv_z(K, xyz)
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
    xyz = desc_map.positions_cam0()
    stem = f"{step_idx:04d}_{i:03d}_{j:03d}"
    top = project_points_topdown(xyz if xyz.shape[0] else np.zeros((0, 3)))
    cv2.imwrite(str(iter_root / f"landmarks_topdown_{stem}.png"), top)

    bgr0 = read_image_bgr(ds.image_paths[0])
    und0, _ = _undistort_if_needed(bgr0, ds)
    uv, z = project_cam0_to_uv_z(K, xyz)
    mask = np.isfinite(uv[:, 0]) & (z > 1e-9)
    panel = estimated_depth_visualization(und0, uv[mask], z[mask])
    cv2.imwrite(str(iter_root / f"sparse_depth_cam0_{stem}.png"), panel)

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from data.dataset import Dataset, load_gt_depth_for_frame, read_image_bgr
from pipeline.config import FeatureConfig
from pipeline.features import FrameFeatures, compute_frame_features_cache, detect_and_compute
from pipeline.geometry import essential_from_world_poses, invert_se3
from pipeline.map import IncrementalMap, MapConfig, TwoViewResult
from pipeline.fusion import FusedLandmarkMap, fused_world_points_homogeneous
from viz.overlays import (
    draw_epilines,
    draw_inlier_outlier_matches,
    draw_keypoints,
    draw_matches,
    estimated_depth_visualization,
    project_points_topdown,
    project_world_points_to_camera_uv_z,
    render_trajectory_topdown,
    sparse_depth_error_heatmap,
)
from viz.recorder import STEP_ORDER, PipelineRecorder


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


def fundamental_from_essential(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    Kinv = np.linalg.inv(K)
    return Kinv.T @ E @ Kinv


def iter_sequence_pairs(n_frames: int, pair_lookback: int) -> list[tuple[int, int]]:
    """Indices ``j`` paired with earlier frames ``j-off`` for ``off`` in ``1..min(pair_lookback, j)``."""
    wl = max(1, int(pair_lookback))
    out: list[tuple[int, int]] = []
    for j in range(1, n_frames):
        for off in range(1, min(wl, j) + 1):
            i = j - off
            out.append((i, j))
    return out


def export_single_pair_stages(
    ds: Dataset,
    pair_run_dir: str | Path,
    *,
    i: int,
    j: int,
    feat_cfg: FeatureConfig | None = None,
    reuse_two_view: TwoViewResult | None = None,
    frame_features_cache: Sequence[FrameFeatures] | None = None,
) -> None:
    """
    Run the two-view pipeline for frames ``(i, j)`` and write every ``STEP_ORDER`` PNG under
    ``pair_run_dir/steps/``.

    If ``reuse_two_view`` is provided (must match ``(i, j)``), geometry from that result is
    reused so sequence export can fuse landmarks without duplicate solving.

    If ``frame_features_cache`` is set (length at least ``max(i, j) + 1``), keypoints for ``02_keypoints``
    and matching inside ``add_frame_pair`` reuse precomputed descriptors instead of running
    detection again per pair.
    """
    feat_cfg = feat_cfg if feat_cfg is not None else ds.feature_config
    rec = PipelineRecorder(pair_run_dir)

    bgr_i = read_image_bgr(ds.image_paths[i])
    bgr_j = read_image_bgr(ds.image_paths[j])
    rec.write("raw_input", np.hstack([bgr_i, bgr_j]))

    und_i, g_i = _undistort_if_needed(bgr_i, ds)
    und_j, g_j = _undistort_if_needed(bgr_j, ds)

    K = ds.calibration.K
    Wi = ds.world_T_camera[i]
    Wj = ds.world_T_camera[j]

    fi = frame_features_cache[i] if frame_features_cache is not None else None
    fj = frame_features_cache[j] if frame_features_cache is not None else None

    if reuse_two_view is not None:
        assert reuse_two_view.frame_i == i and reuse_two_view.frame_j == j
        tw = reuse_two_view
    else:
        map_cfg = MapConfig()
        m = IncrementalMap(cfg=map_cfg, feat_cfg=feat_cfg, K=K, world_T_camera=ds.world_T_camera)
        tw = m.add_frame_pair(i, j, g_i, g_j, features_i=fi, features_j=fj)
    pts1, pts2 = tw.pts1, tw.pts2

    if fi is not None:
        kpi = fi.keypoints
    else:
        kpi, _ = detect_and_compute(g_i, feat_cfg)
    kp_img = draw_keypoints(und_i, np.float32([kp.pt for kp in kpi]))
    rec.write("keypoints", kp_img)
    rec.write("matches", draw_matches(und_i, und_j, pts1, pts2))

    E_viz = tw.E
    if E_viz is None:
        E_viz = essential_from_world_poses(Wi, Wj, K)
    F = fundamental_from_essential(np.asarray(E_viz, dtype=np.float64), K)
    epi_mask = tw.inlier_mask

    rec.write("epilines", draw_epilines(und_i, und_j, pts1, pts2, F, which="second"))
    rec.write("inlier_outlier", draw_inlier_outlier_matches(und_i, und_j, pts1, pts2, epi_mask))

    Xw = tw.X_world_h[:3, :].T
    valid = tw.cheiral_mask & np.all(np.isfinite(Xw), axis=1)
    scatter = project_points_topdown(Xw[valid] if np.any(valid) else np.zeros((0, 3)))
    repro = und_i.copy()
    Tcw_i = invert_se3(Wi)
    P = K @ np.hstack([Tcw_i[:3, :3], Tcw_i[:3, 3].reshape(3, 1)])
    for kk in range(tw.X_world_h.shape[1]):
        if not tw.cheiral_mask[kk]:
            continue
        Xh = tw.X_world_h[:, kk : kk + 1]
        x = P @ Xh
        u = int(x[0, 0] / (x[2, 0] + 1e-9))
        v = int(x[1, 0] / (x[2, 0] + 1e-9))
        if 0 <= u < repro.shape[1] and 0 <= v < repro.shape[0]:
            cv2.circle(repro, (u, v), 3, (0, 128, 255), -1, cv2.LINE_AA)
    tri_panel = np.hstack([repro, cv2.resize(scatter, (repro.shape[1], repro.shape[0]))])
    rec.write("triangulation", tri_panel)

    uv_est, z_est = project_world_points_to_camera_uv_z(tw.X_world_h, tw.cheiral_mask, K, Wi)
    rec.write("estimated_depth", estimated_depth_visualization(und_i, uv_est, z_est))

    gt_depth = load_gt_depth_for_frame(ds, i)
    err_img = und_i.copy()
    if gt_depth is not None and tw.X_world_h.shape[1] > 0:
        uv_list = []
        pred_list = []
        gt_list = []
        for kk in range(tw.X_world_h.shape[1]):
            if not tw.cheiral_mask[kk]:
                continue
            X = tw.X_world_h[:3, kk]
            Tcw = invert_se3(Wi)
            Xc = Tcw[:3, :3] @ X + Tcw[:3, 3]
            if Xc[2] <= 1e-6:
                continue
            u = int(K[0, 0] * (Xc[0] / Xc[2]) + K[0, 2])
            v = int(K[1, 1] * (Xc[1] / Xc[2]) + K[1, 2])
            if u < 0 or v < 0 or u >= gt_depth.shape[1] or v >= gt_depth.shape[0]:
                continue
            gz = float(gt_depth[v, u])
            if not np.isfinite(gz) or gz <= 1e-6:
                continue
            uv_list.append([u, v])
            pred_list.append(float(Xc[2]))
            gt_list.append(gz)
        if uv_list:
            err_img = sparse_depth_error_heatmap(
                err_img,
                np.float32(uv_list),
                np.float32(pred_list),
                np.float32(gt_list),
            )
    rec.write("depth_error", err_img)


def export_all_stages(
    ds: Dataset,
    run_dir: str | Path,
    *,
    i: int = 0,
    j: int = 1,
    feat_cfg: FeatureConfig | None = None,
    frame_features_cache: Sequence[FrameFeatures] | None = None,
) -> None:
    """Single pair ``(i, j)`` into flat ``run_dir/steps/``."""
    export_single_pair_stages(
        ds,
        run_dir,
        i=i,
        j=j,
        feat_cfg=feat_cfg,
        frame_features_cache=frame_features_cache,
    )


def export_sequence_consecutive_pairs(
    ds: Dataset,
    run_dir: str | Path,
    *,
    feat_cfg: FeatureConfig | None = None,
    fuse_merge_px: float = 4.0,
    pair_lookback: int = 10,
) -> list[tuple[int, int]]:
    """
    For each frame index ``j`` from ``1`` to ``n-1``, pair it with frames
    ``j-1, j-2, \\ldots`` up to ``pair_lookback`` prior frames.

    Layout::

        run_dir/
          pairs/
            iii_jjj/steps/*.png
          summary/
            trajectory_topdown_full_sequence.png
            fused_landmarks_topdown.png
            fused_estimated_depth_ref000.png

    A shared ``IncrementalMap`` solves each edge once; ``FusedLandmarkMap`` merges landmarks that
    re-observe the same approximate pixel on a shared frame.

    Using ``pair_lookback=1`` reproduces consecutive pairs only ``(k, k+1)``.
    """
    root = Path(run_dir)
    pairs_root = root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    n = len(ds.image_paths)
    if n < 2:
        raise ValueError(f"need at least 2 images for sequence export, got {n}")

    fc = feat_cfg if feat_cfg is not None else ds.feature_config
    wl = max(1, int(pair_lookback))
    grays: list[np.ndarray] = []
    for path in ds.image_paths:
        bgr = read_image_bgr(path)
        _, g = _undistort_if_needed(bgr, ds)
        grays.append(g)

    frame_cache = compute_frame_features_cache(grays, fc)

    map_cfg = MapConfig()
    inc = IncrementalMap(
        cfg=map_cfg,
        feat_cfg=fc,
        K=ds.calibration.K,
        world_T_camera=ds.world_T_camera,
        window=10**9,
    )
    fused = FusedLandmarkMap(merge_px=fuse_merge_px)

    pairs = iter_sequence_pairs(n, wl)
    for i, j in pairs:
        tw = inc.add_frame_pair(
            i,
            j,
            grays[i],
            grays[j],
            features_i=frame_cache[i],
            features_j=frame_cache[j],
        )
        fused.integrate_two_view_result(tw)
        pair_dir = pairs_root / f"{i:03d}_{j:03d}"
        export_single_pair_stages(
            ds,
            pair_dir,
            i=i,
            j=j,
            feat_cfg=fc,
            reuse_two_view=tw,
            frame_features_cache=frame_cache,
        )

    traj_full = render_trajectory_topdown(
        ds.world_T_camera,
        ds.gt_world_T_camera,
        highlight_frame_indices=None,
    )
    out_png = summary_root / "trajectory_topdown_full_sequence.png"
    ok = cv2.imwrite(str(out_png), traj_full)
    if not ok:
        raise RuntimeError(f"failed to write {out_png}")

    K = ds.calibration.K
    W0 = ds.world_T_camera[0]
    X_h, cheiral_pass = fused_world_points_homogeneous(fused)
    uv_f, z_f = project_world_points_to_camera_uv_z(X_h, cheiral_pass, K, W0)
    bgr0 = read_image_bgr(ds.image_paths[0])
    und0, _ = _undistort_if_needed(bgr0, ds)
    fused_depth = estimated_depth_visualization(und0, uv_f, z_f)
    nf = len(fused.landmarks)
    cv2.putText(
        fused_depth,
        f"fused landmarks: {nf}",
        (12, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    fused_depth_path = summary_root / "fused_estimated_depth_ref000.png"
    if not cv2.imwrite(str(fused_depth_path), fused_depth):
        raise RuntimeError(f"failed to write {fused_depth_path}")

    topdown = project_points_topdown(fused.positions_xyz())
    cv2.putText(
        topdown,
        f"fused n={nf}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    fused_xy_path = summary_root / "fused_landmarks_topdown.png"
    if not cv2.imwrite(str(fused_xy_path), topdown):
        raise RuntimeError(f"failed to write {fused_xy_path}")

    return pairs


def ensure_sequence_outputs_exist(run_dir: str | Path, n_frames: int, pair_lookback: int = 10) -> None:
    """Check pair step PNGs, trajectory summary, and fused-landmark summaries."""
    root = Path(run_dir)
    wl = max(1, int(pair_lookback))
    for i, j in iter_sequence_pairs(n_frames, wl):
        ensure_all_step_pngs_exist(root / "pairs" / f"{i:03d}_{j:03d}")
    sf = root / "summary" / "trajectory_topdown_full_sequence.png"
    if not sf.is_file():
        raise FileNotFoundError(f"missing sequence summary: {sf}")
    for name in ("fused_landmarks_topdown.png", "fused_estimated_depth_ref000.png"):
        p = root / "summary" / name
        if not p.is_file():
            raise FileNotFoundError(f"missing fused summary: {p}")


def ensure_all_step_pngs_exist(run_dir: str | Path) -> list[Path]:
    """Verify every documented stage file is present (after export_all_stages)."""
    rec = PipelineRecorder(run_dir)
    missing = []
    for slug in STEP_ORDER:
        p = rec.path_for(slug)
        if not p.is_file():
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"missing step PNGs: {missing}")
    return [rec.path_for(s) for s in STEP_ORDER]

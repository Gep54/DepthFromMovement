from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data.dataset import Dataset, read_image_bgr, load_gt_depth_for_frame
from pipeline.config import FeatureConfig, MotionMode
from pipeline.features import detect_and_compute
from pipeline.geometry import essential_from_world_poses, relative_motion_from_world_poses
from pipeline.map import IncrementalMap, MapConfig, TwoViewResult
from pipeline.fusion import FusedLandmarkMap, fused_world_points_homogeneous
from pipeline.matching import match_pair_points
from pipeline.triangulation import triangulate_cam1_frame
from pipeline.geometry import (
    estimate_essential_ransac,
    recover_pose_from_essential,
    scale_from_odometry,
    align_translation_direction,
)
from pipeline.geometry import invert_se3
from viz.overlays import (
    draw_epilines,
    draw_inlier_outlier_matches,
    draw_keypoints,
    draw_matches,
    estimated_depth_visualization,
    project_points_topdown,
    project_world_points_to_camera_uv_z,
    render_depth_histogram_panel,
    render_trajectory_topdown,
    sparse_depth_error_heatmap,
)
from viz.recorder import PipelineRecorder, STEP_ORDER


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


def export_single_pair_stages(
    ds: Dataset,
    pair_run_dir: str | Path,
    *,
    i: int,
    j: int,
    motion_mode: MotionMode = "known_pose",
    feat_cfg: FeatureConfig | None = None,
    reuse_two_view: TwoViewResult | None = None,
) -> None:
    """
    Run the two-view pipeline for frames ``(i, j)`` and write every ``STEP_ORDER`` PNG under
    ``pair_run_dir/steps/``.

    If ``reuse_two_view`` is provided (must match ``(i, j)``), geometry from that result is
    reused so sequence export can fuse landmarks without duplicate solving.
    """
    feat_cfg = feat_cfg if feat_cfg is not None else ds.feature_config
    rec = PipelineRecorder(pair_run_dir)

    bgr_i = read_image_bgr(ds.image_paths[i])
    bgr_j = read_image_bgr(ds.image_paths[j])
    rec.write("raw_input", np.hstack([bgr_i, bgr_j]))

    und_i, g_i = _undistort_if_needed(bgr_i, ds)
    und_j, g_j = _undistort_if_needed(bgr_j, ds)
    if ds.calibration.dist_coeffs is not None and np.linalg.norm(ds.calibration.dist_coeffs) > 1e-9:
        rec.write("undistort", np.hstack([und_i, und_j]))
    else:
        rec.write("undistort", np.hstack([bgr_i, bgr_j]))

    K = ds.calibration.K
    Wi = ds.world_T_camera[i]
    Wj = ds.world_T_camera[j]

    if reuse_two_view is not None:
        assert reuse_two_view.frame_i == i and reuse_two_view.frame_j == j
        tw = reuse_two_view
        pts1, pts2 = tw.pts1, tw.pts2
        kpi, di = detect_and_compute(g_i, feat_cfg)
    else:
        kpi, di = detect_and_compute(g_i, feat_cfg)
        kpj, dj = detect_and_compute(g_j, feat_cfg)
        pts1, pts2, _ = match_pair_points(kpi, kpj, di, dj, feat_cfg)

    kp_img = draw_keypoints(und_i, np.float32([kp.pt for kp in kpi]))
    rec.write("keypoints", kp_img)
    rec.write("matches", draw_matches(und_i, und_j, pts1, pts2))

    E = essential_from_world_poses(Wi, Wj, K)
    F = fundamental_from_essential(E, K)
    if reuse_two_view is not None:
        epi_mask = tw.inlier_mask
    else:
        _, epi_mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
    epi_inl = epi_mask.ravel().astype(bool)
    rec.write("epilines", draw_epilines(und_i, und_j, pts1, pts2, F, which="second"))
    rec.write("inlier_outlier", draw_inlier_outlier_matches(und_i, und_j, pts1, pts2, epi_mask))

    if reuse_two_view is None:
        map_cfg = MapConfig(motion_mode=motion_mode)
        m = IncrementalMap(cfg=map_cfg, feat_cfg=feat_cfg, K=K, world_T_camera=ds.world_T_camera)
        tw = m.add_frame_pair(i, j, g_i, g_j)
    Xw = tw.X_world_h[:3, :].T
    valid = tw.cheiral_mask & np.all(np.isfinite(Xw), axis=1)
    scatter = project_points_topdown(Xw[valid] if np.any(valid) else np.zeros((0, 3)))
    repro = und_i.copy()
    Tcw_i = invert_se3(Wi)
    P = K @ np.hstack([Tcw_i[:3, :3], Tcw_i[:3, 3].reshape(3, 1)])
    for k in range(tw.X_world_h.shape[1]):
        if not tw.cheiral_mask[k]:
            continue
        Xh = tw.X_world_h[:, k : k + 1]
        x = P @ Xh
        u = int(x[0, 0] / (x[2, 0] + 1e-9))
        v = int(x[1, 0] / (x[2, 0] + 1e-9))
        if 0 <= u < repro.shape[1] and 0 <= v < repro.shape[0]:
            cv2.circle(repro, (u, v), 3, (0, 128, 255), -1, cv2.LINE_AA)
    tri_panel = np.hstack([repro, cv2.resize(scatter, (repro.shape[1], repro.shape[0]))])
    rec.write("triangulation", tri_panel)

    uv_est, z_est = project_world_points_to_camera_uv_z(tw.X_world_h, tw.cheiral_mask, K, Wi)
    rec.write("estimated_depth", estimated_depth_visualization(und_i, uv_est, z_est))

    if motion_mode == "estimate_essential":
        p1i = pts1[epi_inl]
        p2i = pts2[epi_inl]
        if len(p1i) < 8:
            rec.write("scale_depth", render_depth_histogram_panel(np.array([]), None))
        else:
            E_est, m_est = estimate_essential_ransac(p1i, p2i, K, threshold=1.0)
            Rv, tv, _ = recover_pose_from_essential(E_est, p1i, p2i, K, mask=m_est)
            _, t_gt = relative_motion_from_world_poses(Wi, Wj)
            tv = align_translation_direction(tv, t_gt)
            s, ok = scale_from_odometry(tv, t_gt)
            m2b = m_est.ravel().astype(bool)
            if np.count_nonzero(m2b) < 2:
                rec.write("scale_depth", render_depth_histogram_panel(np.array([]), None))
            else:
                Xb, ch = triangulate_cam1_frame(p1i[m2b], p2i[m2b], K, Rv, tv)
                zb = Xb[2, ch]
                Xa = Xb.copy()
                if ok:
                    Xa[:3, :] *= s
                za = Xa[2, ch]
                rec.write(
                    "scale_depth",
                    render_depth_histogram_panel(zb, za if ok else None),
                )
    else:
        Z = tw.X_world_h[2, :]
        rec.write("scale_depth", render_depth_histogram_panel(Z, None))

    traj = render_trajectory_topdown(
        ds.world_T_camera,
        ds.gt_world_T_camera,
        highlight_frame_indices=(i, j),
    )
    rec.write("trajectory_topdown", traj)

    gt_depth = load_gt_depth_for_frame(ds, i)
    err_img = und_i.copy()
    if gt_depth is not None and tw.X_world_h.shape[1] > 0:
        uv_list = []
        pred_list = []
        gt_list = []
        for k in range(tw.X_world_h.shape[1]):
            if not tw.cheiral_mask[k]:
                continue
            X = tw.X_world_h[:3, k]
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
    motion_mode: MotionMode = "known_pose",
    feat_cfg: FeatureConfig | None = None,
) -> None:
    """Single pair ``(i, j)`` into flat ``run_dir/steps/`` (backwards-compatible layout)."""
    export_single_pair_stages(ds, run_dir, i=i, j=j, motion_mode=motion_mode, feat_cfg=feat_cfg)


def export_sequence_consecutive_pairs(
    ds: Dataset,
    run_dir: str | Path,
    *,
    motion_mode: MotionMode = "known_pose",
    feat_cfg: FeatureConfig | None = None,
    fuse_merge_px: float = 4.0,
) -> list[tuple[int, int]]:
    """
    Run consecutive pairs ``(0,1),(1,2),\\ldots,(n\\!-\\!2,n\\!-\\!1)``.

    Layout::

        run_dir/
          pairs/
            000_001/steps/*.png
            001_002/steps/*.png
            ...
          summary/
            trajectory_topdown_full_sequence.png

    Each pair folder repeats all pipeline PNGs for that slice; trajectory PNG highlights that pair.

    A shared ``IncrementalMap`` solves each edge once; ``FusedLandmarkMap`` merges landmarks that
    re-observe the same approximate pixel on a shared frame. Summary also writes
    ``fused_landmarks_topdown.png`` and ``fused_estimated_depth_ref000.png``.
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
    grays: list[np.ndarray] = []
    for path in ds.image_paths:
        bgr = read_image_bgr(path)
        _, g = _undistort_if_needed(bgr, ds)
        grays.append(g)

    map_cfg = MapConfig(motion_mode=motion_mode)
    inc = IncrementalMap(
        cfg=map_cfg,
        feat_cfg=fc,
        K=ds.calibration.K,
        world_T_camera=ds.world_T_camera,
        window=10**9,
    )
    fused = FusedLandmarkMap(merge_px=fuse_merge_px)

    pairs: list[tuple[int, int]] = []
    for i in range(n - 1):
        j = i + 1
        tw = inc.add_frame_pair(i, j, grays[i], grays[j])
        fused.integrate_two_view_result(tw)
        pair_dir = pairs_root / f"{i:03d}_{j:03d}"
        export_single_pair_stages(
            ds,
            pair_dir,
            i=i,
            j=j,
            motion_mode=motion_mode,
            feat_cfg=fc,
            reuse_two_view=tw,
        )
        pairs.append((i, j))

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


def ensure_sequence_outputs_exist(run_dir: str | Path, n_frames: int) -> None:
    """Check pair step PNGs, trajectory summary, and fused-landmark summaries."""
    root = Path(run_dir)
    for i in range(n_frames - 1):
        ensure_all_step_pngs_exist(root / "pairs" / f"{i:03d}_{i + 1:03d}")
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

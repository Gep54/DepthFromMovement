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
from viz.match_classification import (
    append_rejection_audit,
    audit_record,
    classify_match_rejections,
    write_pairs_all_rejection_types,
)
from viz.epipolar_report import EpipolarPairView, export_epipolar_pdf_bundle
from viz.overlays import (
    MATCH_COLOR_CHEIRAL,
    MATCH_COLOR_EPIPOLAR,
    canny_edges_bgr,
    draw_classified_matches,
    draw_matches,
    draw_rich_keypoints,
    estimated_depth_visualization,
    grayscale_to_bgr,
    project_points_topdown,
    project_world_points_to_camera_uv_z,
    render_trajectory_topdown,
    sparse_depth_error_heatmap,
)
from viz.match_classification import MatchClassification
from viz.recorder import (
    PipelineRecorder,
    geometry_slug_order,
    pair_slug_orders,
    ensure_step_pngs_exist as _ensure_step_pngs_exist,
)


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


def _log_progress(message: str) -> None:
    """Stdout progress line (flushed) for long offline exports."""
    print(message, flush=True)


def fundamental_from_essential(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    Kinv = np.linalg.inv(K)
    return Kinv.T @ E @ Kinv


def epipolar_pair_view(
    *,
    i: int,
    j: int,
    und_i: np.ndarray,
    und_j: np.ndarray,
    tw: TwoViewResult,
    K: np.ndarray,
    world_T_camera: Sequence[np.ndarray],
) -> EpipolarPairView:
    E_viz = tw.E
    if E_viz is None:
        E_viz = essential_from_world_poses(world_T_camera[i], world_T_camera[j], K)
    F = fundamental_from_essential(np.asarray(E_viz, dtype=np.float64), K)
    return EpipolarPairView(
        frame_i=i,
        frame_j=j,
        und_i=und_i,
        und_j=und_j,
        pts1=tw.pts1,
        pts2=tw.pts2,
        F=F,
    )


def iter_sequence_pairs(n_frames: int, pair_lookback: int) -> list[tuple[int, int]]:
    """Indices ``j`` paired with earlier frames ``j-off`` for ``off`` in ``1..min(pair_lookback, j)``."""
    wl = max(1, int(pair_lookback))
    out: list[tuple[int, int]] = []
    for j in range(1, n_frames):
        for off in range(1, min(wl, j) + 1):
            i = j - off
            out.append((i, j))
    return out


def export_single_frame_stages(
    ds: Dataset,
    pair_run_dir: str | Path,
    *,
    frame_idx: int,
    feat_cfg: FeatureConfig | None = None,
    frame_features_cache: Sequence[FrameFeatures] | None = None,
) -> None:
    """Write ``steps/single/`` preprocessing figures for reference frame ``frame_idx``."""
    feat_cfg = feat_cfg if feat_cfg is not None else ds.feature_config
    rec = PipelineRecorder(pair_run_dir, subdir="single")

    bgr = read_image_bgr(ds.image_paths[frame_idx])
    und, gray = _undistort_if_needed(bgr, ds)

    rec.write("original", und)
    rec.write("grayscale", grayscale_to_bgr(gray))
    rec.write("edges", canny_edges_bgr(gray))

    if frame_features_cache is not None:
        kpi = frame_features_cache[frame_idx].keypoints
    else:
        kpi, _ = detect_and_compute(gray, feat_cfg)
    rec.write("descriptors", draw_rich_keypoints(und, kpi))


def _write_rejection_detail_stages(
    rec_pair: PipelineRecorder,
    und_i: np.ndarray,
    und_j: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    cls: MatchClassification,
) -> None:
    """One mosaic per rejection method (epipolar / cheiral)."""
    panels = (
        ("rejected_epipolar", cls.epipolar, MATCH_COLOR_EPIPOLAR),
        ("rejected_cheiral", cls.cheiral, MATCH_COLOR_CHEIRAL),
    )
    for slug, mask, color in panels:
        m = mask.reshape(-1).astype(bool)
        if not np.any(m):
            rec_pair.write(slug, np.hstack([und_i, und_j]))
            continue
        rec_pair.write(slug, draw_matches(und_i, und_j, pts1[m], pts2[m], color=color))


def export_geometry_stages(
    ds: Dataset,
    pair_run_dir: str | Path,
    *,
    i: int,
    j: int,
    tw: TwoViewResult,
    und_i: np.ndarray,
    full_steps: bool = False,
) -> None:
    """Write ``steps/geometry/`` (minimal: estimated depth only; full: all panels)."""
    geom_order = geometry_slug_order(full_steps=full_steps)
    rec = PipelineRecorder(pair_run_dir, subdir="geometry", slug_order=geom_order)
    K = ds.calibration.K
    Wi = ds.world_T_camera[i]
    Wj = ds.world_T_camera[j]

    Xw = tw.X_world_h[:3, :].T
    valid = np.all(np.isfinite(Xw), axis=1)
    scatter = project_points_topdown(Xw[valid] if np.any(valid) else np.zeros((0, 3)))
    repro = und_i.copy()
    Tcw_i = invert_se3(Wi)
    P = K @ np.hstack([Tcw_i[:3, :3], Tcw_i[:3, 3].reshape(3, 1)])
    for kk in range(tw.X_world_h.shape[1]):
        if not np.all(np.isfinite(tw.X_world_h[:3, kk])):
            continue
        Xh = tw.X_world_h[:, kk : kk + 1]
        x = P @ Xh
        u = int(x[0, 0] / (x[2, 0] + 1e-9))
        v = int(x[1, 0] / (x[2, 0] + 1e-9))
        if 0 <= u < repro.shape[1] and 0 <= v < repro.shape[0]:
            cv2.circle(repro, (u, v), 3, (0, 128, 255), -1, cv2.LINE_AA)
    if full_steps:
        tri_panel = np.hstack([repro, cv2.resize(scatter, (repro.shape[1], repro.shape[0]))])
        rec.write("triangulation", tri_panel)

    uv_est, z_est = project_world_points_to_camera_uv_z(tw.X_world_h, tw.cheiral_mask, K, Wi)
    rec.write("estimated_depth", estimated_depth_visualization(und_i, uv_est, z_est))

    if not full_steps:
        return

    gt_depth = load_gt_depth_for_frame(ds, i)
    err_img = und_i.copy()
    if gt_depth is not None and tw.X_world_h.shape[1] > 0:
        uv_list = []
        pred_list = []
        gt_list = []
        for kk in range(tw.X_world_h.shape[1]):
            X = tw.X_world_h[:3, kk]
            if not np.all(np.isfinite(X)):
                continue
            Tcw = invert_se3(Wi)
            Xc = Tcw[:3, :3] @ X + Tcw[:3, 3]
            if not np.isfinite(Xc[2]) or abs(Xc[2]) <= 1e-12:
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


def export_single_pair_stages(
    ds: Dataset,
    pair_run_dir: str | Path,
    *,
    i: int,
    j: int,
    feat_cfg: FeatureConfig | None = None,
    reuse_two_view: TwoViewResult | None = None,
    frame_features_cache: Sequence[FrameFeatures] | None = None,
    include_geometry: bool = True,
    rejection_audit_path: str | Path | None = None,
    check_cheiral: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
    export_epipolar: bool = False,
) -> dict:
    """
    Run the two-view pipeline for frames ``(i, j)`` and write illustration PNGs.

    Returns the rejection audit record for this pair.
    """
    feat_cfg = feat_cfg if feat_cfg is not None else ds.feature_config
    pair_root = Path(pair_run_dir)

    if full_steps:
        export_single_frame_stages(
            ds,
            pair_root,
            frame_idx=i,
            feat_cfg=feat_cfg,
            frame_features_cache=frame_features_cache,
        )

    bgr_i = read_image_bgr(ds.image_paths[i])
    bgr_j = read_image_bgr(ds.image_paths[j])
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
        map_cfg = MapConfig(check_cheiral=check_cheiral)
        m = IncrementalMap(cfg=map_cfg, feat_cfg=feat_cfg, K=K, world_T_camera=ds.world_T_camera)
        tw = m.add_frame_pair(i, j, g_i, g_j, features_i=fi, features_j=fj)

    pts1, pts2 = tw.pts1, tw.pts2
    pair_order = pair_slug_orders(full_steps=full_steps, detail_log=detail_log)
    rec_pair = PipelineRecorder(pair_root, subdir="pair", slug_order=pair_order)

    if full_steps:
        rec_pair.write("raw_input", np.hstack([und_i, und_j]))
    rec_pair.write("matches", draw_matches(und_i, und_j, pts1, pts2))

    cls = classify_match_rejections(tw, check_cheiral=check_cheiral)
    rec_pair.write(
        "match_classifications",
        draw_classified_matches(
            und_i,
            und_j,
            pts1,
            pts2,
            epipolar=cls.epipolar,
            cheiral=cls.cheiral,
            inlier=cls.inlier,
        ),
    )
    rec_pair.write(
        "inliers",
        draw_matches(und_i, und_j, pts1[cls.inlier], pts2[cls.inlier]),
    )

    if detail_log:
        _write_rejection_detail_stages(rec_pair, und_i, und_j, pts1, pts2, cls)

    record = audit_record(i, j, cls)
    if rejection_audit_path is not None:
        append_rejection_audit(rejection_audit_path, record)

    if include_geometry:
        export_geometry_stages(
            ds, pair_root, i=i, j=j, tw=tw, und_i=und_i, full_steps=full_steps
        )

    if export_epipolar:
        export_epipolar_pdf_bundle(
            pair_root,
            [epipolar_pair_view(i=i, j=j, und_i=und_i, und_j=und_j, tw=tw, K=K, world_T_camera=ds.world_T_camera)],
        )

    return record


def export_all_stages(
    ds: Dataset,
    run_dir: str | Path,
    *,
    i: int = 0,
    j: int = 1,
    feat_cfg: FeatureConfig | None = None,
    frame_features_cache: Sequence[FrameFeatures] | None = None,
    include_geometry: bool = True,
    rejection_audit_path: str | Path | None = None,
    check_cheiral: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
    export_epipolar: bool = False,
) -> dict:
    """Single pair ``(i, j)`` into ``run_dir/steps/{single,pair,geometry}/``."""
    _log_progress(f"dfm-export-steps: pair {i}-{j} -> {Path(run_dir).resolve()}")
    audit_path = rejection_audit_path if rejection_audit_path is not None else Path(run_dir) / "rejection_audit.jsonl"
    if Path(audit_path).exists():
        Path(audit_path).unlink()
    record = export_single_pair_stages(
        ds,
        run_dir,
        i=i,
        j=j,
        feat_cfg=feat_cfg,
        frame_features_cache=frame_features_cache,
        include_geometry=include_geometry,
        rejection_audit_path=audit_path,
        check_cheiral=check_cheiral,
        full_steps=full_steps,
        detail_log=detail_log,
        export_epipolar=export_epipolar,
    )
    _log_progress("dfm-export-steps: writing summary (pairs_all_rejection_types.json)")
    summary_dir = Path(run_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_all_rejection_types(summary_dir / "pairs_all_rejection_types.json", [record])
    _log_progress("dfm-export-steps: done")
    return record


def export_sequence_consecutive_pairs(
    ds: Dataset,
    run_dir: str | Path,
    *,
    feat_cfg: FeatureConfig | None = None,
    fuse_merge_px: float = 4.0,
    pair_lookback: int = 10,
    include_geometry: bool = True,
    rejection_audit_path: str | Path | None = None,
    check_cheiral: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
    export_epipolar: bool = False,
) -> list[tuple[int, int]]:
    """
    For each frame index ``j`` from ``1`` to ``n-1``, pair it with frames
    ``j-1, j-2, \\ldots`` up to ``pair_lookback`` prior frames.

    Layout::

        run_dir/
          pairs/
            iii_jjj/steps/{single,pair,geometry}/...
          summary/
            trajectory_topdown_full_sequence.png
            fused_landmarks_topdown.png
            fused_estimated_depth_ref000.png
            pairs_all_rejection_types.json
          rejection_audit.jsonl
    """
    root = Path(run_dir)
    pairs_root = root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    n = len(ds.image_paths)
    if n < 2:
        raise ValueError(f"need at least 2 images for sequence export, got {n}")

    audit_path = rejection_audit_path if rejection_audit_path is not None else root / "rejection_audit.jsonl"
    if Path(audit_path).exists():
        Path(audit_path).unlink()

    fc = feat_cfg if feat_cfg is not None else ds.feature_config
    wl = max(1, int(pair_lookback))
    grays: list[np.ndarray] = []
    for path in ds.image_paths:
        bgr = read_image_bgr(path)
        _, g = _undistort_if_needed(bgr, ds)
        grays.append(g)

    _log_progress("dfm-export-steps: computing per-frame features …")
    frame_cache = compute_frame_features_cache(grays, fc)

    map_cfg = MapConfig(check_cheiral=check_cheiral)
    inc = IncrementalMap(
        cfg=map_cfg,
        feat_cfg=fc,
        K=ds.calibration.K,
        world_T_camera=ds.world_T_camera,
        window=10**9,
    )
    fused = FusedLandmarkMap(merge_px=fuse_merge_px)

    pairs = iter_sequence_pairs(n, wl)
    n_pairs = len(pairs)
    _log_progress(
        f"dfm-export-steps: {n} frames, {n_pairs} pairs (lookback={wl}) -> {root.resolve()}"
    )
    _log_progress("dfm-export-steps: computing per-frame features …")
    audit_records: list[dict] = []
    epipolar_views: list[EpipolarPairView] = []
    for pair_idx, (i, j) in enumerate(pairs, start=1):
        _log_progress(f"dfm-export-steps: pair {i}-{j} ({pair_idx}/{n_pairs}) …")
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
        bgr_i = read_image_bgr(ds.image_paths[i])
        bgr_j = read_image_bgr(ds.image_paths[j])
        und_i, _ = _undistort_if_needed(bgr_i, ds)
        und_j, _ = _undistort_if_needed(bgr_j, ds)
        if export_epipolar:
            epipolar_views.append(
                epipolar_pair_view(
                    i=i,
                    j=j,
                    und_i=und_i,
                    und_j=und_j,
                    tw=tw,
                    K=ds.calibration.K,
                    world_T_camera=ds.world_T_camera,
                )
            )
        record = export_single_pair_stages(
            ds,
            pair_dir,
            i=i,
            j=j,
            feat_cfg=fc,
            reuse_two_view=tw,
            frame_features_cache=frame_cache,
            include_geometry=include_geometry,
            rejection_audit_path=audit_path,
            check_cheiral=check_cheiral,
            full_steps=full_steps,
            detail_log=detail_log,
            export_epipolar=False,
        )
        n_tri = int(tw.X_world_h.shape[1]) if tw.X_world_h.size else 0
        _log_progress(
            f"dfm-export-steps: pair {i}-{j} done ({n_tri} triangulated cols, fused n={len(fused.landmarks)})"
        )
        audit_records.append(record)

    _log_progress(f"dfm-export-steps: finished {n_pairs} pairs, writing summary …")

    if export_epipolar and epipolar_views:
        _log_progress(f"dfm-export-steps: writing epipolar PDFs ({len(epipolar_views)} pages) …")
        export_epipolar_pdf_bundle(root, epipolar_views)

    _log_progress("dfm-export-steps: writing summary/pairs_all_rejection_types.json …")
    write_pairs_all_rejection_types(summary_root / "pairs_all_rejection_types.json", audit_records)

    _log_progress("dfm-export-steps: writing summary/trajectory_topdown_full_sequence.png …")
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
    nf = len(fused.landmarks)
    _log_progress(f"dfm-export-steps: writing summary/fused_estimated_depth_ref000.png ({nf} landmarks) …")
    X_h, cheiral_pass = fused_world_points_homogeneous(fused)
    uv_f, z_f = project_world_points_to_camera_uv_z(X_h, cheiral_pass, K, W0)
    bgr0 = read_image_bgr(ds.image_paths[0])
    und0, _ = _undistort_if_needed(bgr0, ds)
    fused_depth = estimated_depth_visualization(und0, uv_f, z_f)
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
    _log_progress("dfm-export-steps: writing summary/fused_landmarks_topdown.png …")
    fused_xy_path = summary_root / "fused_landmarks_topdown.png"
    if not cv2.imwrite(str(fused_xy_path), topdown):
        raise RuntimeError(f"failed to write {fused_xy_path}")

    _log_progress("dfm-export-steps: done")
    return pairs


def ensure_sequence_outputs_exist(
    run_dir: str | Path,
    n_frames: int,
    pair_lookback: int = 10,
    *,
    include_geometry: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
) -> None:
    """Check pair step PNGs, trajectory summary, fused-landmark summaries, and audit file."""
    root = Path(run_dir)
    wl = max(1, int(pair_lookback))
    for i, j in iter_sequence_pairs(n_frames, wl):
        ensure_step_pngs_exist(
            root / "pairs" / f"{i:03d}_{j:03d}",
            include_geometry=include_geometry,
            full_steps=full_steps,
            detail_log=detail_log,
        )
    sf = root / "summary" / "trajectory_topdown_full_sequence.png"
    if not sf.is_file():
        raise FileNotFoundError(f"missing sequence summary: {sf}")
    for name in ("fused_landmarks_topdown.png", "fused_estimated_depth_ref000.png"):
        p = root / "summary" / name
        if not p.is_file():
            raise FileNotFoundError(f"missing fused summary: {p}")
    audit = root / "rejection_audit.jsonl"
    if not audit.is_file():
        raise FileNotFoundError(f"missing rejection audit: {audit}")
    pairs_json = root / "summary" / "pairs_all_rejection_types.json"
    if not pairs_json.is_file():
        raise FileNotFoundError(f"missing pairs summary: {pairs_json}")


def ensure_step_pngs_exist(
    run_dir: str | Path,
    *,
    include_geometry: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
) -> list[Path]:
    """Verify stage PNGs for the active export profile."""
    return _ensure_step_pngs_exist(
        run_dir,
        include_geometry=include_geometry,
        full_steps=full_steps,
        detail_log=detail_log,
    )


def ensure_all_step_pngs_exist(
    run_dir: str | Path,
    *,
    include_geometry: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
) -> list[Path]:
    return ensure_step_pngs_exist(
        run_dir,
        include_geometry=include_geometry,
        full_steps=full_steps,
        detail_log=detail_log,
    )

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.config import FeatureConfig, clamp_motion_confidence
from pipeline.geometry import (
    essential_from_world_poses,
    blend_relative_pose,
    essential_from_R_t,
    recover_pose_from_essential,
    relative_motion_from_world_poses,
    align_translation_direction,
)
from pipeline.matching import match_pair_points
from pipeline.triangulation import triangulate_world_points, triangulate_cam1_frame, cam1_to_world_points
from pipeline.features import FrameFeatures, detect_and_compute
from pipeline.metrics import reprojection_errors, summarize_reprojection


@dataclass
class MapConfig:
    """``motion_confidence`` in [0, 1]: 1 = odometry relative pose; 0 = vision-only relative pose."""

    motion_confidence: float = 1.0
    ransac_epipolar_thresh: float = 1.0
    min_parallax_deg: float = 0.5

    def __post_init__(self) -> None:
        self.motion_confidence = clamp_motion_confidence(self.motion_confidence)


@dataclass
class TwoViewResult:
    frame_i: int
    frame_j: int
    pts1: np.ndarray
    pts2: np.ndarray
    inlier_mask: np.ndarray
    E: np.ndarray | None
    R_est: np.ndarray | None
    t_est: np.ndarray | None
    scale: float
    scale_ok: bool
    X_world_h: np.ndarray
    cheiral_mask: np.ndarray
    reproj: dict[str, float]
    """Rows aligned with columns of ``X_world_h`` (inlier correspondences); invalid rows ignored."""
    descriptors: np.ndarray | None = None


@dataclass
class IncrementalMap:
    """Sliding-window landmark store: last `window` frames as indices."""

    cfg: MapConfig
    feat_cfg: FeatureConfig
    K: np.ndarray
    world_T_camera: list[np.ndarray]
    window: int = 5
    tracks: dict[int, dict[int, tuple[float, float]]] = field(default_factory=dict)
    """landmark_id -> frame_idx -> (u,v)"""
    landmarks: dict[int, np.ndarray] = field(default_factory=dict)
    """landmark_id -> 3-vector world"""
    next_landmark_id: int = 0
    pair_results: list[TwoViewResult] = field(default_factory=list)

    def add_frame_pair(
        self,
        i: int,
        j: int,
        gray_i: np.ndarray,
        gray_j: np.ndarray,
        *,
        features_i: FrameFeatures | None = None,
        features_j: FrameFeatures | None = None,
    ) -> TwoViewResult:
        if features_i is not None:
            kpi, di = features_i.keypoints, features_i.descriptors
        else:
            kpi, di = detect_and_compute(gray_i, self.feat_cfg)
        if features_j is not None:
            kpj, dj = features_j.keypoints, features_j.descriptors
        else:
            kpj, dj = detect_and_compute(gray_j, self.feat_cfg)
        pts1, pts2, matches = match_pair_points(kpi, kpj, di, dj, self.feat_cfg)
        if matches and di is not None:
            desc_rows = np.stack([np.asarray(di[m.queryIdx]) for m in matches], axis=0)
        else:
            d_dim = int(di.shape[1]) if di is not None else 0
            desc_rows = np.empty((0, d_dim), dtype=di.dtype if di is not None else np.uint8)
        if len(pts1) < 8:
            empty = np.zeros((4, 0), np.float64)
            return TwoViewResult(
                frame_i=i,
                frame_j=j,
                pts1=pts1,
                pts2=pts2,
                inlier_mask=np.zeros((len(pts1), 1), np.uint8),
                E=None,
                R_est=None,
                t_est=None,
                scale=1.0,
                scale_ok=False,
                X_world_h=empty,
                cheiral_mask=np.zeros((0,), bool),
                reproj={},
                descriptors=None,
            )

        Wi = self.world_T_camera[i]
        Wj = self.world_T_camera[j]
        R_gt, t_gt = relative_motion_from_world_poses(Wi, Wj)
        scale = 1.0
        scale_ok = True
        alpha = float(self.cfg.motion_confidence)

        E_fm, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.cfg.ransac_epipolar_thresh,
        )
        E: np.ndarray | None
        R_est: np.ndarray | None
        t_est: np.ndarray | None

        inlier = mask.ravel().astype(bool)
        pts1_i = pts1[inlier]
        pts2_i = pts2[inlier]
        desc_inlier = desc_rows[inlier] if desc_rows.shape[0] else np.empty((0, desc_rows.shape[1]), dtype=desc_rows.dtype)

        def _failure_result(E_store: np.ndarray | None) -> TwoViewResult:
            empty_h = np.zeros((4, 0), np.float64)
            return TwoViewResult(
                frame_i=i,
                frame_j=j,
                pts1=pts1,
                pts2=pts2,
                inlier_mask=mask,
                E=E_store,
                R_est=None,
                t_est=None,
                scale=1.0,
                scale_ok=False,
                X_world_h=empty_h,
                cheiral_mask=np.zeros((0,), bool),
                reproj={},
                descriptors=(
                    np.empty((0, desc_inlier.shape[1]), dtype=desc_inlier.dtype)
                    if desc_inlier.shape[1] > 0
                    else None
                ),
            )

        if pts1_i.shape[0] == 0:
            return _failure_result(essential_from_world_poses(Wi, Wj, self.K))

        if alpha >= 1.0:
            E = essential_from_world_poses(Wi, Wj, self.K)
            X_h, cheiral = triangulate_world_points(pts1_i, pts2_i, Wi, Wj, self.K)
            R_est, t_est = R_gt.copy(), t_gt.copy()
        else:
            if pts1_i.shape[0] < 8:
                return _failure_result(essential_from_world_poses(Wi, Wj, self.K))
            E_fm33 = np.asarray(E_fm, dtype=np.float64).reshape(3, 3)
            R_vis, t_vis, _ = recover_pose_from_essential(E_fm33, pts1_i, pts2_i, self.K, mask=None)
            if alpha > 0.0:
                t_vis = align_translation_direction(t_vis, t_gt)
            R_b, t_b = blend_relative_pose(R_vis, t_vis, R_gt, t_gt, alpha)
            R_est, t_est = R_b, t_b
            E = essential_from_R_t(R_b, t_b)
            X_cam_h, cheiral = triangulate_cam1_frame(pts1_i, pts2_i, self.K, R_b, t_b)
            X_h = cam1_to_world_points(X_cam_h, Wi)
        X_h[:, ~cheiral] = np.nan
        err1 = reprojection_errors(X_h, pts1_i, self.K, Wi)
        err2 = reprojection_errors(X_h, pts2_i, self.K, Wj)
        reproj = summarize_reprojection(err1, err2)

        res = TwoViewResult(
            frame_i=i,
            frame_j=j,
            pts1=pts1,
            pts2=pts2,
            inlier_mask=mask,
            E=E,
            R_est=R_est,
            t_est=t_est,
            scale=scale,
            scale_ok=scale_ok,
            X_world_h=X_h,
            cheiral_mask=cheiral,
            reproj=reproj,
            descriptors=desc_inlier,
        )
        self.pair_results.append(res)
        while len(self.pair_results) > self.window:
            self.pair_results.pop(0)
        self._update_landmarks(i, j, pts1_i, pts2_i, X_h, cheiral)
        return res

    def _update_landmarks(
        self,
        i: int,
        j: int,
        pts1: np.ndarray,
        pts2: np.ndarray,
        X_h: np.ndarray,
        cheiral: np.ndarray,
    ) -> None:
        for k in range(X_h.shape[1]):
            if not cheiral[k] or not np.all(np.isfinite(X_h[:3, k])):
                continue
            lid = self.next_landmark_id
            self.next_landmark_id += 1
            self.landmarks[lid] = X_h[:3, k].copy()
            self.tracks[lid] = {i: tuple(pts1[k]), j: tuple(pts2[k])}

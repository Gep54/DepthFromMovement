from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.config import MotionMode
from pipeline.geometry import (
    essential_from_world_poses,
    estimate_essential_ransac,
    recover_pose_from_essential,
    relative_motion_from_world_poses,
    scale_from_odometry,
    align_translation_direction,
)
from pipeline.matching import match_pair_points
from pipeline.triangulation import triangulate_world_points, triangulate_cam1_frame, cam1_to_world_points
from pipeline.features import detect_and_compute
from pipeline.config import FeatureConfig
from pipeline.metrics import reprojection_errors, summarize_reprojection


@dataclass
class MapConfig:
    motion_mode: MotionMode = "known_pose"
    ransac_epipolar_thresh: float = 1.0
    min_parallax_deg: float = 0.5


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
    ) -> TwoViewResult:
        kpi, di = detect_and_compute(gray_i, self.feat_cfg)
        kpj, dj = detect_and_compute(gray_j, self.feat_cfg)
        pts1, pts2, _ = match_pair_points(kpi, kpj, di, dj, self.feat_cfg)
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
            )

        Wi = self.world_T_camera[i]
        Wj = self.world_T_camera[j]
        R_gt, t_gt = relative_motion_from_world_poses(Wi, Wj)
        scale = 1.0
        scale_ok = True
        E = None
        R_est = None
        t_est = None
        mask = np.ones((len(pts1), 1), np.uint8)

        if self.cfg.motion_mode == "known_pose":
            E = essential_from_world_poses(Wi, Wj, self.K)
            _, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.cfg.ransac_epipolar_thresh,
            )
        else:
            E, mask = estimate_essential_ransac(
                pts1,
                pts2,
                self.K,
                threshold=self.cfg.ransac_epipolar_thresh,
            )
            R_est, t_est, _ = recover_pose_from_essential(E, pts1, pts2, self.K, mask=mask)
            t_est = align_translation_direction(t_est, t_gt)
            scale, scale_ok = scale_from_odometry(t_est, t_gt)

        inlier = mask.ravel().astype(bool)
        pts1_i = pts1[inlier]
        pts2_i = pts2[inlier]
        if self.cfg.motion_mode == "known_pose":
            X_h, cheiral = triangulate_world_points(pts1_i, pts2_i, Wi, Wj, self.K)
        else:
            assert R_est is not None and t_est is not None
            X_cam_h, cheiral = triangulate_cam1_frame(pts1_i, pts2_i, self.K, R_est, t_est)
            if scale_ok:
                X_cam_h = X_cam_h.copy()
                X_cam_h[:3, :] *= scale
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

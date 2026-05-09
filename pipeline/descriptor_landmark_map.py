from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from pipeline.geometry import invert_se3
from pipeline.map import TwoViewResult


def world_point_to_cam0(X_w: np.ndarray, world_T_camera_0: np.ndarray) -> np.ndarray:
    """Map a 3D world point into camera-0 coordinates (``T_c0_w = invert(world_T_camera_0)``)."""
    T_c0_w = invert_se3(np.asarray(world_T_camera_0, dtype=np.float64))
    Xh = np.array([X_w[0], X_w[1], X_w[2], 1.0], dtype=np.float64)
    return (T_c0_w @ Xh)[:3].astype(np.float64)


def descriptor_distance(a: np.ndarray, b: np.ndarray, method: Literal["ORB", "SIFT"]) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    if method == "ORB":
        return float(cv2.norm(aa.astype(np.uint8), bb.astype(np.uint8), cv2.NORM_HAMMING))
    return float(np.linalg.norm(aa.astype(np.float64) - bb.astype(np.float64)))


@dataclass
class DescriptorMapConfig:
    """Landmark fusion settings; ``merge_beta=None`` uses mean-equivalent weights ``1/(n+1)``."""

    method: Literal["ORB", "SIFT"] = "ORB"
    merge_beta: float | None = None
    r"""Fixed EMA beta in (0,1]; ``None`` uses ``1/(n_updates+1)`` (incremental mean)."""
    max_match_distance: float = 64.0
    """ORB: Hamming threshold; SIFT: L2 threshold."""
    ratio_second_best: float | None = None
    """If set, require ``best < ratio * second`` (Lowe-style)."""

    @staticmethod
    def defaults(method: Literal["ORB", "SIFT"]) -> DescriptorMapConfig:
        if method == "ORB":
            return DescriptorMapConfig(method=method, merge_beta=None, max_match_distance=64.0, ratio_second_best=None)
        return DescriptorMapConfig(method=method, merge_beta=None, max_match_distance=220.0, ratio_second_best=None)


@dataclass
class DescriptorLandmark:
    id: int
    position_cam0: np.ndarray
    """(3,) float64 in first-camera frame."""
    n_updates: int
    descriptor: np.ndarray
    best_merge_distance: float = field(default_factory=lambda: float("inf"))


class DescriptorLandmarkMap:
    """Sparse map in camera-0 frame with descriptor NN association."""

    def __init__(self, cfg: DescriptorMapConfig) -> None:
        self.cfg = cfg
        self.landmarks: list[DescriptorLandmark] = []
        self._next_id = 0

    def positions_cam0(self) -> np.ndarray:
        if not self.landmarks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([lm.position_cam0 for lm in self.landmarks], axis=0)

    def _nearest_landmark(self, d_obs: np.ndarray) -> tuple[int, float, float]:
        """Return (index, best_distance, second_best_distance). ``index=-1`` if empty."""
        if not self.landmarks:
            return -1, float("inf"), float("inf")
        best_i = 0
        best_d = descriptor_distance(d_obs, self.landmarks[0].descriptor, self.cfg.method)
        second_d = float("inf")
        for idx in range(1, len(self.landmarks)):
            d = descriptor_distance(d_obs, self.landmarks[idx].descriptor, self.cfg.method)
            if d < best_d:
                second_d = min(second_d, best_d)
                best_d = d
                best_i = idx
            else:
                second_d = min(second_d, d)
        return best_i, best_d, second_d

    def _merge_beta_eff(self, lm: DescriptorLandmark) -> float:
        if self.cfg.merge_beta is not None:
            b = float(self.cfg.merge_beta)
            return float(np.clip(b, 1e-6, 1.0))
        return 1.0 / float(lm.n_updates + 1)

    def integrate(self, tw: TwoViewResult, world_T_camera_0: np.ndarray) -> None:
        """Ingest triangulated points from ``tw``; transform to cam0 before fusion."""
        if tw.descriptors is None or tw.descriptors.shape[0] == 0:
            return
        n = tw.X_world_h.shape[1]
        if tw.descriptors.shape[0] != n:
            raise ValueError(
                f"descriptors rows ({tw.descriptors.shape[0]}) != X columns ({n})"
            )
        W0 = np.asarray(world_T_camera_0, dtype=np.float64)

        for k in range(n):
            if not tw.cheiral_mask[k]:
                continue
            Xw = tw.X_world_h[:3, k]
            if not np.all(np.isfinite(Xw)):
                continue
            X_cam0 = world_point_to_cam0(Xw, W0)
            d_obs = tw.descriptors[k]

            best_i, best_d, second_d = self._nearest_landmark(d_obs)

            reject_ratio = False
            if (
                self.cfg.ratio_second_best is not None
                and np.isfinite(second_d)
                and second_d > 1e-12
                and best_d >= self.cfg.ratio_second_best * second_d
            ):
                reject_ratio = True

            if (
                best_i < 0
                or best_d > self.cfg.max_match_distance
                or reject_ratio
            ):
                lid = self._next_id
                self._next_id += 1
                desc_copy = np.array(d_obs, copy=True, dtype=d_obs.dtype)
                self.landmarks.append(
                    DescriptorLandmark(
                        id=lid,
                        position_cam0=X_cam0.copy(),
                        n_updates=1,
                        descriptor=desc_copy,
                        best_merge_distance=float("inf"),
                    )
                )
                continue

            lm = self.landmarks[best_i]
            beta = self._merge_beta_eff(lm)
            lm.position_cam0 = (1.0 - beta) * lm.position_cam0 + beta * X_cam0
            lm.n_updates += 1

            d_match = descriptor_distance(d_obs, lm.descriptor, self.cfg.method)
            if d_match < lm.best_merge_distance:
                lm.descriptor = np.array(d_obs, copy=True, dtype=d_obs.dtype)
                lm.best_merge_distance = d_match


def export_landmarks_csv(path: str | Path, desc_map: DescriptorLandmarkMap) -> None:
    """Write ``id,x_cam0,y_cam0,z_cam0,n_updates,descriptor_hex``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_cam0", "y_cam0", "z_cam0", "n_updates", "descriptor_hex"])
        for lm in desc_map.landmarks:
            raw = np.asarray(lm.descriptor).tobytes()
            w.writerow(
                [
                    lm.id,
                    f"{lm.position_cam0[0]:.12g}",
                    f"{lm.position_cam0[1]:.12g}",
                    f"{lm.position_cam0[2]:.12g}",
                    lm.n_updates,
                    raw.hex(),
                ]
            )

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from pipeline.geometry import invert_se3
from pipeline.map import TwoViewResult


def within_merge_sphere(p: np.ndarray, q: np.ndarray, radius_m: float) -> bool:
    """True if ``p`` and ``q`` lie within a sphere of radius ``radius_m`` (world frame)."""
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))) <= float(
        radius_m
    )


def project_world_to_camera_uv_z(
    K: np.ndarray, X_world: np.ndarray, world_T_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame points into ``world_T_camera`` (pixels and camera Z)."""
    X = np.asarray(X_world, dtype=np.float64).reshape(-1, 3)
    if X.shape[0] == 0:
        return np.zeros((0, 2), np.float64), np.zeros(0, np.float64)
    Tcw = invert_se3(np.asarray(world_T_camera, dtype=np.float64))
    R = Tcw[:3, :3]
    t = Tcw[:3, 3].reshape(3, 1)
    Xc = (R @ X.T + t).T
    z = Xc[:, 2]
    uv = np.zeros((X.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * (Xc[:, 0] / np.maximum(z, 1e-9)) + K[0, 2]
    uv[:, 1] = K[1, 1] * (Xc[:, 1] / np.maximum(z, 1e-9)) + K[1, 2]
    return uv, z


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
    spatial_merge_radius_m: float | None = None
    """If set, descriptor matches merge only when 3D distance in world <= this value."""

    @staticmethod
    def defaults(method: Literal["ORB", "SIFT"]) -> DescriptorMapConfig:
        if method == "ORB":
            return DescriptorMapConfig(method=method, merge_beta=None, max_match_distance=64.0, ratio_second_best=None)
        return DescriptorMapConfig(method=method, merge_beta=None, max_match_distance=220.0, ratio_second_best=None)


@dataclass
class DescriptorLandmark:
    id: int
    position_world: np.ndarray
    """(3,) float64 in metric world frame (e.g. ``gps_baro_origin``)."""
    n_updates: int
    descriptor: np.ndarray
    best_merge_distance: float = field(default_factory=lambda: float("inf"))


class DescriptorLandmarkMap:
    """Sparse map in metric world frame with descriptor NN association."""

    def __init__(self, cfg: DescriptorMapConfig) -> None:
        self.cfg = cfg
        self.landmarks: list[DescriptorLandmark] = []
        self._next_id = 0

    def positions_world(self) -> np.ndarray:
        if not self.landmarks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([lm.position_world for lm in self.landmarks], axis=0)

    def prune_beyond_range_world(self, max_range_m: float, anchor_world: np.ndarray) -> int:
        """Drop landmarks with ``||position_world - anchor|| > max_range_m``; return count removed."""
        if max_range_m <= 0.0 or not self.landmarks:
            return 0
        anchor = np.asarray(anchor_world, dtype=np.float64).reshape(3)
        kept: list[DescriptorLandmark] = []
        removed = 0
        for lm in self.landmarks:
            if float(np.linalg.norm(lm.position_world - anchor)) > max_range_m:
                removed += 1
            else:
                kept.append(lm)
        self.landmarks = kept
        return removed

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

    def _append_landmark(self, X_world: np.ndarray, d_obs: np.ndarray) -> None:
        lid = self._next_id
        self._next_id += 1
        desc_copy = np.array(d_obs, copy=True, dtype=d_obs.dtype)
        self.landmarks.append(
            DescriptorLandmark(
                id=lid,
                position_world=np.asarray(X_world, dtype=np.float64).reshape(3).copy(),
                n_updates=1,
                descriptor=desc_copy,
                best_merge_distance=float("inf"),
            )
        )

    def integrate(
        self,
        tw: TwoViewResult,
        *,
        anchor_position_world: np.ndarray | None = None,
        max_range_world: float | None = None,
        spatial_merge_radius_m: float | None = None,
    ) -> None:
        """Ingest triangulated world points from ``tw`` (``X_world_h`` already in metric world)."""
        if tw.descriptors is None or tw.descriptors.shape[0] == 0:
            return
        n = tw.X_world_h.shape[1]
        if tw.descriptors.shape[0] != n:
            raise ValueError(
                f"descriptors rows ({tw.descriptors.shape[0]}) != X columns ({n})"
            )
        anchor = (
            np.asarray(anchor_position_world, dtype=np.float64).reshape(3)
            if anchor_position_world is not None
            else None
        )
        radius = (
            spatial_merge_radius_m
            if spatial_merge_radius_m is not None
            else self.cfg.spatial_merge_radius_m
        )

        for k in range(n):
            if not tw.cheiral_mask[k]:
                continue
            Xw = tw.X_world_h[:3, k]
            if not np.all(np.isfinite(Xw)):
                continue
            if anchor is not None and max_range_world is not None:
                if float(np.linalg.norm(Xw - anchor)) > max_range_world:
                    continue
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
                self._append_landmark(Xw, d_obs)
                continue

            lm = self.landmarks[best_i]
            if radius is not None and radius > 0 and not within_merge_sphere(Xw, lm.position_world, radius):
                self._append_landmark(Xw, d_obs)
                continue

            beta = self._merge_beta_eff(lm)
            lm.position_world = (1.0 - beta) * lm.position_world + beta * Xw
            lm.n_updates += 1

            d_match = descriptor_distance(d_obs, lm.descriptor, self.cfg.method)
            if d_match < lm.best_merge_distance:
                lm.descriptor = np.array(d_obs, copy=True, dtype=d_obs.dtype)
                lm.best_merge_distance = d_match


def export_landmarks_csv(path: str | Path, desc_map: DescriptorLandmarkMap) -> None:
    """Write ``id,x_world,y_world,z_world,n_updates,descriptor_hex``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_world", "y_world", "z_world", "n_updates", "descriptor_hex"])
        for lm in desc_map.landmarks:
            raw = np.asarray(lm.descriptor).tobytes()
            w.writerow(
                [
                    lm.id,
                    f"{lm.position_world[0]:.12g}",
                    f"{lm.position_world[1]:.12g}",
                    f"{lm.position_world[2]:.12g}",
                    lm.n_updates,
                    raw.hex(),
                ]
            )

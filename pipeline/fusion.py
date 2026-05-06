from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.map import TwoViewResult


@dataclass
class FusedLandmark:
    """Single fused 3D point with multi-frame image observations."""

    position_world: np.ndarray
    """Running mean (3,) in world coordinates."""
    n_updates: int
    """Number of triangulation samples fused into ``position_world``."""
    observations: dict[int, tuple[float, float]] = field(default_factory=dict)
    """frame_index → (u, v) pixel observation (latest merged estimate)."""


class FusedLandmarkMap:
    """
    Fuse two-view landmarks across consecutive edges by **shared observations**:

    if a new point sees frame ``f`` near ``(u,v)`` (within ``merge_px`` pixels) as an
    existing landmark, treat it as the same 3D point and average world coordinates.
    """

    def __init__(self, merge_px: float = 4.0) -> None:
        self.merge_px = float(merge_px)
        self.landmarks: list[FusedLandmark] = []

    def _pix_dist(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float(np.hypot(dx, dy))

    def _shares_observation(self, obs_new: list[tuple[int, float, float]], lm: FusedLandmark) -> bool:
        for f, u, v in obs_new:
            if f not in lm.observations:
                continue
            if self._pix_dist((u, v), lm.observations[f]) <= self.merge_px:
                return True
        return False

    def _merge_observations(self, lm: FusedLandmark, obs_new: list[tuple[int, float, float]]) -> None:
        for f, u, v in obs_new:
            if f not in lm.observations:
                lm.observations[f] = (u, v)
                continue
            ou, ov = lm.observations[f]
            lm.observations[f] = ((ou + u) * 0.5, (ov + v) * 0.5)

    def integrate_two_view_result(self, tw: TwoViewResult) -> None:
        inl = tw.inlier_mask.ravel().astype(bool)
        p1 = tw.pts1[inl]
        p2 = tw.pts2[inl]
        fi, fj = tw.frame_i, tw.frame_j

        for k in range(tw.X_world_h.shape[1]):
            if not tw.cheiral_mask[k]:
                continue
            X = tw.X_world_h[:3, k].astype(np.float64)
            if not np.all(np.isfinite(X)):
                continue
            obs: list[tuple[int, float, float]] = [
                (fi, float(p1[k, 0]), float(p1[k, 1])),
                (fj, float(p2[k, 0]), float(p2[k, 1])),
            ]

            merged_into: FusedLandmark | None = None
            for lm in self.landmarks:
                if self._shares_observation(obs, lm):
                    merged_into = lm
                    break

            if merged_into is None:
                self.landmarks.append(
                    FusedLandmark(position_world=X.copy(), n_updates=1, observations=dict())
                )
                self._merge_observations(self.landmarks[-1], obs)
                continue

            n = merged_into.n_updates
            merged_into.position_world = (merged_into.position_world * n + X) / (n + 1.0)
            merged_into.n_updates = n + 1
            self._merge_observations(merged_into, obs)

    def positions_xyz(self) -> np.ndarray:
        if not self.landmarks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([lm.position_world for lm in self.landmarks], axis=0)

    def statistics(self) -> dict[str, int]:
        return {"n_fused_landmarks": len(self.landmarks)}


def fused_world_points_homogeneous(fused: FusedLandmarkMap) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous (4, N) and all-True mask for reuse with projection helpers."""
    n = len(fused.landmarks)
    if n == 0:
        return np.zeros((4, 0), np.float64), np.zeros((0,), bool)
    X = np.ones((4, n), dtype=np.float64)
    for i, lm in enumerate(fused.landmarks):
        X[:3, i] = lm.position_world
    return X, np.ones(n, dtype=bool)

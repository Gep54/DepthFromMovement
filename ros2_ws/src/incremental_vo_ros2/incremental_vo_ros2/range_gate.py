"""Pure numpy helpers for sparse-map range gating (no ROS imports)."""

from __future__ import annotations

import numpy as np


def consecutive_keyframe_baseline_m(Wi: np.ndarray, Wj: np.ndarray) -> float:
    """Translation norm between consecutive keyframe ``world_T_camera`` poses."""
    ti = np.asarray(Wi, dtype=np.float64)[:3, 3]
    tj = np.asarray(Wj, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(tj - ti))


def max_sparse_range_m(
    baseline_m: float,
    factor: float,
    *,
    min_baseline_m: float = 1e-3,
) -> float | None:
    """``factor * max(baseline_m, min_baseline_m)``; ``None`` if ``factor <= 0`` (disabled)."""
    if factor <= 0.0:
        return None
    b = max(float(baseline_m), float(min_baseline_m))
    return float(factor) * b


def should_reset_bag_replay(last_stamp_s: float | None, new_stamp_s: float, jump_s: float) -> bool:
    """True when ``new_stamp_s`` jumps backward by more than ``jump_s`` (rosbag loop/restart)."""
    if last_stamp_s is None:
        return False
    return new_stamp_s < last_stamp_s - float(jump_s)


def distance_from_anchor_world(X_world: np.ndarray, anchor_xyz: np.ndarray) -> float:
    """Euclidean distance from a world point to an anchor (e.g. current camera position)."""
    return float(
        np.linalg.norm(
            np.asarray(X_world, dtype=np.float64).reshape(3) - np.asarray(anchor_xyz, dtype=np.float64).reshape(3)
        )
    )

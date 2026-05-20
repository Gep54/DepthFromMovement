"""Baseline-scaled max-range gating for triangulated 3D points."""

from __future__ import annotations

import numpy as np


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


def apply_max_range_gate_cam1(
    X_cam_h: np.ndarray,
    cheiral_mask: np.ndarray,
    baseline_m: float,
    factor: float,
    *,
    min_baseline_m: float = 1e-3,
) -> np.ndarray:
    """
    After cheirality, drop points with ``||X_cam1||^2 > max_range^2``.

    ``max_range = factor * max(baseline_m, min_baseline_m)``.
    Returns an updated boolean mask (same length as columns of ``X_cam_h``).
    """
    max_range = max_sparse_range_m(
        baseline_m, factor, min_baseline_m=min_baseline_m
    )
    if max_range is None:
        return np.asarray(cheiral_mask, dtype=bool).copy()
    cheiral = np.asarray(cheiral_mask, dtype=bool).copy()
    if not cheiral.any():
        return cheiral
    Xc = np.asarray(X_cam_h[:3, :], dtype=np.float64)
    sq_dist = np.sum(Xc * Xc, axis=0)
    max_sq = max_range * max_range
    cheiral &= sq_dist <= max_sq
    return cheiral

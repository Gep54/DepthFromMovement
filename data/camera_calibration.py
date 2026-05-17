"""Build :class:`Calibration` from flat intrinsics (ROS CameraInfo, etc.)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data.schema import Calibration

_SUPPORTED_DISTORTION_MODELS = frozenset({"", "plumb_bob", "pinhole"})


def calibration_from_intrinsics(
    *,
    K_flat: Sequence[float],
    D: Sequence[float] | None,
    width: int,
    height: int,
    distortion_model: str = "",
) -> Calibration:
    """
    Parse row-major 3×3 ``K`` and optional distortion into :class:`Calibration`.

    ``D`` uses OpenCV plumb_bob order (k1, k2, p1, p2, k3). Unknown
    ``distortion_model`` values yield zero distortion with a warning implied by caller.
    """
    k_arr = np.asarray(list(K_flat), dtype=np.float64).ravel()
    if k_arr.size != 9:
        raise ValueError(f"K must have 9 elements, got {k_arr.size}")
    if not np.all(np.isfinite(k_arr)):
        raise ValueError("K must contain finite values")
    K = k_arr.reshape(3, 3)

    model = (distortion_model or "").strip().lower()
    dist_arr: np.ndarray | None = None
    if D is not None and len(D) > 0:
        d = np.asarray(list(D), dtype=np.float64).reshape(-1)
        if np.linalg.norm(d) >= 1e-9:
            if model not in _SUPPORTED_DISTORTION_MODELS:
                d = np.zeros(0, dtype=np.float64)
            else:
                dist_arr = d

    image_size: tuple[int, int] | None = None
    w, h = int(width), int(height)
    if w > 0 and h > 0:
        image_size = (w, h)

    return Calibration(K=K, dist_coeffs=dist_arr, image_size=image_size)


def distortion_model_supported(distortion_model: str) -> bool:
    """Whether ``distortion_model`` is handled when D is non-zero."""
    return (distortion_model or "").strip().lower() in _SUPPORTED_DISTORTION_MODELS

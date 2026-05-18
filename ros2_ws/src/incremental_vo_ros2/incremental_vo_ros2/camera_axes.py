"""Fixed camera-frame axis remaps (no ROS message imports)."""

from __future__ import annotations

import numpy as np

# OpenCV / REP-103 optical → body/FCU (race bag): cam Z→body X, Y→body Z, X→body Y.
CAMERA_AXES_SWAP_OPENCV_TO_BODY_RACE = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def camera_axes_swap_rotation(name: str) -> np.ndarray | None:
    """Return 3×3 ``R_swap`` for ``world_T_camera' = world_T_camera @ homog(R_swap)``, or ``None``."""
    key = (name or "").strip().lower()
    if key in ("", "none", "off", "false", "0"):
        return None
    if key in ("opencv_to_body_race", "race", "race_fcu", "zyx_to_xyz"):
        return CAMERA_AXES_SWAP_OPENCV_TO_BODY_RACE.copy()
    raise ValueError(
        f"Unknown camera_axes_swap={name!r}; use none or opencv_to_body_race"
    )


def apply_camera_axes_swap(world_T_camera: np.ndarray, swap_R: np.ndarray | None) -> np.ndarray:
    """``X_world = T @ R_swap @ X_cam`` — post-multiply on ``world_T_camera`` (cam→world)."""
    if swap_R is None:
        return np.asarray(world_T_camera, dtype=np.float64)
    T = np.asarray(world_T_camera, dtype=np.float64).copy()
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] = np.asarray(swap_R, dtype=np.float64)[:3, :3]
    return T @ S

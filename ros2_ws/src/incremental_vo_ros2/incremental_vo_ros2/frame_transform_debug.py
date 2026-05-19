"""SE(3) formatting and camera–drone chain helpers (no ROS message imports)."""

from __future__ import annotations

import numpy as np


def format_se3_4x4(T: np.ndarray, *, precision: int = 4) -> str:
    """Format a 4×4 SE(3) matrix as fixed-width rows for log output."""
    M = np.asarray(T, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got shape {M.shape}")
    width = precision + 8
    lines: list[str] = []
    for row in range(4):
        cells = " ".join(f"{M[row, col]:+{width}.{precision}f}" for col in range(4))
        lines.append(f"  | {cells} |")
    return "\n".join(lines)


def drone_T_camera_from_world_poses(
    world_T_drone: np.ndarray, world_T_camera: np.ndarray
) -> np.ndarray:
    """Camera → drone: ``X_drone = drone_T_camera @ X_cam`` (matches descriptor map chain)."""
    from pipeline.geometry import invert_se3

    W_drone = np.asarray(world_T_drone, dtype=np.float64)
    W_cam = np.asarray(world_T_camera, dtype=np.float64)
    return invert_se3(W_drone) @ W_cam

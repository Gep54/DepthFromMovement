"""Helpers for sampling and formatting triangulated points that pass map-ingest gates."""

from __future__ import annotations

import numpy as np

from pipeline.descriptor_landmark_map import (
    _homog_point3,
    distance_from_anchor,
    point_camera_to_drone_to_world,
)
from pipeline.geometry import invert_se3
from pipeline.map import TwoViewResult


def x_cam_column(tw: TwoViewResult, k: int) -> np.ndarray | None:
    """Camera-frame 3D point for column ``k``; ``None`` if non-finite."""
    if tw.X_cam_h is not None and tw.X_cam_h.shape[1] > k:
        Xc = tw.X_cam_h[:3, k]
        if np.all(np.isfinite(Xc)):
            return Xc.astype(np.float64)
    Xw = tw.X_world_h[:3, k]
    if not np.all(np.isfinite(Xw)):
        return None
    return Xw.astype(np.float64)


def valid_integrate_indices(
    tw: TwoViewResult,
    *,
    world_T_camera_raw: np.ndarray,
    world_T_drone_raw: np.ndarray,
    world_T_camera_j_raw: np.ndarray,
    max_range_world: float | None,
) -> list[int]:
    """Column indices that would pass ``DescriptorLandmarkMap.integrate`` ingest gates."""
    n = tw.X_world_h.shape[1]
    W_cam = np.asarray(world_T_camera_raw, dtype=np.float64)
    W_drone = np.asarray(world_T_drone_raw, dtype=np.float64)
    W_cam_j = np.asarray(world_T_camera_j_raw, dtype=np.float64)
    range_anchor = W_cam_j[:3, 3].copy()
    valid: list[int] = []
    for k in range(n):
        if not tw.cheiral_mask[k]:
            continue
        X_cam = x_cam_column(tw, k)
        if X_cam is None:
            continue
        X_world = point_camera_to_drone_to_world(X_cam, W_cam, W_drone)
        if max_range_world is not None and distance_from_anchor(X_world, range_anchor) > max_range_world:
            continue
        valid.append(k)
    return valid


def triangulation_point_coords(
    tw: TwoViewResult,
    k: int,
    *,
    world_T_camera_raw: np.ndarray,
    world_T_drone_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_cam, X_drone, X_world)`` for column ``k``."""
    X_cam = x_cam_column(tw, k)
    if X_cam is None:
        raise ValueError(f"column {k} has no finite camera-frame point")
    W_cam = np.asarray(world_T_camera_raw, dtype=np.float64)
    W_drone = np.asarray(world_T_drone_raw, dtype=np.float64)
    X_world = point_camera_to_drone_to_world(X_cam, W_cam, W_drone)
    T_w_d = invert_se3(W_drone)
    X_drone = (T_w_d @ _homog_point3(X_world))[:3]
    return X_cam, X_drone, X_world


def format_triangulation_debug_line(
    *,
    keyframe_idx: int,
    frame_i: int,
    frame_j: int,
    col_k: int,
    X_cam: np.ndarray,
    X_drone: np.ndarray,
    X_world: np.ndarray,
    precision: int = 4,
) -> str:
    """One-line summary for ROS / offline logging."""
    fmt = f"{{:.{precision}f}}"
    cam_s = ", ".join(fmt.format(v) for v in X_cam)
    drone_s = ", ".join(fmt.format(v) for v in X_drone)
    world_s = ", ".join(fmt.format(v) for v in X_world)
    return (
        f"Triang debug kf={keyframe_idx} pair={frame_i}->{frame_j} k={col_k} "
        f"cam=[{cam_s}] drone=[{drone_s}] world=[{world_s}]"
    )


def sample_random_integrate_point(
    tw: TwoViewResult,
    *,
    world_T_camera_raw: np.ndarray,
    world_T_drone_raw: np.ndarray,
    world_T_camera_j_raw: np.ndarray,
    max_range_world: float | None,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pick a random valid column; return ``(k, X_cam, X_drone, X_world)`` or ``None``."""
    valid = valid_integrate_indices(
        tw,
        world_T_camera_raw=world_T_camera_raw,
        world_T_drone_raw=world_T_drone_raw,
        world_T_camera_j_raw=world_T_camera_j_raw,
        max_range_world=max_range_world,
    )
    if not valid:
        return None
    gen = rng if rng is not None else np.random.default_rng()
    k = int(gen.choice(valid))
    X_cam, X_drone, X_world = triangulation_point_coords(
        tw,
        k,
        world_T_camera_raw=world_T_camera_raw,
        world_T_drone_raw=world_T_drone_raw,
    )
    return k, X_cam, X_drone, X_world

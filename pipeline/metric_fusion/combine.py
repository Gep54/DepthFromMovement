"""Frame-wise fusion of odometry vs externally provided camera poses (numpy only)."""

from __future__ import annotations

import numpy as np


def fused_pose_from_pair(
    T_odom: np.ndarray,
    T_provided: np.ndarray | None,
    method: str,
    *,
    position_blend_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine two ``world_T_camera`` estimates for the same time index.

    Parameters
    ----------
    T_odom
        4×4 camera→world from odometry (or primary metric track).
    T_provided
        Optional 4×4 from a second source (e.g. optical-flow–derived pose upstream).
    method
        ``odom_only`` | ``provided_if_available`` | ``position_blend``.
    position_blend_weight
        For ``position_blend``: weight on **provided** translation; odometry gets ``1 - w``.
        Rotation is always taken from odometry. If ``T_provided`` is missing, odometry is used.
    """
    To = np.asarray(T_odom, dtype=np.float64).reshape(4, 4)
    if method == "odom_only":
        return To.copy()
    if method == "provided_if_available":
        if T_provided is None:
            return To.copy()
        Tp = np.asarray(T_provided, dtype=np.float64).reshape(4, 4)
        return Tp.copy()
    if method == "position_blend":
        w = float(np.clip(position_blend_weight, 0.0, 1.0))
        R = To[:3, :3].copy()
        t_odom = To[:3, 3].copy()
        if T_provided is None or w <= 0.0:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t_odom
            return T
        Tp = np.asarray(T_provided, dtype=np.float64).reshape(4, 4)
        t_prov = Tp[:3, 3].copy()
        t = (1.0 - w) * t_odom + w * t_prov
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    raise ValueError(f"unknown fusion method {method!r}")


def fuse_pose_sequence(
    odom_poses: list[np.ndarray],
    provided_poses: list[np.ndarray] | None,
    method: str,
    *,
    position_blend_weight: float = 0.5,
) -> list[np.ndarray]:
    """Align by index: ``provided_poses[i]`` pairs with ``odom_poses[i]`` when present."""
    n = len(odom_poses)
    if str(method).strip().lower().replace("-", "_") == "ekf_pose_velocity":
        raise ValueError(
            "fusion method 'ekf_pose_velocity' is only for ROS streaming, not offline fuse_pose_sequence"
        )
    if provided_poses is not None and len(provided_poses) != n:
        raise ValueError(
            f"provided_poses length {len(provided_poses)} != odom_poses length {n}"
        )
    out: list[np.ndarray] = []
    for i in range(n):
        prov = None if provided_poses is None else provided_poses[i]
        out.append(
            fused_pose_from_pair(
                odom_poses[i],
                prov,
                method,
                position_blend_weight=position_blend_weight,
            )
        )
    return out

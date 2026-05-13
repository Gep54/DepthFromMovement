"""Stateful metric pose fusion for streaming (ROS): last odom + last provided → fused SE(3).

Additional strategies (e.g. EKF coupling odometry increments with velocity measurements) can be
implemented as new subclasses of ``MetricPoseFusion`` without changing ``IncrementalMap``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pipeline.metric_fusion.combine import fused_pose_from_pair


class MetricPoseFusion(ABC):
    """Pluggable fusion of odometry pose vs externally provided ``world_T_camera``."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal buffers (new run / bag rewind)."""

    @abstractmethod
    def push_odom_world_T_camera(
        self, T: np.ndarray, stamp: tuple[int, int] | None = None
    ) -> None:
        """Update with latest odometry-derived camera→world pose."""

    @abstractmethod
    def push_provided_world_T_camera(
        self, T: np.ndarray | None, stamp: tuple[int, int] | None = None
    ) -> None:
        """Update optional second track (full SE(3)); ``None`` if no message."""

    @abstractmethod
    def fused_world_T_camera(self) -> np.ndarray:
        """Current fused 4×4; must match ``pipeline`` / dataset ``world_T_camera`` convention."""

    def fused_position_xyz(self) -> np.ndarray:
        """Translation of ``fused_world_T_camera()``."""
        return self.fused_world_T_camera()[:3, 3].copy()

    def push_body_velocity(self, v_b: np.ndarray, stamp: tuple[int, int] | None = None) -> None:
        """Optional body-frame linear velocity (3,). Default no-op for snapshot fusion."""
        del v_b, stamp


class OdomOnlyFusion(MetricPoseFusion):
    """Ignore provided track; identical to pre-fusion ROS behaviour."""

    def __init__(self) -> None:
        self._T_odom = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        self._T_odom = np.eye(4, dtype=np.float64)

    def push_odom_world_T_camera(
        self, T: np.ndarray, stamp: tuple[int, int] | None = None
    ) -> None:
        self._T_odom = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()

    def push_provided_world_T_camera(
        self, T: np.ndarray | None, stamp: tuple[int, int] | None = None
    ) -> None:
        pass

    def fused_world_T_camera(self) -> np.ndarray:
        return self._T_odom.copy()


class StatefulPairFusion(MetricPoseFusion):
    """Keeps last odom and last provided; combines via :func:`fused_pose_from_pair`."""

    def __init__(self, method: str, *, position_blend_weight: float = 0.5) -> None:
        self._method = method
        self._position_blend_weight = float(position_blend_weight)
        self._T_odom = np.eye(4, dtype=np.float64)
        self._T_provided: np.ndarray | None = None

    def reset(self) -> None:
        self._T_odom = np.eye(4, dtype=np.float64)
        self._T_provided = None

    def push_odom_world_T_camera(
        self, T: np.ndarray, stamp: tuple[int, int] | None = None
    ) -> None:
        self._T_odom = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()

    def push_provided_world_T_camera(
        self, T: np.ndarray | None, stamp: tuple[int, int] | None = None
    ) -> None:
        if T is None:
            return
        self._T_provided = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()

    def fused_world_T_camera(self) -> np.ndarray:
        return fused_pose_from_pair(
            self._T_odom,
            self._T_provided,
            self._method,
            position_blend_weight=self._position_blend_weight,
        )


def create_metric_pose_fusion(
    name: str,
    *,
    position_blend_weight: float = 0.5,
    ekf_sigma_process_pos: float = 0.05,
    ekf_sigma_process_vel: float = 0.5,
    ekf_sigma_odom_position: float = 0.02,
    ekf_sigma_velocity: float = 0.3,
    ekf_sigma_vo_position: float = 2.0,
) -> MetricPoseFusion:
    """
    Instantiate a streaming fusion strategy by registry name.

    Names: ``odom_only``, ``provided_if_available``, ``position_blend``, ``ekf_pose_velocity``.
    """
    key = name.strip().lower().replace("-", "_")
    if key == "odom_only":
        return OdomOnlyFusion()
    if key == "provided_if_available":
        return StatefulPairFusion("provided_if_available")
    if key == "position_blend":
        return StatefulPairFusion("position_blend", position_blend_weight=position_blend_weight)
    if key == "ekf_pose_velocity":
        from pipeline.metric_fusion.ekf_pose_velocity import EkfPoseVelocityFusion

        return EkfPoseVelocityFusion(
            sigma_process_pos=ekf_sigma_process_pos,
            sigma_process_vel=ekf_sigma_process_vel,
            sigma_odom_position=ekf_sigma_odom_position,
            sigma_velocity=ekf_sigma_velocity,
            sigma_vo_position=ekf_sigma_vo_position,
        )
    known = ", ".join(sorted(list_registered_metric_fusion_methods()))
    raise ValueError(f"unknown fusion_method {name!r}; expected one of: {known}")


def list_registered_metric_fusion_methods() -> tuple[str, ...]:
    return ("ekf_pose_velocity", "odom_only", "provided_if_available", "position_blend")

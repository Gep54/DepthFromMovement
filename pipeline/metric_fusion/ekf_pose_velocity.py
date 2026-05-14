"""Constant-velocity EKF on world position + velocity; odometry pose, body velocity, VO increment."""

from __future__ import annotations

import numpy as np

from pipeline.metric_fusion.base import MetricPoseFusion


def _stamp_to_ns(stamp: tuple[int, int] | None) -> int | None:
    if stamp is None:
        return None
    sec, nsec = int(stamp[0]), int(stamp[1])
    return sec * 1_000_000_000 + nsec


class EkfPoseVelocityFusion(MetricPoseFusion):
    """
    State x = [p_w (3), v_w (3)]; constant-velocity predict; updates:

    - Odometry: position z = t_odom (3), H = [I 0], R small.
    - Body velocity: z = R_odom v_b in world, H = [0 I].
    - VO keyframe (optional): soft position z = p_i + u * ||Δp_odo|| with large R,
      where u is VO translation direction in world (from two-view ``t_est``).
    """

    def __init__(
        self,
        *,
        sigma_process_pos: float = 0.05,
        sigma_process_vel: float = 0.5,
        sigma_odom_position: float = 0.02,
        sigma_velocity: float = 0.3,
        sigma_vo_position: float = 2.0,
    ) -> None:
        self._sq = float(sigma_process_pos) ** 2
        self._svq = float(sigma_process_vel) ** 2
        self._so2 = float(sigma_odom_position) ** 2
        self._sv2 = float(sigma_velocity) ** 2
        self._svo2 = float(sigma_vo_position) ** 2

        self._x = np.zeros(6, dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64) * 100.0
        self._initialized = False
        self._last_stamp_ns: int | None = None
        self._R_odom = np.eye(3, dtype=np.float64)
        self._T_odom_full = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        self._x[:] = 0.0
        self._P = np.eye(6, dtype=np.float64) * 100.0
        self._initialized = False
        self._last_stamp_ns = None
        self._R_odom = np.eye(3, dtype=np.float64)
        self._T_odom_full = np.eye(4, dtype=np.float64)

    def _predict_to(self, stamp_ns: int | None) -> None:
        if stamp_ns is None or self._last_stamp_ns is None:
            if stamp_ns is not None:
                self._last_stamp_ns = stamp_ns
            return
        dt = (stamp_ns - self._last_stamp_ns) * 1e-9
        self._last_stamp_ns = stamp_ns
        if dt <= 0.0:
            return
        # Constant velocity: p += v dt
        F = np.eye(6, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        self._x = F @ self._x
        q_p = self._sq * max(dt, 1e-6)
        q_v = self._svq * max(dt, 1e-6)
        Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v]).astype(np.float64)
        self._P = F @ self._P @ F.T + Q

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Linear Gaussian measurement update (Joseph form for P)."""
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        I = np.eye(6, dtype=np.float64)
        self._x = self._x + K @ y
        I_KH = I - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

    def push_odom_world_T_camera(
        self, T: np.ndarray, stamp: tuple[int, int] | None = None
    ) -> None:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        self._R_odom = T[:3, :3].copy()
        self._T_odom_full = T.copy()
        stamp_ns = _stamp_to_ns(stamp)

        if not self._initialized:
            self._x[:3] = T[:3, 3].copy()
            self._x[3:6] = 0.0
            self._P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).astype(np.float64)
            self._initialized = True
            self._last_stamp_ns = stamp_ns
            return

        self._predict_to(stamp_ns)
        z = T[:3, 3].copy()
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, 0:3] = np.eye(3)
        R = np.eye(3, dtype=np.float64) * self._so2
        self._kalman_update(z, H, R)

    def push_provided_world_T_camera(
        self, T: np.ndarray | None, stamp: tuple[int, int] | None = None
    ) -> None:
        del T, stamp

    def push_body_velocity(self, v_b: np.ndarray, stamp: tuple[int, int] | None = None) -> None:
        if not self._initialized:
            return
        vb = np.asarray(v_b, dtype=np.float64).reshape(3)
        stamp_ns = _stamp_to_ns(stamp)
        self._predict_to(stamp_ns)
        z = self._R_odom @ vb
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, 3:6] = np.eye(3)
        R = np.eye(3, dtype=np.float64) * self._sv2
        self._kalman_update(z, H, R)

    def ingest_vo_keyframe_increment(
        self,
        Wi: np.ndarray,
        Wj: np.ndarray,
        _R_est: np.ndarray | None,
        t_est: np.ndarray | None,
        stamp: tuple[int, int] | None = None,
    ) -> None:
        """
        Soft position measurement from two-view translation direction scaled by odometry step.

        ``t_est`` is in recoverPose / cam1 convention relative to the pair; map direction to world
        using the previous keyframe rotation ``Wi[:3,:3]``.
        """
        if not self._initialized or t_est is None:
            return
        Wi = np.asarray(Wi, dtype=np.float64).reshape(4, 4)
        Wj = np.asarray(Wj, dtype=np.float64).reshape(4, 4)
        tv = np.asarray(t_est, dtype=np.float64).reshape(3)
        nt = float(np.linalg.norm(tv))
        if nt < 1e-9:
            return
        d_odo = Wj[:3, 3] - Wi[:3, 3]
        n_odo = float(np.linalg.norm(d_odo))
        u = Wi[:3, :3] @ (tv / nt)
        p_meas = Wi[:3, 3] + u * n_odo
        stamp_ns = _stamp_to_ns(stamp)
        self._predict_to(stamp_ns)
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, 0:3] = np.eye(3)
        R = np.eye(3, dtype=np.float64) * self._svo2
        self._kalman_update(p_meas, H, R)

    def fused_world_T_camera(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        if self._initialized:
            T[:3, 3] = self._x[:3].copy()
        T[:3, :3] = self._R_odom.copy()
        return T

from __future__ import annotations

import numpy as np
import pytest

from pipeline.metric_fusion import create_metric_pose_fusion
from pipeline.metric_fusion.ekf_pose_velocity import EkfPoseVelocityFusion


def test_ekf_init_and_odom_only_matches_position() -> None:
    f = EkfPoseVelocityFusion(
        sigma_process_pos=0.01,
        sigma_process_vel=0.1,
        sigma_odom_position=1e-6,
        sigma_velocity=1.0,
        sigma_vo_position=10.0,
    )
    T0 = np.eye(4)
    T0[:3, 3] = [1.0, 2.0, 3.0]
    f.push_odom_world_T_camera(T0, (0, 0))
    np.testing.assert_allclose(f.fused_position_xyz(), [1.0, 2.0, 3.0])


def test_ekf_velocity_update_rotates_body_to_world() -> None:
    f = EkfPoseVelocityFusion(
        sigma_process_pos=0.001,
        sigma_process_vel=0.001,
        sigma_odom_position=1e-6,
        sigma_velocity=0.01,
        sigma_vo_position=10.0,
    )
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.0]
    T[0, 0] = 0.0
    T[0, 1] = -1.0
    T[1, 0] = 1.0
    T[1, 1] = 0.0
    f.push_odom_world_T_camera(T, (0, 0))
    # Body x=1 -> world y=1 (90 deg yaw about z)
    f.push_body_velocity(np.array([1.0, 0.0, 0.0]), (0, int(1e8)))
    np.testing.assert_allclose(f._x[3:6], [0.0, 1.0, 0.0], atol=0.05)


def test_ekf_vo_increment_shifts_toward_vo_direction() -> None:
    f = EkfPoseVelocityFusion(
        sigma_process_pos=0.001,
        sigma_process_vel=0.001,
        sigma_odom_position=1e-6,
        sigma_velocity=1.0,
        sigma_vo_position=0.25,
    )
    Wi = np.eye(4)
    Wj = np.eye(4)
    Wj[0, 3] = 1.0
    f.push_odom_world_T_camera(Wj.copy(), (1, 0))
    t_est = np.array([[1.0], [0.0], [0.0]])
    f.ingest_vo_keyframe_increment(Wi, Wj, None, t_est, (1, 0))
    # VO direction matches odom step; position should stay near odom
    np.testing.assert_allclose(f.fused_position_xyz(), [1.0, 0.0, 0.0], atol=0.1)


def test_create_metric_pose_fusion_registers_ekf() -> None:
    g = create_metric_pose_fusion("ekf_pose_velocity", ekf_sigma_odom_position=0.05)
    assert isinstance(g, EkfPoseVelocityFusion)

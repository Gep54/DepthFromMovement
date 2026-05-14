from __future__ import annotations

import numpy as np
import pytest

from pipeline.metric_fusion import (
    OdomOnlyFusion,
    create_metric_pose_fusion,
    fuse_pose_sequence,
    fused_pose_from_pair,
)


def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def test_fused_pose_odom_only_ignores_provided() -> None:
    To = _T_from_Rt(np.eye(3), np.array([1.0, 2.0, 3.0]))
    Tp = _T_from_Rt(np.eye(3), np.array([9.0, 9.0, 9.0]))
    out = fused_pose_from_pair(To, Tp, "odom_only")
    np.testing.assert_allclose(out[:3, 3], [1.0, 2.0, 3.0])


def test_fused_pose_provided_if_available() -> None:
    To = _T_from_Rt(np.eye(3), np.array([1.0, 0.0, 0.0]))
    Tp = _T_from_Rt(np.eye(3), np.array([5.0, 0.0, 0.0]))
    assert fused_pose_from_pair(To, None, "provided_if_available") is not None
    out_none = fused_pose_from_pair(To, None, "provided_if_available")
    np.testing.assert_allclose(out_none[:3, 3], [1.0, 0.0, 0.0])
    out = fused_pose_from_pair(To, Tp, "provided_if_available")
    np.testing.assert_allclose(out[:3, 3], [5.0, 0.0, 0.0])


def test_fused_pose_position_blend() -> None:
    To = _T_from_Rt(np.eye(3), np.array([0.0, 0.0, 0.0]))
    Tp = _T_from_Rt(np.eye(3), np.array([10.0, 0.0, 0.0]))
    out = fused_pose_from_pair(To, Tp, "position_blend", position_blend_weight=0.25)
    np.testing.assert_allclose(out[:3, 3], [2.5, 0.0, 0.0])
    out2 = fused_pose_from_pair(To, Tp, "position_blend", position_blend_weight=1.0)
    np.testing.assert_allclose(out2[:3, 3], [10.0, 0.0, 0.0])


def test_fuse_pose_sequence_rejects_ekf() -> None:
    odom = [np.eye(4)]
    with pytest.raises(ValueError, match="ROS streaming"):
        fuse_pose_sequence(odom, None, "ekf_pose_velocity")


def test_fuse_pose_sequence_length_mismatch() -> None:
    odom = [np.eye(4), np.eye(4)]
    prov = [np.eye(4)]
    with pytest.raises(ValueError, match="length"):
        fuse_pose_sequence(odom, prov, "odom_only")


def test_streaming_odom_only() -> None:
    f = OdomOnlyFusion()
    f.push_odom_world_T_camera(_T_from_Rt(np.eye(3), np.array([0.0, 1.0, 0.0])))
    f.push_provided_world_T_camera(_T_from_Rt(np.eye(3), np.array([99.0, 99.0, 99.0])))
    np.testing.assert_allclose(f.fused_position_xyz(), [0.0, 1.0, 0.0])


def test_create_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown fusion_method"):
        create_metric_pose_fusion("ekf_not_implemented")

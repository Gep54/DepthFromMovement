"""Pluggable fusion of metric camera poses (odometry vs external provided track)."""

from pipeline.metric_fusion.base import (
    MetricPoseFusion,
    OdomOnlyFusion,
    StatefulPairFusion,
    create_metric_pose_fusion,
    list_registered_metric_fusion_methods,
)
from pipeline.metric_fusion.combine import fuse_pose_sequence, fused_pose_from_pair
from pipeline.metric_fusion.ekf_pose_velocity import EkfPoseVelocityFusion

__all__ = [
    "MetricPoseFusion",
    "OdomOnlyFusion",
    "StatefulPairFusion",
    "EkfPoseVelocityFusion",
    "create_metric_pose_fusion",
    "list_registered_metric_fusion_methods",
    "fused_pose_from_pair",
    "fuse_pose_sequence",
]

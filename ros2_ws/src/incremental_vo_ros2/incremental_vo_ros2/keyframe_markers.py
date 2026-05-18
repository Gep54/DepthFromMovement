"""RViz markers for keyframe debug (camera +Z arrows in world frame)."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from incremental_vo_ros2.se3 import camera_z_arrow_endpoints_world

MARKER_NAMESPACE = "keyframe_camera_z"


def make_camera_z_arrow_marker(
    *,
    frame_id: str,
    stamp,
    keyframe_index: int,
    world_T_camera: np.ndarray,
    length_m: float,
    shaft_diameter_m: float = 0.04,
    head_diameter_m: float = 0.08,
    color: tuple[float, float, float, float] = (0.1, 0.85, 0.2, 0.95),
) -> Marker:
    """Build a single ``visualization_msgs/Marker`` ``ARROW`` along camera +Z."""
    tail, head = camera_z_arrow_endpoints_world(world_T_camera, length_m)
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = MARKER_NAMESPACE
    m.id = int(keyframe_index)
    m.type = Marker.ARROW
    m.action = Marker.ADD
    p0 = Point()
    p0.x, p0.y, p0.z = float(tail[0]), float(tail[1]), float(tail[2])
    p1 = Point()
    p1.x, p1.y, p1.z = float(head[0]), float(head[1]), float(head[2])
    m.points = [p0, p1]
    m.scale.x = float(shaft_diameter_m)
    m.scale.y = float(head_diameter_m)
    m.scale.z = 0.0
    m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    m.lifetime.sec = 0
    m.lifetime.nanosec = 0
    return m

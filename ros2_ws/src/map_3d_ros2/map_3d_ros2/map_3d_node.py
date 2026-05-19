#!/usr/bin/env python3
"""ROS 2 node: sparse 3D map from monocular keyframes + odometry + two-view triangulation.

Subscribes to image, CameraInfo, odometry, /tf, /tf_static.
Publishes ``3d_map`` (PointCloud2) and ``camera_pose`` (inverse of world_T_camera used for points).
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException

from map_3d_ros2.param_config import apply_config_to_argv
from map_3d_ros2.range_gate import (
    consecutive_keyframe_baseline_m,
    distance_from_anchor_world,
    max_sparse_range_m,
)
from map_3d_ros2.ros_support import (
    BufferedFrame,
    camera_T_world_to_pose_stamped,
    camera_info_to_calibration,
    copy_image_msg,
    effective_K_from_calibration,
    ensure_pipeline_on_path,
    image_msg_to_gray_undistorted,
    odom_to_cam_to_world_T,
    should_buffer_image,
    transform_stamped_to_world_T,
    transform_points_world_T_camera,
    world_T_camera_to_quaternion_xyzw,
)

if TYPE_CHECKING:
    from pipeline.map import IncrementalMap


def _sensor_data_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


def _camera_info_qos(durability: DurabilityPolicy) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=durability,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _parse_camera_info_qos_durability(raw: str) -> DurabilityPolicy:
    key = raw.strip().lower()
    if key in ("transient_local", "transientlocal"):
        return DurabilityPolicy.TRANSIENT_LOCAL
    return DurabilityPolicy.VOLATILE


def _odom_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def _map_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _tf_rosbag_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _tf_static_live_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=32,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


class Map3dNode(Node):
    """Distance-based keyframes, odometry-grounded poses, raw sparse triangulation map."""

    def __init__(self) -> None:
        super().__init__("map_3d")

        repo = ensure_pipeline_on_path()
        if repo is None:
            self.get_logger().error(
                "Could not locate DepthFromMovement repo (missing pipeline/map.py). "
                "Install the thesis package editable or run from the cloned workspace."
            )
        self._repo_root = repo

        self.declare_parameter("image_topic", "/uav1/stereo/left/image_color")
        self.declare_parameter("camera_info_topic", "/uav1/stereo/left/camera_info")
        self.declare_parameter("odom_topic", "/uav1/estimation_manager/odom_main")
        self.declare_parameter("motion_threshold_m", 0.5)
        self.declare_parameter("keyframe_buffer_start_fraction", 0.8)
        self.declare_parameter("max_image_buffer", 64)
        self.declare_parameter("camera_info_qos_durability", "transient_local")
        self.declare_parameter("require_camera_info", True)
        self.declare_parameter("camera_fx", 600.0)
        self.declare_parameter("camera_fy", 600.0)
        self.declare_parameter("camera_cx", -1.0)
        self.declare_parameter("camera_cy", -1.0)
        self.declare_parameter("map_topic", "3d_map")
        self.declare_parameter("camera_pose_topic", "camera_pose")
        self.declare_parameter("map_publish_period_s", 1.0)
        self.declare_parameter("map_frame_id", "")
        self.declare_parameter("apply_tf_to_camera_pose", False)
        self.declare_parameter("base_frame", "uav1/base_link")
        self.declare_parameter("camera_frame", "uav1/stereo/left_optical")
        self.declare_parameter("tf_use_latest_time", False)
        self.declare_parameter("tf_static_volatile_qos", True)
        self.declare_parameter("feature_method", "ORB")
        self.declare_parameter("feature_n_features", 2000)
        self.declare_parameter("max_range_baseline_factor", 20.0)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self._camera_info_topic = camera_info_topic
        self._require_camera_info = (
            self.get_parameter("require_camera_info").get_parameter_value().bool_value
        )
        camera_info_durability = _parse_camera_info_qos_durability(
            self.get_parameter("camera_info_qos_durability").get_parameter_value().string_value
        )

        self._d = float(self.get_parameter("motion_threshold_m").get_parameter_value().double_value)
        buf_frac = float(
            self.get_parameter("keyframe_buffer_start_fraction").get_parameter_value().double_value
        )
        self._buffer_start_fraction = min(1.0, max(1e-6, buf_frac))
        self._max_buf = max(16, int(self.get_parameter("max_image_buffer").value))
        self._fx = float(self.get_parameter("camera_fx").get_parameter_value().double_value)
        self._fy = float(self.get_parameter("camera_fy").get_parameter_value().double_value)
        self._cx_auto = float(self.get_parameter("camera_cx").get_parameter_value().double_value)
        self._cy_auto = float(self.get_parameter("camera_cy").get_parameter_value().double_value)

        self._base_frame = self._normalize_tf_frame(
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self._camera_frame = self._normalize_tf_frame(
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self._tf_use_latest = (
            self.get_parameter("tf_use_latest_time").get_parameter_value().bool_value
        )
        self._tf_static_volatile_qos = (
            self.get_parameter("tf_static_volatile_qos").get_parameter_value().bool_value
        )
        self._apply_tf_to_camera_pose = (
            self.get_parameter("apply_tf_to_camera_pose").get_parameter_value().bool_value
        )

        self._map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self._camera_pose_topic = (
            self.get_parameter("camera_pose_topic").get_parameter_value().string_value
        )
        map_period = float(self.get_parameter("map_publish_period_s").get_parameter_value().double_value)
        self._map_period_s = max(0.05, map_period)
        self._map_frame_id_override = (
            self.get_parameter("map_frame_id").get_parameter_value().string_value.strip()
        )
        self._range_factor = float(
            self.get_parameter("max_range_baseline_factor").get_parameter_value().double_value
        )

        self._feature_method = (
            self.get_parameter("feature_method").get_parameter_value().string_value.strip().upper()
        )
        if self._feature_method not in ("ORB", "SIFT"):
            self._feature_method = "ORB"
        self._feature_n = max(1, int(self.get_parameter("feature_n_features").value))

        self._calibration = None
        self._K: np.ndarray | None = None
        self._last_camera_info_warn_time = 0.0
        self._last_odom: Odometry | None = None
        self._buffer: list[BufferedFrame] = []
        self._world_T_camera: list[np.ndarray] = []
        self._world_T_camera_raw: list[np.ndarray] = []
        self._gray_kf: list[np.ndarray] = []
        self._last_kf_pos: np.ndarray | None = None
        self._inc_map: IncrementalMap | None = None
        self._map_points: list[np.ndarray] = []
        self._latched_sensor_frame_id = ""
        self._last_world_T_camera: np.ndarray | None = None

        self._tf_buffer: Buffer | None = None
        self._tf_message_count = 0
        self._tf_static_message_count = 0
        self._last_tf_warn_time = 0.0
        self._last_tf_wait_log_time = 0.0

        self._map_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self._map_pub = self.create_publisher(PointCloud2, self._map_topic, _map_qos())
        self._camera_pose_pub = self.create_publisher(PoseStamped, self._camera_pose_topic, _odom_qos())
        self.create_timer(self._map_period_s, self._on_map_timer)

        self.create_subscription(Image, image_topic, self._on_image, _sensor_data_qos())
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            _camera_info_qos(camera_info_durability),
        )
        self.create_subscription(Odometry, odom_topic, self._on_odom, _odom_qos())

        if self._apply_tf_to_camera_pose:
            self._init_tf_subscriptions()

        tf_mode = (
            f"TF {self._base_frame!r}->{self._camera_frame!r}"
            if self._apply_tf_to_camera_pose
            else "odom child as camera"
        )
        _log(
            f"map_3d: image={image_topic!r} odom={odom_topic!r} "
            f"motion_threshold_m={self._d} map={self._map_topic!r} "
            f"camera_pose={self._camera_pose_topic!r} | {tf_mode}"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._calibration is not None:
            return
        try:
            cal = camera_info_to_calibration(msg)
        except ValueError as e:
            self.get_logger().error(f"Invalid CameraInfo: {e}")
            return
        self._calibration = cal
        self._K = effective_K_from_calibration(cal)
        sensor_frame = self._normalize_tf_frame(msg.header.frame_id)
        if sensor_frame:
            self._latched_sensor_frame_id = sensor_frame
        self._init_map_if_possible()

    def _require_calibration_for_image(self, msg: Image) -> bool:
        if self._calibration is None:
            if self._require_camera_info:
                now = time.monotonic()
                if now - self._last_camera_info_warn_time >= 5.0:
                    self._last_camera_info_warn_time = now
                    self.get_logger().warning(
                        f"Waiting for CameraInfo on {self._camera_info_topic!r}; skipping image."
                    )
                return False
            if self._K is None:
                w, h = int(msg.width), int(msg.height)
                cx = w * 0.5 if self._cx_auto < 0 else self._cx_auto
                cy = h * 0.5 if self._cy_auto < 0 else self._cy_auto
                self._K = np.array(
                    [[self._fx, 0.0, cx], [0.0, self._fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                self._init_map_if_possible()
            return True
        size = self._calibration.image_size
        if size is not None and (int(msg.width), int(msg.height)) != size:
            return False
        return True

    def _init_map_if_possible(self) -> None:
        if self._K is None or self._repo_root is None or self._inc_map is not None:
            return
        try:
            from pipeline.config import FeatureConfig
            from pipeline.map import IncrementalMap, MapConfig
        except ImportError as e:
            self.get_logger().error(f"Could not import pipeline: {e}")
            return
        feat = FeatureConfig(method=self._feature_method, n_features=self._feature_n)
        self._inc_map = IncrementalMap(
            cfg=MapConfig(), feat_cfg=feat, K=self._K, world_T_camera=self._world_T_camera
        )

    @staticmethod
    def _normalize_tf_frame(name: str) -> str:
        s = (name or "").strip()
        if s.startswith("/"):
            s = s[1:]
        return s

    def _camera_frame_candidates(self) -> list[str]:
        cands: list[str] = []
        for raw in (self._latched_sensor_frame_id, self._camera_frame):
            n = self._normalize_tf_frame(raw)
            if n and n not in cands:
                cands.append(n)
        return cands or [self._camera_frame]

    def _init_tf_subscriptions(self) -> None:
        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        tf_qos = _tf_rosbag_qos()
        static_qos = tf_qos if self._tf_static_volatile_qos else _tf_static_live_qos()
        self.create_subscription(TFMessage, "/tf", self._on_tf_message, tf_qos)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static_message, static_qos)

    def _on_tf_message(self, msg: TFMessage) -> None:
        if self._tf_buffer is None:
            return
        for tr in msg.transforms:
            try:
                self._tf_buffer.set_transform(tr, "map_3d_ros2")
            except TransformException:
                pass
        self._tf_message_count += 1

    def _on_tf_static_message(self, msg: TFMessage) -> None:
        if self._tf_buffer is None:
            return
        for tr in msg.transforms:
            try:
                self._tf_buffer.set_transform_static(tr, "map_3d_ros2")
            except TransformException:
                try:
                    self._tf_buffer.set_transform(tr, "map_3d_ros2")
                except TransformException:
                    pass
        self._tf_static_message_count += 1

    def _tf_time_from_stamp(self, sec: int, nsec: int) -> Time:
        if self._tf_use_latest or (sec == 0 and nsec == 0):
            return Time()
        from builtin_interfaces.msg import Time as TimeMsg

        return Time.from_msg(TimeMsg(sec=int(sec), nanosec=int(nsec)))

    def _tf_stamps_for_lookup(self, stamp_sec: int, stamp_nsec: int) -> list[Time]:
        stamps: list[Time] = [Time()]
        if not self._tf_use_latest:
            stamps.append(self._tf_time_from_stamp(stamp_sec, stamp_nsec))
        return stamps

    def _lookup_transform_to_matrix(
        self, parent_frame: str, child_frame: str, stamps: list[Time]
    ) -> np.ndarray | None:
        parent_frame = self._normalize_tf_frame(parent_frame)
        child_frame = self._normalize_tf_frame(child_frame)
        if self._tf_buffer is None or not parent_frame or not child_frame:
            return None
        for stamp in stamps:
            try:
                if not self._tf_buffer.can_transform(
                    parent_frame, child_frame, stamp, timeout=Duration(seconds=1.0)
                ):
                    continue
                t = self._tf_buffer.lookup_transform(
                    parent_frame, child_frame, stamp, timeout=Duration(seconds=0.5)
                )
                return transform_stamped_to_world_T(t)
            except TransformException:
                continue
        return None

    def _odom_child_frame_id(self, msg: Odometry) -> str:
        return self._normalize_tf_frame(msg.child_frame_id)

    def _lookup_world_T_camera_via_tf(
        self, msg: Odometry, stamps: list[Time]
    ) -> np.ndarray | None:
        world = self._normalize_tf_frame(msg.header.frame_id)
        child = self._odom_child_frame_id(msg)
        world_T_child = odom_to_cam_to_world_T(msg)
        base = self._normalize_tf_frame(self._base_frame)

        for camera in self._camera_frame_candidates():
            if world:
                T_direct = self._lookup_transform_to_matrix(world, camera, stamps)
                if T_direct is not None:
                    return T_direct
            if not child:
                continue
            for parent in (child, base):
                T_parent_cam = self._lookup_transform_to_matrix(parent, camera, stamps)
                if T_parent_cam is None:
                    continue
                if parent == child:
                    return world_T_child @ T_parent_cam
                T_child_parent = self._lookup_transform_to_matrix(child, parent, stamps)
                if T_child_parent is not None:
                    return world_T_child @ T_child_parent @ T_parent_cam
        return None

    def _resolve_world_T_camera(
        self,
        msg: Odometry,
        stamp_sec: int,
        stamp_nsec: int,
        *,
        allow_odom_fallback: bool,
    ) -> np.ndarray | None:
        if not self._apply_tf_to_camera_pose or self._tf_buffer is None:
            return odom_to_cam_to_world_T(msg)
        child = self._odom_child_frame_id(msg)
        for camera in self._camera_frame_candidates():
            if child and child == camera:
                return odom_to_cam_to_world_T(msg)
        stamps = self._tf_stamps_for_lookup(stamp_sec, stamp_nsec)
        T = self._lookup_world_T_camera_via_tf(msg, stamps)
        if T is not None:
            return T
        if not allow_odom_fallback:
            return None
        self._maybe_warn_tf("TF missing; using odom child pose (body frame).")
        return odom_to_cam_to_world_T(msg)

    def _maybe_warn_tf(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_tf_warn_time >= 5.0:
            self._last_tf_warn_time = now
            self.get_logger().warning(message)

    def _can_resolve_camera_pose_from_odom(self, msg: Odometry) -> bool:
        if not self._apply_tf_to_camera_pose:
            return True
        if self._tf_static_volatile_qos and self._tf_static_message_count < 1:
            return False
        stamps = self._tf_stamps_for_lookup(msg.header.stamp.sec, msg.header.stamp.nanosec)
        return self._lookup_world_T_camera_via_tf(msg, stamps) is not None

    def _current_metric_position(self) -> np.ndarray | None:
        if self._last_odom is None:
            return None
        allow_fb = self._last_kf_pos is not None
        T = self._resolve_world_T_camera(
            self._last_odom,
            self._last_odom.header.stamp.sec,
            self._last_odom.header.stamp.nanosec,
            allow_odom_fallback=allow_fb,
        )
        if T is None:
            return None
        return T[:3, 3].copy()

    def _travel_fraction_since_last_kf(self) -> float | None:
        if self._last_kf_pos is None or self._d <= 0.0:
            return None
        pos_cur = self._current_metric_position()
        if pos_cur is None:
            return None
        return float(np.linalg.norm(pos_cur - self._last_kf_pos) / self._d)

    def _maybe_clear_buffer_for_fraction(self) -> None:
        if self._last_kf_pos is None or not self._buffer:
            return
        frac = self._travel_fraction_since_last_kf()
        if frac is not None and frac < self._buffer_start_fraction:
            self._buffer.clear()

    def _map_parent_frame(self) -> str:
        if self._map_frame_id_override:
            return self._map_frame_id_override
        if self._last_odom is not None and self._last_odom.header.frame_id:
            return self._last_odom.header.frame_id
        return ""

    def _on_image(self, msg: Image) -> None:
        if not self._require_calibration_for_image(msg):
            return
        if self._last_odom is None:
            return
        if self._last_kf_pos is None and not self._can_resolve_camera_pose_from_odom(self._last_odom):
            self._maybe_log_waiting_for_tf()
            return
        fraction = self._travel_fraction_since_last_kf()
        if not should_buffer_image(
            fraction,
            self._buffer_start_fraction,
            has_last_keyframe=self._last_kf_pos is not None,
        ):
            self._maybe_clear_buffer_for_fraction()
            return
        allow_fb = self._last_kf_pos is not None or not self._apply_tf_to_camera_pose
        T = self._resolve_world_T_camera(
            self._last_odom,
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            allow_odom_fallback=allow_fb,
        )
        if T is None:
            return
        pos = T[:3, 3].copy()
        qx, qy, qz, qw = world_T_camera_to_quaternion_xyzw(T)
        self._buffer.append(
            BufferedFrame(
                stamp_sec=msg.header.stamp.sec,
                stamp_nsec=msg.header.stamp.nanosec,
                image_msg=copy_image_msg(msg),
                pos_odom=pos,
                cam_to_world=T,
                qx=float(qx),
                qy=float(qy),
                qz=float(qz),
                qw=float(qw),
            )
        )
        while len(self._buffer) > self._max_buf:
            self._buffer.pop(0)
        self._try_keyframe_selection()

    def _maybe_log_waiting_for_tf(self) -> None:
        now = time.monotonic()
        if now - self._last_tf_wait_log_time >= 5.0:
            self._last_tf_wait_log_time = now
            self.get_logger().info(
                f"Waiting for optical TF (/tf={self._tf_message_count} "
                f"/tf_static={self._tf_static_message_count}) before first keyframe."
            )

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._maybe_clear_buffer_for_fraction()
        self._try_keyframe_selection()

    def _try_keyframe_selection(self) -> None:
        if self._last_odom is None or self._K is None or not self._buffer:
            return
        if self._inc_map is None:
            self._init_map_if_possible()
        if self._inc_map is None:
            return

        if self._last_kf_pos is None:
            if not self._can_resolve_camera_pose_from_odom(self._last_odom):
                self._maybe_log_waiting_for_tf()
                return
            bf0 = self._buffer.pop(0)
            self._buffer.clear()
            self._commit_keyframe(bf0, distance_trigger_m=None)
            return

        pos_cur = self._current_metric_position()
        if pos_cur is None:
            return
        dist = float(np.linalg.norm(pos_cur - self._last_kf_pos))
        if dist < self._d:
            return
        v = pos_cur - self._last_kf_pos
        u = v / (np.linalg.norm(v) + 1e-12)
        target = self._last_kf_pos + self._d * u
        best_i = min(
            range(len(self._buffer)),
            key=lambda i: float(np.linalg.norm(self._buffer[i].pos_odom - target)),
        )
        chosen = self._buffer[best_i]
        self._buffer.clear()
        self._commit_keyframe(chosen, distance_trigger_m=dist)

    def _append_triangulated_points(
        self,
        tw,
        *,
        frame_i: int,
        world_T_camera_raw_i: np.ndarray,
        max_range_world: float | None,
    ) -> int:
        if tw.X_cam_h is None or not tw.scale_ok:
            return 0
        anchor = np.asarray(world_T_camera_raw_i, dtype=np.float64)[:3, 3]
        added = 0
        X_cam = tw.X_cam_h
        mask = np.asarray(tw.cheiral_mask, dtype=bool)
        for k in range(X_cam.shape[1]):
            if not mask[k]:
                continue
            Xc = X_cam[:3, k]
            if not np.all(np.isfinite(Xc)):
                continue
            Xw = transform_points_world_T_camera(Xc.reshape(1, 3), world_T_camera_raw_i)[0]
            if max_range_world is not None:
                if distance_from_anchor_world(Xw, anchor) > max_range_world:
                    continue
            self._map_points.append(Xw.astype(np.float64))
            added += 1
        return added

    def _publish_camera_pose(self, world_T_camera: np.ndarray, stamp_sec: int, stamp_nsec: int) -> None:
        parent = self._map_parent_frame()
        if not parent:
            return
        hdr = Header()
        hdr.stamp.sec = int(stamp_sec)
        hdr.stamp.nanosec = int(stamp_nsec)
        hdr.frame_id = parent
        self._camera_pose_pub.publish(camera_T_world_to_pose_stamped(world_T_camera, header=hdr))
        self._last_world_T_camera = world_T_camera.copy()

    def _commit_keyframe(self, bf: BufferedFrame, *, distance_trigger_m: float | None) -> None:
        from pipeline.geometry import canonicalize_world_T_camera_to_first

        if self._last_odom is None:
            return
        allow_fb = not self._apply_tf_to_camera_pose
        T_opt = self._resolve_world_T_camera(
            self._last_odom,
            bf.stamp_sec,
            bf.stamp_nsec,
            allow_odom_fallback=allow_fb,
        )
        if T_opt is None:
            self.get_logger().warning(
                f"Keyframe skip: no optical TF at {bf.stamp_sec}.{bf.stamp_nsec:09d}"
            )
            return
        bf.cam_to_world = T_opt
        bf.pos_odom = T_opt[:3, 3].copy()

        gray, k_eff = image_msg_to_gray_undistorted(bf.image_msg, self._calibration)
        if gray is None:
            self.get_logger().warning(
                f"Keyframe skip: unsupported encoding {bf.image_msg.encoding!r}"
            )
            return
        if k_eff is not None and (self._K is None or not np.allclose(self._K, k_eff)):
            self._K = k_eff
            self._inc_map = None
            self._init_map_if_possible()

        idx = len(self._world_T_camera_raw)
        self._world_T_camera_raw.append(bf.cam_to_world.copy())
        self._world_T_camera[:] = canonicalize_world_T_camera_to_first(self._world_T_camera_raw)
        self._gray_kf.append(gray)
        self._last_kf_pos = bf.pos_odom.copy()

        added = 0
        reproj_mean = float("nan")
        baseline_m = distance_trigger_m if distance_trigger_m is not None else 0.0

        if idx >= 1 and self._inc_map is not None:
            i = idx - 1
            max_range_world: float | None = None
            if len(self._world_T_camera) > idx:
                baseline_m = consecutive_keyframe_baseline_m(
                    self._world_T_camera[i], self._world_T_camera[idx]
                )
                max_range_world = max_sparse_range_m(baseline_m, self._range_factor)
            try:
                tw = self._inc_map.add_frame_pair(i, idx, self._gray_kf[i], self._gray_kf[idx])
                added = self._append_triangulated_points(
                    tw,
                    frame_i=i,
                    world_T_camera_raw_i=self._world_T_camera_raw[i],
                    max_range_world=max_range_world,
                )
                reproj_mean = float(tw.reproj.get("mean", float("nan")))
            except Exception as e:
                self.get_logger().error(f"add_frame_pair ({i}->{idx}) failed: {e}")

        self._publish_camera_pose(bf.cam_to_world, bf.stamp_sec, bf.stamp_nsec)

        total = len(self._map_points)
        d_str = f" baseline={baseline_m:.2f}m" if idx >= 1 else ""
        reproj_str = f" reproj_mean={reproj_mean:.2f}px" if np.isfinite(reproj_mean) else ""
        _log(f"map_3d kf={idx} +{added} pts total={total}{d_str}{reproj_str}")

    def _on_map_timer(self) -> None:
        if not self._map_points:
            return
        fid = self._map_parent_frame()
        if not fid:
            return
        pts = np.vstack(self._map_points)
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = fid
        tuples = [(float(r[0]), float(r[1]), float(r[2])) for r in pts]
        cloud = point_cloud2.create_cloud(hdr, self._map_fields, tuples)
        self._map_pub.publish(cloud)


def main(argv: list[str] | None = None) -> None:
    try:
        ros_args = apply_config_to_argv(argv)
    except (FileNotFoundError, ValueError) as e:
        print(f"map_3d: config error: {e}", file=sys.stderr)
        sys.exit(2)
    rclpy.init(args=ros_args)
    node = Map3dNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

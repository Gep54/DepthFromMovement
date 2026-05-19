#!/usr/bin/env python3
"""ROS 2 node: live sparse 3D map from monocular keyframes + odometry + two-view triangulation.

Odometry supplies the metric ``world_T_camera`` track (optionally composed with TF to the optical frame).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
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
from tf2_ros import Buffer, TransformBroadcaster, TransformException
from visualization_msgs.msg import Marker

from incremental_vo_ros2.offline_dataset import offline_dataset_image_basename
from incremental_vo_ros2.param_config import apply_config_to_argv
from incremental_vo_ros2.support import (
    BufferedFrame,
    camera_info_to_calibration,
    consecutive_keyframe_baseline_m,
    copy_image_msg,
    ensure_pipeline_on_path,
    eval_world_T_camera0_from_parameter,
    image_msg_to_gray_undistorted,
    log_frame_transforms_block,
    max_sparse_range_m,
    odom_to_cam_to_world_T,
    should_reset_bag_replay,
    save_keyframe_manifest,
    save_sparse_map_eval_world_npz,
    save_sparse_map_npz,
    should_buffer_image,
    transform_stamped_to_world_T,
    effective_K_from_calibration,
    world_T_camera_to_pose_stamped,
    world_T_camera_to_quaternion_xyzw,
    world_T_camera_to_transform_stamped,
)


def _sensor_data_qos() -> QoSProfile:
    # Cameras usually publish best-effort; depth 5 drops old frames under load instead of blocking.
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


def _parse_camera_info_qos_durability(raw: str, logger) -> DurabilityPolicy:
    key = raw.strip().lower()
    if key in ("transient_local", "transientlocal"):
        return DurabilityPolicy.TRANSIENT_LOCAL
    if key == "volatile":
        return DurabilityPolicy.VOLATILE
    logger.warning(
        f"camera_info_qos_durability={raw!r} not in {{transient_local, volatile}}; "
        "falling back to transient_local."
    )
    return DurabilityPolicy.TRANSIENT_LOCAL


def _odom_qos() -> QoSProfile:
    # Odom streams are usually reliable; if your publisher uses different QoS, match it or use a remap.
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def _sparse_map_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _tf_rosbag_qos() -> QoSProfile:
    """Volatile QoS — required for rosbag2 replay of ``/tf`` and ``/tf_static``."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _tf_static_live_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class IncrementalVoNode(Node):
    """Distance-based keyframes, odometry-grounded poses, two-view sparse triangulation."""

    def __init__(self) -> None:
        super().__init__("incremental_vo_node")

        repo = ensure_pipeline_on_path()
        if repo is None:
            self.get_logger().error(
                "Could not locate DepthFromMovement repo (missing pipeline/map.py on parents of this "
                "package). Install the thesis package editable or run from the cloned workspace."
            )
        self._repo_root = repo

        # Topics (defaults match uav1 stack).
        self.declare_parameter("image_topic", "/uav1/stereo/left/image_color")
        self.declare_parameter("odom_main_topic", "/uav1/estimation_manager/odom_main")
        self.declare_parameter("subscribe_odom_gt", False)
        self.declare_parameter("odom_gt_topic", "/flightforge_simulator/uav1/odom")

        # Keyframe + output.
        self.declare_parameter("keyframe_distance_m", 0.5)
        self.declare_parameter("keyframe_buffer_start_fraction", 0.8)
        self.declare_parameter("output_root", ".")
        self.declare_parameter("max_image_buffer", 64)
        self.declare_parameter("camera_info_topic", "/uav1/stereo/left/camera_info")
        self.declare_parameter("camera_info_qos_durability", "transient_local")
        self.declare_parameter("require_camera_info", True)
        # Deprecated when require_camera_info=true (intrinsics come from CameraInfo).
        self.declare_parameter("camera_fx", 600.0)
        self.declare_parameter("camera_fy", 600.0)
        self.declare_parameter("camera_cx", -1.0)
        self.declare_parameter("camera_cy", -1.0)
        self.declare_parameter("pair_lookback", 1)

        # Live sparse map (``sensor_msgs/PointCloud2``, metric world = odometry parent frame).
        self.declare_parameter("publish_sparse_map", True)
        self.declare_parameter("sparse_map_topic", "sparse_map")
        self.declare_parameter("sparse_map_publish_period_s", 1.0)
        self.declare_parameter("sparse_map_frame_id", "")
        self.declare_parameter("sparse_map_max_range_baseline_factor", 20.0)
        self.declare_parameter("save_run_on_shutdown", False)

        # Offline dataset export (``load_dataset`` / ``dfm-export-steps``-ready layout).
        self.declare_parameter("export_offline_dataset", False)
        self.declare_parameter("offline_dataset_root", "")
        self.declare_parameter("offline_dataset_image_prefix", "frame")

        # RViz debug: camera pose/orientation (PoseStamped + optional TF).
        self.declare_parameter("publish_camera_pose_debug", True)
        self.declare_parameter("camera_pose_debug_topic", "camera_pose_debug")
        self.declare_parameter("camera_pose_debug_frame_id", "")
        self.declare_parameter("camera_pose_debug_child_frame_id", "dfm/camera_optical")
        self.declare_parameter("publish_camera_pose_tf", True)

        # RViz: camera +Z line per keyframe; compose odom child → optical via TF.
        self.declare_parameter("publish_keyframe_markers", False)
        self.declare_parameter("keyframe_marker_topic", "keyframe_camera_z_arrows")
        self.declare_parameter("keyframe_marker_length_m", 0.5)
        self.declare_parameter("keyframe_marker_frame_id", "")
        self.declare_parameter("apply_tf_to_camera_pose", False)

        # Optional TF debug (off by default for rosbag-focused runs).
        self.declare_parameter("base_frame", "uav1/base_link")
        self.declare_parameter("camera_frame", "uav1/stereo/left_optical")
        self.declare_parameter("tf_lookup_period_s", 0.0)
        self.declare_parameter("tf_use_latest_time", False)
        self.declare_parameter("tf_static_volatile_qos", True)
        self.declare_parameter("log_image_hz", 0.0)
        self.declare_parameter("log_frame_transforms", False)
        self.declare_parameter("log_random_triangulated_point", False)

        # Descriptor-based landmark fusion (replace-if-better descriptors + EMA position update).
        # Sentinel ``-1.0`` means "use DescriptorMapConfig.defaults(method)" / ``None`` for nullable
        # fields; sentinel -1.0 selects DescriptorMapConfig.defaults(method).
        self.declare_parameter("feature_method", "ORB")
        self.declare_parameter("feature_n_features", 2000)
        self.declare_parameter("descriptor_fusion_enabled", True)
        self.declare_parameter("descriptor_spatial_merge_baseline_factor", 0.25)
        self.declare_parameter("descriptor_spatial_merge_radius_m", -1.0)
        self.declare_parameter("descriptor_merge_beta", -1.0)
        self.declare_parameter("descriptor_max_match_distance", -1.0)
        self.declare_parameter("descriptor_ratio_second_best", -1.0)

        self.declare_parameter("eval_world_T_camera0", [0.0] * 16)
        self.declare_parameter("bag_replay_reset_enabled", True)
        self.declare_parameter("bag_replay_reset_jump_s", 1.0)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self._camera_info_topic = camera_info_topic
        self._require_camera_info = (
            self.get_parameter("require_camera_info").get_parameter_value().bool_value
        )
        camera_info_durability_raw = (
            self.get_parameter("camera_info_qos_durability")
            .get_parameter_value()
            .string_value
        )
        camera_info_durability = _parse_camera_info_qos_durability(
            camera_info_durability_raw, self.get_logger()
        )
        odom_main_topic = self.get_parameter("odom_main_topic").get_parameter_value().string_value
        subscribe_gt = self.get_parameter("subscribe_odom_gt").get_parameter_value().bool_value
        odom_gt_topic = self.get_parameter("odom_gt_topic").get_parameter_value().string_value

        self._d = float(self.get_parameter("keyframe_distance_m").get_parameter_value().double_value)
        buf_frac = float(
            self.get_parameter("keyframe_buffer_start_fraction").get_parameter_value().double_value
        )
        self._buffer_start_fraction = min(1.0, max(1e-6, buf_frac))
        out_root = Path(self.get_parameter("output_root").get_parameter_value().string_value).expanduser()
        self._max_buf = max(16, int(self.get_parameter("max_image_buffer").value))
        self._pair_lookback = max(1, int(self.get_parameter("pair_lookback").value))
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
        tf_period = self.get_parameter("tf_lookup_period_s").get_parameter_value().double_value
        self._tf_use_latest = (
            self.get_parameter("tf_use_latest_time").get_parameter_value().bool_value
        )
        self._tf_static_volatile_qos = (
            self.get_parameter("tf_static_volatile_qos").get_parameter_value().bool_value
        )
        log_hz = self.get_parameter("log_image_hz").get_parameter_value().double_value
        self._image_log_interval_s = 1.0 / log_hz if log_hz > 0.0 else math.inf
        self._last_image_log_time = self.get_clock().now()
        self._log_frame_transforms = (
            self.get_parameter("log_frame_transforms").get_parameter_value().bool_value
        )
        self._log_random_triangulated_point = (
            self.get_parameter("log_random_triangulated_point")
            .get_parameter_value()
            .bool_value
        )

        eval_param = self.get_parameter("eval_world_T_camera0").value
        eval_seq = eval_param if isinstance(eval_param, (list, tuple)) else []
        self._eval_world_T_cam0 = eval_world_T_camera0_from_parameter(eval_seq)

        self._feature_method = (
            self.get_parameter("feature_method").get_parameter_value().string_value.strip().upper()
        )
        if self._feature_method not in ("ORB", "SIFT"):
            self.get_logger().warning(
                f"feature_method={self._feature_method!r} not in {{ORB,SIFT}}; falling back to ORB."
            )
            self._feature_method = "ORB"
        self._feature_n = max(1, int(self.get_parameter("feature_n_features").value))
        self._descriptor_fusion_enabled = (
            self.get_parameter("descriptor_fusion_enabled").get_parameter_value().bool_value
        )
        self._descriptor_spatial_merge_baseline_factor = float(
            self.get_parameter("descriptor_spatial_merge_baseline_factor")
            .get_parameter_value()
            .double_value
        )
        self._descriptor_spatial_merge_radius_m = float(
            self.get_parameter("descriptor_spatial_merge_radius_m").get_parameter_value().double_value
        )
        self._descriptor_merge_beta = float(
            self.get_parameter("descriptor_merge_beta").get_parameter_value().double_value
        )
        self._descriptor_max_match_distance = float(
            self.get_parameter("descriptor_max_match_distance").get_parameter_value().double_value
        )
        self._descriptor_ratio_second_best = float(
            self.get_parameter("descriptor_ratio_second_best").get_parameter_value().double_value
        )
        self._publish_sparse_map = (
            self.get_parameter("publish_sparse_map").get_parameter_value().bool_value
        )
        self._sparse_map_topic = (
            self.get_parameter("sparse_map_topic").get_parameter_value().string_value
        )
        sparse_period = float(
            self.get_parameter("sparse_map_publish_period_s").get_parameter_value().double_value
        )
        self._sparse_map_period_s = max(0.05, sparse_period)
        self._sparse_map_frame_id_override = (
            self.get_parameter("sparse_map_frame_id").get_parameter_value().string_value.strip()
        )
        self._save_run_on_shutdown = (
            self.get_parameter("save_run_on_shutdown").get_parameter_value().bool_value
        )
        self._export_offline_dataset = (
            self.get_parameter("export_offline_dataset").get_parameter_value().bool_value
        )
        offline_root_param = (
            self.get_parameter("offline_dataset_root").get_parameter_value().string_value.strip()
        )
        self._offline_image_prefix = (
            self.get_parameter("offline_dataset_image_prefix")
            .get_parameter_value()
            .string_value.strip()
            or "frame"
        )
        self._sparse_map_range_factor = float(
            self.get_parameter("sparse_map_max_range_baseline_factor")
            .get_parameter_value()
            .double_value
        )
        self._camera_pose_debug_enabled = (
            self.get_parameter("publish_camera_pose_debug").get_parameter_value().bool_value
        )
        self._camera_pose_debug_topic = (
            self.get_parameter("camera_pose_debug_topic").get_parameter_value().string_value
        )
        self._camera_pose_debug_frame_id = (
            self.get_parameter("camera_pose_debug_frame_id").get_parameter_value().string_value.strip()
        )
        self._camera_pose_debug_child_frame_id = (
            self.get_parameter("camera_pose_debug_child_frame_id")
            .get_parameter_value()
            .string_value.strip()
            or "dfm/camera_optical"
        )
        self._publish_camera_pose_tf = (
            self.get_parameter("publish_camera_pose_tf").get_parameter_value().bool_value
        )
        self._publish_keyframe_markers = (
            self.get_parameter("publish_keyframe_markers").get_parameter_value().bool_value
        )
        self._keyframe_marker_topic = (
            self.get_parameter("keyframe_marker_topic").get_parameter_value().string_value
        )
        self._keyframe_marker_length_m = float(
            self.get_parameter("keyframe_marker_length_m").get_parameter_value().double_value
        )
        self._keyframe_marker_frame_id = (
            self.get_parameter("keyframe_marker_frame_id").get_parameter_value().string_value.strip()
        )
        self._apply_tf_to_camera_pose = (
            self.get_parameter("apply_tf_to_camera_pose").get_parameter_value().bool_value
        )
        self._last_tf_warn_time = 0.0
        self._last_tf_wait_log_time = 0.0
        self._last_consecutive_baseline_m: float | None = None
        self._bag_replay_reset_enabled = (
            self.get_parameter("bag_replay_reset_enabled").get_parameter_value().bool_value
        )
        self._bag_replay_reset_jump_s = float(
            self.get_parameter("bag_replay_reset_jump_s").get_parameter_value().double_value
        )
        self._last_message_stamp_s: float | None = None

        self._allocate_run_dir(out_root)
        run_tag = self._run_dir.name

        self._offline_dataset_dir: Path | None = None
        self._offline_motion_frames: list[dict] = []
        if self._export_offline_dataset:
            if offline_root_param:
                self._offline_dataset_dir = Path(offline_root_param).expanduser().resolve()
            else:
                self._offline_dataset_dir = (
                    out_root.resolve() / "offline_datasets" / run_tag
                ).resolve()
            self._offline_dataset_dir.mkdir(parents=True, exist_ok=True)
            (self._offline_dataset_dir / "images").mkdir(exist_ok=True)

        self._calibration = None
        self._K: np.ndarray | None = None
        self._last_camera_info_warn_time = 0.0
        self._last_odom: Odometry | None = None
        self._last_image_msg: Image | None = None
        self._buffer: list[BufferedFrame] = []
        self._world_T_camera: list[np.ndarray] = []
        self._gray_kf: list[np.ndarray] = []
        self._kf_records: list[dict] = []
        self._last_kf_pos: np.ndarray | None = None
        self._inc_map = None
        self._desc_map = None
        self._effective_desc_cfg = None
        self._persisted = False
        self._world_T_camera_raw: list[np.ndarray] = []
        self._world_T_drone_raw: list[np.ndarray] = []
        self._tf_buffer: Buffer | None = None
        self._tf_message_count = 0
        self._tf_static_message_count = 0
        self._tf_edges: set[tuple[str, str]] = set()
        self._tf_static_edges: set[tuple[str, str]] = set()
        self._tf_set_fail_count = 0
        self._tf_static_summary_logged = False
        self._latched_sensor_frame_id = ""
        self._resolved_camera_frame = ""

        self._sparse_map_pub = None
        self._camera_pose_debug_pub = None
        self._camera_pose_tf_broadcaster: TransformBroadcaster | None = None
        self._keyframe_marker_pub = None
        self._sparse_map_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if self._publish_sparse_map:
            self._sparse_map_pub = self.create_publisher(
                PointCloud2, self._sparse_map_topic, _sparse_map_qos()
            )
            self.create_timer(self._sparse_map_period_s, self._on_sparse_map_timer)
        if self._camera_pose_debug_enabled:
            self._camera_pose_debug_pub = self.create_publisher(
                PoseStamped, self._camera_pose_debug_topic, _odom_qos()
            )
            if self._publish_camera_pose_tf:
                self._camera_pose_tf_broadcaster = TransformBroadcaster(self)
        if self._publish_keyframe_markers:
            self._keyframe_marker_pub = self.create_publisher(
                Marker, self._keyframe_marker_topic, _odom_qos()
            )

        self.create_subscription(Image, image_topic, self._on_image, _sensor_data_qos())
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            _camera_info_qos(camera_info_durability),
        )
        self.create_subscription(Odometry, odom_main_topic, self._on_odom_main, _odom_qos())
        self._last_odom_gt: Odometry | None = None
        if subscribe_gt:
            self.create_subscription(Odometry, odom_gt_topic, self._on_odom_gt, _odom_qos())

        if tf_period > 0.0 or self._apply_tf_to_camera_pose:
            self._init_tf_subscriptions()
            if tf_period > 0.0:
                self.create_timer(tf_period, self._on_tf_timer)
        if self._apply_tf_to_camera_pose:
            self._log_tf_startup_hints()

        self.get_logger().info(
            f"Run directory: {self._run_dir} | keyframe_distance_m={self._d} | "
            f"keyframe_buffer_start_fraction={self._buffer_start_fraction} "
            f"(lazy preprocess on commit) | "
            f"pair_lookback={self._pair_lookback} | "
            f"image={image_topic!r} camera_info={camera_info_topic!r} "
            f"camera_info_qos_durability={camera_info_durability_raw!r} "
            f"require_camera_info={self._require_camera_info} odom={odom_main_topic!r}"
            + (f" | odom_gt={odom_gt_topic!r}" if subscribe_gt else "")
            + (
                f" | TF periodic debug {self._base_frame!r}<-{self._camera_frame!r} every {tf_period}s"
                if tf_period > 0.0
                else (
                    " | TF extrinsic listener on (odom child→camera_frame)"
                    if self._apply_tf_to_camera_pose
                    else " | TF listener off"
                )
            )
            + (
                (
                    f" | sparse_map topic={self._sparse_map_topic!r} every {self._sparse_map_period_s}s "
                    + (
                        f"frame_id={self._sparse_map_frame_id_override!r}"
                        if self._sparse_map_frame_id_override
                        else "frame_id=odom.header.frame_id"
                    )
                )
                if self._publish_sparse_map
                else " | sparse_map publishing disabled"
            )
            + f" | save_run_on_shutdown={self._save_run_on_shutdown}"
            + (
                f" | offline_dataset={self._offline_dataset_dir!s} "
                f"prefix={self._offline_image_prefix!r}"
                if self._offline_dataset_dir is not None
                else " | offline_dataset export disabled"
            )
            + (
                f" | sparse_map_max_range_baseline_factor={self._sparse_map_range_factor}"
                if self._sparse_map_range_factor > 0.0
                else " | sparse_map range filter disabled (factor<=0)"
            )
            + (
                f" | camera_pose_debug topic={self._camera_pose_debug_topic!r}"
                + (
                    f" frame_id={self._camera_pose_debug_frame_id!r}"
                    if self._camera_pose_debug_frame_id
                    else " frame_id=odom.header.frame_id"
                )
                + f" child_tf={self._camera_pose_debug_child_frame_id!r}"
                + (" + TF" if self._publish_camera_pose_tf else " pose only")
                if self._camera_pose_debug_enabled
                else " | camera_pose_debug disabled"
            )
            + (
                f" | apply_tf_to_camera_pose={self._apply_tf_to_camera_pose}"
                f" camera_frame={self._camera_frame!r}"
                if self._apply_tf_to_camera_pose
                else ""
            )
            + (
                f" | keyframe_markers topic={self._keyframe_marker_topic!r} "
                f"length_m={self._keyframe_marker_length_m}"
                if self._publish_keyframe_markers
                else ""
            )
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._calibration is not None:
            return
        try:
            cal = camera_info_to_calibration(msg)
        except ValueError as e:
            self.get_logger().error(f"Invalid CameraInfo on {self._camera_info_topic!r}: {e}")
            return
        d_raw = np.asarray(list(msg.d), dtype=np.float64).reshape(-1) if msg.d else np.zeros(0)
        if d_raw.size > 0 and np.linalg.norm(d_raw) >= 1e-9:
            ensure_pipeline_on_path()
            from data.camera_calibration import distortion_model_supported

            if not distortion_model_supported(msg.distortion_model):
                self.get_logger().warning(
                    f"Unsupported distortion_model={msg.distortion_model!r}; "
                    "ignoring non-zero D coefficients."
                )
        self._calibration = cal
        self._K = effective_K_from_calibration(cal)
        d_norm = 0.0 if cal.dist_coeffs is None else float(np.linalg.norm(cal.dist_coeffs))
        K = cal.K
        sensor_frame = self._normalize_tf_frame(msg.header.frame_id)
        if sensor_frame:
            self._latched_sensor_frame_id = sensor_frame
            if sensor_frame != self._camera_frame:
                self.get_logger().info(
                    f"CameraInfo header.frame_id={sensor_frame!r} "
                    f"(config camera_frame={self._camera_frame!r})"
                )
        self.get_logger().info(
            f"CameraInfo latched from {self._camera_info_topic!r}: "
            f"size={cal.image_size} model={msg.distortion_model!r} "
            f"fx={K[0, 0]:.4f} fy={K[1, 1]:.4f} cx={K[0, 2]:.4f} cy={K[1, 2]:.4f} |D|={d_norm:.6g}"
        )
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
            self.get_logger().warning(
                f"Image {msg.width}x{msg.height} does not match CameraInfo {size}; skip frame."
            )
            return False
        return True

    def _init_map_if_possible(self) -> None:
        if self._K is None or self._repo_root is None or self._inc_map is not None:
            return
        try:
            from dataclasses import replace as dc_replace

            from pipeline.config import FeatureConfig
            from pipeline.descriptor_landmark_map import (
                DescriptorLandmarkMap,
                DescriptorMapConfig,
            )
            from pipeline.map import IncrementalMap, MapConfig
        except ImportError as e:
            self.get_logger().error(f"Could not import pipeline: {e}")
            return

        cfg = MapConfig()
        feat = FeatureConfig(method=self._feature_method, n_features=self._feature_n)
        self._inc_map = IncrementalMap(
            cfg=cfg, feat_cfg=feat, K=self._K, world_T_camera=self._world_T_camera
        )

        desc_cfg = DescriptorMapConfig.defaults(self._feature_method)
        desc_cfg = dc_replace(desc_cfg, fusion_enabled=self._descriptor_fusion_enabled)
        if self._descriptor_spatial_merge_baseline_factor >= 0.0:
            desc_cfg = dc_replace(
                desc_cfg,
                spatial_merge_baseline_factor=self._descriptor_spatial_merge_baseline_factor,
            )
        if self._descriptor_spatial_merge_radius_m >= 0.0:
            desc_cfg = dc_replace(
                desc_cfg, spatial_merge_radius_m=self._descriptor_spatial_merge_radius_m
            )
        if self._descriptor_max_match_distance >= 0.0:
            desc_cfg = dc_replace(desc_cfg, max_match_distance=self._descriptor_max_match_distance)
        if self._descriptor_merge_beta >= 0.0:
            desc_cfg = dc_replace(desc_cfg, merge_beta=self._descriptor_merge_beta)
        if self._descriptor_ratio_second_best >= 0.0:
            desc_cfg = dc_replace(desc_cfg, ratio_second_best=self._descriptor_ratio_second_best)
        self._desc_map = DescriptorLandmarkMap(desc_cfg)
        self._effective_desc_cfg = desc_cfg
        self.get_logger().info(
            f"Descriptor map: method={self._feature_method} "
            f"fusion_enabled={desc_cfg.fusion_enabled} "
            f"spatial_merge_baseline_factor={desc_cfg.spatial_merge_baseline_factor} "
            f"spatial_merge_radius_m={desc_cfg.spatial_merge_radius_m} "
            f"merge_beta={desc_cfg.merge_beta} "
            f"max_match_distance={desc_cfg.max_match_distance} "
            f"ratio_second_best={desc_cfg.ratio_second_best}"
        )

    def _cam_to_world_and_pos(self) -> tuple[np.ndarray, np.ndarray] | None:
        assert self._last_odom is not None
        allow_fb = self._last_kf_pos is not None
        T = self._world_T_camera_from_odom(self._last_odom, allow_odom_fallback=allow_fb)
        if T is None:
            return None
        return T, T[:3, 3].copy()

    def _current_metric_position(self) -> np.ndarray | None:
        resolved = self._cam_to_world_and_pos()
        if resolved is None:
            return None
        return resolved[1]

    def _travel_fraction_since_last_kf(self) -> float | None:
        """``‖pos − last_kf‖ / keyframe_distance_m``; ``None`` if no keyframe committed yet."""
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

    @staticmethod
    def _stamp_to_sec(stamp_sec: int, stamp_nsec: int) -> float:
        return float(stamp_sec) + float(stamp_nsec) * 1e-9

    def _allocate_run_dir(self, out_root: Path) -> None:
        stamp_tag = time.strftime("%Y%m%d_%H%M%S")
        self._run_dir = (out_root.resolve() / "ros2_runs" / f"run_{stamp_tag}").resolve()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        if self._save_run_on_shutdown:
            (self._run_dir / "images").mkdir(parents=True, exist_ok=True)

    def _check_bag_replay_reset(self, stamp_sec: int, stamp_nsec: int) -> None:
        if not self._bag_replay_reset_enabled:
            return
        t = self._stamp_to_sec(stamp_sec, stamp_nsec)
        if should_reset_bag_replay(self._last_message_stamp_s, t, self._bag_replay_reset_jump_s):
            self._reset_for_new_bag()
        self._last_message_stamp_s = t

    def _reset_for_new_bag(self) -> None:
        self._desc_map = None
        self._inc_map = None
        self._effective_desc_cfg = None
        self._world_T_camera.clear()
        self._world_T_camera_raw.clear()
        self._world_T_drone_raw.clear()
        self._gray_kf.clear()
        self._kf_records.clear()
        self._buffer.clear()
        self._last_kf_pos = None
        self._last_consecutive_baseline_m = None
        self._persisted = False
        self._last_message_stamp_s = None
        out_root = Path(self.get_parameter("output_root").get_parameter_value().string_value).expanduser()
        self._allocate_run_dir(out_root)
        self.get_logger().info(
            f"Bag replay detected (sim time jumped backward); sparse map cleared. "
            f"New run directory: {self._run_dir}"
        )
        if self._K is not None:
            self._init_map_if_possible()

    def _on_image(self, msg: Image) -> None:
        self._check_bag_replay_reset(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._last_image_msg = msg
        image_frame = self._normalize_tf_frame(msg.header.frame_id)
        if image_frame and not self._latched_sensor_frame_id:
            self._latched_sensor_frame_id = image_frame
        if not self._require_calibration_for_image(msg):
            return
        if self._last_odom is None:
            return
        if self._last_kf_pos is None and not self._can_resolve_camera_pose_from_odom(self._last_odom):
            self._maybe_log_waiting_for_tf(self._last_odom)
            return
        fraction = self._travel_fraction_since_last_kf()
        if not should_buffer_image(
            fraction,
            self._buffer_start_fraction,
            has_last_keyframe=self._last_kf_pos is not None,
        ):
            self._maybe_clear_buffer_for_fraction()
            self._maybe_log_image_throttle(msg)
            return
        allow_fb = self._last_kf_pos is not None or not self._apply_tf_to_camera_pose
        T, _pose_src = self._resolve_world_T_camera(
            self._last_odom,
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            allow_odom_fallback=allow_fb,
        )
        if T is None:
            return
        pos = T[:3, 3].copy()
        qx, qy, qz, qw = world_T_camera_to_quaternion_xyzw(T)
        bf = BufferedFrame(
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
        self._buffer.append(bf)
        while len(self._buffer) > self._max_buf:
            self._buffer.pop(0)
        self._try_keyframe_selection()
        self._maybe_log_image_throttle(msg)

    def _maybe_log_image_throttle(self, msg: Image) -> None:
        if self._image_log_interval_s is math.inf:
            return
        now = self.get_clock().now()
        dt = (now - self._last_image_log_time).nanoseconds * 1e-9
        if dt >= self._image_log_interval_s:
            self._last_image_log_time = now
            self.get_logger().info(
                f"Image {msg.width}x{msg.height} encoding={msg.encoding} frame={msg.header.frame_id!r}"
            )

    @staticmethod
    def _normalize_tf_frame(name: str) -> str:
        s = (name or "").strip()
        if s.startswith("/"):
            s = s[1:]
        return s

    def _record_tf_edges(self, transforms, *, static: bool) -> None:
        for tr in transforms:
            parent = self._normalize_tf_frame(tr.header.frame_id)
            child = self._normalize_tf_frame(tr.child_frame_id)
            if not parent or not child:
                continue
            edge = (parent, child)
            self._tf_edges.add(edge)
            if static:
                self._tf_static_edges.add(edge)

    def _log_tf_set_failure(self, tr, exc: TransformException) -> None:
        if self._tf_set_fail_count >= 8:
            return
        self._tf_set_fail_count += 1
        self.get_logger().warning(
            f"TF buffer rejected {tr.header.frame_id!r} -> {tr.child_frame_id!r}: {exc}"
        )

    def _maybe_log_tf_static_summary(self) -> None:
        if self._tf_static_summary_logged or self._tf_static_message_count < 3:
            return
        self._tf_static_summary_logged = True
        edges = sorted(f"{p}->{c}" for p, c in self._tf_static_edges)
        self.get_logger().info(
            f"tf_static complete ({len(edges)} edges): {edges if edges else '(none recorded)'}"
        )

    def _camera_frame_candidates(self) -> list[str]:
        cands: list[str] = []
        for raw in (self._latched_sensor_frame_id, self._camera_frame):
            n = self._normalize_tf_frame(raw)
            if n and n not in cands:
                cands.append(n)
        for _parent, child in sorted(self._tf_edges):
            if any(k in child for k in ("optical", "rgb_camera", "camera")):
                if child not in cands:
                    cands.append(child)
        return cands or [self._camera_frame]

    def _extrinsic_parent_candidates(self, msg: Odometry) -> list[str]:
        out: list[str] = []
        child = self._odom_child_frame_id(msg)
        base = self._normalize_tf_frame(self._base_frame)
        for parent in (child, base):
            if parent and parent not in out:
                out.append(parent)
        cam_set = set(self._camera_frame_candidates())
        for parent, child_frame in self._tf_edges:
            if child_frame in cam_set and parent not in out:
                out.append(parent)
        return out

    def _init_tf_subscriptions(self) -> None:
        """Subscribe ``/tf`` + ``/tf_static`` with rosbag-compatible QoS (not default ``TransformListener``)."""
        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        tf_qos = _tf_rosbag_qos()
        static_qos = tf_qos if self._tf_static_volatile_qos else _tf_static_live_qos()
        self.create_subscription(TFMessage, "/tf", self._on_tf_message, tf_qos)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static_message, static_qos)
        self.get_logger().info(
            "TF subscriptions: /tf + /tf_static "
            f"(static_qos={'volatile' if self._tf_static_volatile_qos else 'transient_local'})"
        )

    def _log_tf_startup_hints(self) -> None:
        use_sim = False
        try:
            use_sim = bool(self.get_parameter("use_sim_time").value)
        except Exception:
            pass
        if not use_sim:
            self.get_logger().warning(
                "apply_tf_to_camera_pose=true but use_sim_time is false. "
                "For rosbag: play with --clock and run --ros-args -p use_sim_time:=true"
            )
        self.get_logger().info(
            "Rosbag TF tip: start the bag from t=0 AFTER this node is running "
            "(tf_static has only 3 msgs at bag start; volatile replay will not repeat them)."
        )

    def _on_tf_message(self, msg: TFMessage) -> None:
        if self._tf_buffer is None:
            return
        for tr in msg.transforms:
            try:
                self._tf_buffer.set_transform(tr, "incremental_vo_ros2")
            except TransformException as e:
                self._log_tf_set_failure(tr, e)
        self._record_tf_edges(msg.transforms, static=False)
        self._tf_message_count += 1

    def _on_tf_static_message(self, msg: TFMessage) -> None:
        if self._tf_buffer is None:
            return
        for tr in msg.transforms:
            try:
                self._tf_buffer.set_transform_static(tr, "incremental_vo_ros2")
            except TransformException as e:
                try:
                    self._tf_buffer.set_transform(tr, "incremental_vo_ros2")
                except TransformException as e2:
                    self._log_tf_set_failure(tr, e2)
                else:
                    self._log_tf_set_failure(tr, e)
        self._record_tf_edges(msg.transforms, static=True)
        self._tf_static_message_count += 1
        if self._tf_static_message_count == 1:
            frames = [self._normalize_tf_frame(t.child_frame_id) for t in msg.transforms]
            self.get_logger().info(f"tf_static latched ({len(frames)} transforms), e.g. {frames[:4]}")
        self._maybe_log_tf_static_summary()

    def _tf_time_from_odom(self, msg: Odometry) -> Time:
        return self._tf_time_from_stamp(msg.header.stamp.sec, msg.header.stamp.nanosec)

    def _tf_time_from_stamp(self, sec: int, nsec: int) -> Time:
        if self._tf_use_latest or (sec == 0 and nsec == 0):
            return Time()
        from builtin_interfaces.msg import Time as TimeMsg

        return Time.from_msg(TimeMsg(sec=int(sec), nanosec=int(nsec)))

    def _tf_lookup_stamps_from_time(
        self, sec: int, nsec: int, *, allow_latest_fallback: bool
    ) -> list[Time]:
        stamps: list[Time] = [self._tf_time_from_stamp(sec, nsec)]
        if allow_latest_fallback and not self._tf_use_latest:
            stamps.append(Time())
        return stamps

    def _maybe_warn_tf(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_tf_warn_time >= 5.0:
            self._last_tf_warn_time = now
            self.get_logger().warning(message)

    def _odom_child_frame_id(self, msg: Odometry) -> str:
        return self._normalize_tf_frame(msg.child_frame_id)

    def _lookup_transform_to_matrix(
        self,
        parent_frame: str,
        child_frame: str,
        stamps: list[Time],
    ) -> np.ndarray | None:
        parent_frame = self._normalize_tf_frame(parent_frame)
        child_frame = self._normalize_tf_frame(child_frame)
        if self._tf_buffer is None or not parent_frame or not child_frame:
            return None
        for stamp in stamps:
            try:
                if not self._tf_buffer.can_transform(
                    parent_frame,
                    child_frame,
                    stamp,
                    timeout=Duration(seconds=1.0),
                ):
                    continue
                t = self._tf_buffer.lookup_transform(
                    parent_frame,
                    child_frame,
                    stamp,
                    timeout=Duration(seconds=0.5),
                )
                return transform_stamped_to_world_T(t)
            except TransformException:
                continue
        return None

    def _tf_stamps_for_lookup(self, stamp_sec: int, stamp_nsec: int) -> list[Time]:
        """Prefer latest TF (rosbag); optionally also try odom/image stamp."""
        stamps: list[Time] = [Time()]
        if not self._tf_use_latest:
            stamps.append(self._tf_time_from_stamp(stamp_sec, stamp_nsec))
        return stamps

    def _lookup_world_T_camera_via_tf(
        self,
        msg: Odometry,
        stamps: list[Time],
    ) -> tuple[np.ndarray | None, str, str]:
        """
        Resolve optical ``world_T_camera`` using TF.

        Returns ``(T, source, camera_frame)`` with source one of
        ``tf_direct``, ``tf_compose``, ``tf_compose_via_base``, or ``tf_missing``.
        """
        world = self._normalize_tf_frame(msg.header.frame_id)
        child = self._odom_child_frame_id(msg)
        world_T_child = odom_to_cam_to_world_T(msg)
        base = self._normalize_tf_frame(self._base_frame)

        for camera in self._camera_frame_candidates():
            if world:
                T_direct = self._lookup_transform_to_matrix(world, camera, stamps)
                if T_direct is not None:
                    return T_direct, "tf_direct", camera
            if not child:
                continue
            for parent in self._extrinsic_parent_candidates(msg):
                T_parent_cam = self._lookup_transform_to_matrix(parent, camera, stamps)
                if T_parent_cam is None:
                    continue
                if parent == child:
                    return world_T_child @ T_parent_cam, "tf_compose", camera
                T_child_parent = self._lookup_transform_to_matrix(child, parent, stamps)
                if T_child_parent is not None:
                    src = (
                        "tf_compose_via_base"
                        if parent == base
                        else "tf_compose"
                    )
                    return world_T_child @ T_child_parent @ T_parent_cam, src, camera
        return None, "tf_missing", ""

    def _resolve_world_T_camera(
        self,
        msg: Odometry,
        stamp_sec: int,
        stamp_nsec: int,
        *,
        allow_odom_fallback: bool,
    ) -> tuple[np.ndarray | None, str]:
        """
        Optical ``world_T_camera`` in ``msg.header.frame_id``.

        Returns ``(T, source)`` with source one of:
        ``odom_raw``, ``odom_child_is_camera``, ``tf_direct``, ``tf_compose``,
        ``tf_compose_via_base``, ``odom_body_fallback``, ``tf_missing``.
        """
        if not self._apply_tf_to_camera_pose or self._tf_buffer is None:
            return odom_to_cam_to_world_T(msg), "odom_raw"
        child = self._odom_child_frame_id(msg)
        for camera in self._camera_frame_candidates():
            if child and child == camera:
                return odom_to_cam_to_world_T(msg), "odom_child_is_camera"
        stamps = self._tf_stamps_for_lookup(stamp_sec, stamp_nsec)
        T, source, camera = self._lookup_world_T_camera_via_tf(msg, stamps)
        if T is not None:
            if camera:
                self._resolved_camera_frame = camera
            return T, source
        if not allow_odom_fallback:
            return None, "tf_missing"
        self._maybe_warn_tf(
            f"TF could not resolve optical pose (world={msg.header.frame_id!r}, "
            f"odom_child={child!r}, camera_candidates={self._camera_frame_candidates()!r}); "
            "using odom child pose (body frame)."
        )
        return odom_to_cam_to_world_T(msg), "odom_body_fallback"

    def _maybe_log_waiting_for_tf(self, msg: Odometry) -> None:
        now = time.monotonic()
        if now - self._last_tf_wait_log_time >= 5.0:
            self._last_tf_wait_log_time = now
            child = self._odom_child_frame_id(msg) or "?"
            world = self._normalize_tf_frame(msg.header.frame_id)
            cams = self._camera_frame_candidates()
            cam_hint = cams[0] if cams else self._camera_frame
            related = sorted(
                f"{p}->{c}"
                for p, c in self._tf_edges
                if any(
                    k in p or k in c
                    for k in ("fcu", "camera", "rgb", "optical", "base", "local_origin")
                )
            )
            edge_hint = related[:12] if related else ["(no camera/fcu edges seen yet)"]
            self.get_logger().info(
                f"Waiting for optical TF (odom child={child!r} -> camera ~{cam_hint!r}, "
                f"world={world!r}) before first keyframe. "
                f"Received: /tf msgs={self._tf_message_count} /tf_static msgs="
                f"{self._tf_static_message_count}. "
                f"TF edges (sample): {edge_hint}. "
                "If tf_static=0: restart bag from beginning (--clock) while this node runs."
            )

    def _can_resolve_camera_pose_from_odom(self, msg: Odometry) -> bool:
        """True when TF can supply optical ``world_T_camera`` (full chain or extrinsic)."""
        if not self._apply_tf_to_camera_pose:
            return True
        if self._tf_static_volatile_qos and self._tf_static_message_count < 1:
            return False
        stamps = self._tf_stamps_for_lookup(msg.header.stamp.sec, msg.header.stamp.nanosec)
        T, _source, _camera = self._lookup_world_T_camera_via_tf(msg, stamps)
        return T is not None

    def _world_T_camera_from_odom(
        self, msg: Odometry, *, allow_odom_fallback: bool = True
    ) -> np.ndarray | None:
        T, _ = self._resolve_world_T_camera(
            msg,
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            allow_odom_fallback=allow_odom_fallback,
        )
        return T

    def _log_keyframe0_optical_axes(
        self, idx: int, world_T_camera: np.ndarray, parent: str, pose_source: str
    ) -> None:
        if idx != 0:
            return
        R = np.asarray(world_T_camera, dtype=np.float64)[:3, :3]
        z_ax = R[:, 2]
        x_ax = R[:, 0]
        self.get_logger().info(
            f"Keyframe 0 optical axes in {parent!r} (pose_source={pose_source!r}): "
            f"+Z(view)=({z_ax[0]:.3f},{z_ax[1]:.3f},{z_ax[2]:.3f}) "
            f"+X(right)=({x_ax[0]:.3f},{x_ax[1]:.3f},{x_ax[2]:.3f})"
        )
        if pose_source == "odom_body_fallback":
            self.get_logger().error(
                "W0_raw is FCU/body (no optical TF): sparse map will stretch along world Z. "
                "Fix TF or set apply_tf_to_camera_pose=false only if odom child is already optical."
            )
        elif abs(float(z_ax[2])) > 0.85:
            self.get_logger().warning(
                "Optical +Z is nearly aligned with world Z (points up). "
                "Expected mostly horizontal for forward flight — check camera_frame and /tf."
            )

    def _camera_pose_debug_parent_frame(self) -> str:
        if self._camera_pose_debug_frame_id:
            return self._camera_pose_debug_frame_id
        if self._last_odom is not None and self._last_odom.header.frame_id:
            return self._last_odom.header.frame_id
        return ""

    def _publish_camera_pose_debug(self, stamp) -> None:
        if not self._camera_pose_debug_enabled or self._last_odom is None:
            return
        parent = self._camera_pose_debug_parent_frame()
        if not parent:
            return
        resolved = self._cam_to_world_and_pos()
        if resolved is None:
            return
        T, _ = resolved
        hdr = Header()
        hdr.stamp = stamp
        hdr.frame_id = parent
        pose_pub = self._camera_pose_debug_pub
        if pose_pub is not None:
            pose_pub.publish(world_T_camera_to_pose_stamped(T, header=hdr))
        tf_pub = self._camera_pose_tf_broadcaster
        if tf_pub is not None:
            tf_pub.sendTransform(
                world_T_camera_to_transform_stamped(
                    T,
                    header=hdr,
                    child_frame_id=self._camera_pose_debug_child_frame_id,
                )
            )

    def _on_odom_main(self, msg: Odometry) -> None:
        self._check_bag_replay_reset(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._last_odom = msg
        self._publish_camera_pose_debug(msg.header.stamp)
        self._maybe_clear_buffer_for_fraction()
        self._try_keyframe_selection()

    def _on_odom_gt(self, msg: Odometry) -> None:
        self._last_odom_gt = msg

    def _try_keyframe_selection(self) -> None:
        if self._last_odom is None or self._K is None or not self._buffer:
            return
        if self._inc_map is None:
            self._init_map_if_possible()
        if self._inc_map is None:
            return

        # First keyframe: require optical-frame TF when apply_tf_to_camera_pose (locks W0_raw).
        if self._last_kf_pos is None:
            if not self._can_resolve_camera_pose_from_odom(self._last_odom):
                self._maybe_log_waiting_for_tf(self._last_odom)
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
        best_i = min(range(len(self._buffer)), key=lambda i: float(np.linalg.norm(self._buffer[i].pos_odom - target)))
        chosen = self._buffer[best_i]
        self._buffer.clear()
        self._commit_keyframe(chosen, distance_trigger_m=dist)

    def _commit_keyframe(self, bf: BufferedFrame, *, distance_trigger_m: float | None) -> None:
        from pipeline.geometry import canonicalize_world_T_camera_to_first

        if self._last_odom is None:
            return
        allow_fb = not self._apply_tf_to_camera_pose
        T_opt, pose_src = self._resolve_world_T_camera(
            self._last_odom,
            bf.stamp_sec,
            bf.stamp_nsec,
            allow_odom_fallback=allow_fb,
        )
        if T_opt is None:
            self.get_logger().warning(
                f"Keyframe skip: no optical TF at image stamp "
                f"{bf.stamp_sec}.{bf.stamp_nsec:09d} (pose_source=tf_missing)."
            )
            return
        bf.cam_to_world = T_opt
        qx, qy, qz, qw = world_T_camera_to_quaternion_xyzw(T_opt)
        bf.qx, bf.qy, bf.qz, bf.qw = float(qx), float(qy), float(qz), float(qw)
        bf.pos_odom = T_opt[:3, 3].copy()

        gray, k_eff = image_msg_to_gray_undistorted(bf.image_msg, self._calibration)
        if gray is None:
            self.get_logger().warn(
                f"Keyframe skip: unsupported encoding {bf.image_msg.encoding!r} "
                f"(stamp {bf.stamp_sec}.{bf.stamp_nsec:09d})"
            )
            return
        if k_eff is not None and (self._K is None or not np.allclose(self._K, k_eff)):
            self._K = k_eff
            self._init_map_if_possible()

        idx = len(self._world_T_camera_raw)

        log_path = ""
        manifest_path = ""
        if self._offline_dataset_dir is not None:
            stem = offline_dataset_image_basename(self._offline_image_prefix, idx)
            offline_path = self._offline_dataset_dir / "images" / stem
            cv2.imwrite(str(offline_path), gray)
            self._offline_motion_frames.append(
                {"index": idx, "filename": stem, "T": bf.cam_to_world.tolist()}
            )
            self._write_offline_motion_json()
            log_path = f"images/{stem}"

        if self._save_run_on_shutdown:
            kf_rel = f"images/kf_{idx:05d}.png"
            cv2.imwrite(str(self._run_dir / kf_rel), gray)
            manifest_path = kf_rel
            if not log_path:
                log_path = kf_rel
        elif log_path:
            manifest_path = log_path

        self._world_T_camera_raw.append(bf.cam_to_world.copy())
        self._world_T_drone_raw.append(odom_to_cam_to_world_T(self._last_odom))
        self._world_T_camera[:] = canonicalize_world_T_camera_to_first(self._world_T_camera_raw)
        self._gray_kf.append(gray)

        rec = {
            "index": idx,
            "stamp": {"sec": bf.stamp_sec, "nanosec": bf.stamp_nsec},
            "position": {"x": float(bf.pos_odom[0]), "y": float(bf.pos_odom[1]), "z": float(bf.pos_odom[2])},
            "orientation_xyzw": {
                "x": bf.qx,
                "y": bf.qy,
                "z": bf.qz,
                "w": bf.qw,
            },
            "image_path": manifest_path.replace("\\", "/"),
            "distance_from_previous_keyframe_m": distance_trigger_m,
        }
        self._kf_records.append(rec)
        self._last_kf_pos = bf.pos_odom.copy()
        parent_frame = self._camera_pose_debug_parent_frame() or "world"
        self._log_keyframe0_optical_axes(idx, bf.cam_to_world, parent_frame, pose_src)
        if self._log_frame_transforms and idx == 0 and self._last_odom is not None:
            world_frame = self._normalize_tf_frame(self._last_odom.header.frame_id) or parent_frame
            drone_frame = self._odom_child_frame_id(self._last_odom) or "?"
            camera_frame = (
                self._resolved_camera_frame
                or self._normalize_tf_frame(self._camera_frame)
                or "?"
            )
            log_frame_transforms_block(
                self.get_logger(),
                idx=idx,
                pose_source=pose_src,
                world_frame=world_frame,
                drone_frame=drone_frame,
                camera_frame=camera_frame,
                world_T_drone=self._world_T_drone_raw[-1],
                world_T_camera=bf.cam_to_world,
            )

        saved = log_path or f"keyframe {idx} (no image export)"
        msg = (
            f"Keyframe {idx}: saved {saved} | pos=({bf.pos_odom[0]:.3f},{bf.pos_odom[1]:.3f},{bf.pos_odom[2]:.3f})"
            f" | pose_source={pose_src!r}"
        )
        if distance_trigger_m is not None:
            msg += f" | odom spacing ~{distance_trigger_m:.3f} m (threshold {self._d})"
        self.get_logger().info(msg)
        self._publish_keyframe_z_marker(idx, bf.cam_to_world, bf.stamp_sec, bf.stamp_nsec)

        max_range_world: float | None = None
        if idx >= 1 and len(self._world_T_camera) > idx:
            baseline_m = consecutive_keyframe_baseline_m(
                self._world_T_camera[idx - 1], self._world_T_camera[idx]
            )
            self._last_consecutive_baseline_m = baseline_m
            max_range_world = max_sparse_range_m(baseline_m, self._sparse_map_range_factor)

        if idx >= 1 and self._inc_map is not None:
            from pipeline.descriptor_landmark_map import pose_translation_baseline_m

            for off in range(1, min(self._pair_lookback, idx) + 1):
                i = idx - off
                try:
                    tw = self._inc_map.add_frame_pair(
                        i, idx, self._gray_kf[i], self._gray_kf[idx]
                    )
                    if (
                        tw.scale_ok
                        and self._desc_map is not None
                        and len(self._world_T_camera_raw) > i
                    ):
                        try:
                            pair_baseline_m = pose_translation_baseline_m(
                                self._world_T_camera[i], self._world_T_camera[idx]
                            )
                            self._desc_map.integrate(
                                tw,
                                world_T_camera_raw=self._world_T_camera_raw[i],
                                world_T_drone_raw=self._world_T_drone_raw[i],
                                world_T_camera_j_raw=self._world_T_camera_raw[idx],
                                max_range_world=max_range_world,
                                pair_baseline_m=pair_baseline_m,
                            )
                        except Exception as ex:
                            self.get_logger().warn(
                                f"DescriptorLandmarkMap.integrate failed for ({i}->{idx}): {ex}"
                            )
                    n_desc = len(self._desc_map.landmarks) if self._desc_map is not None else 0
                    self.get_logger().info(
                        f"Two-view {i}->{idx}: triangulated cols={tw.X_world_h.shape[1]} "
                        f"descriptor_landmarks_total={n_desc} reproj={tw.reproj!r}"
                    )
                    if (
                        self._log_random_triangulated_point
                        and off == 1
                        and tw.scale_ok
                        and len(self._world_T_camera_raw) > i
                    ):
                        self._log_random_triangulated_point_debug(
                            idx,
                            tw,
                            frame_i=i,
                            world_T_camera_raw=self._world_T_camera_raw[i],
                            world_T_drone_raw=self._world_T_drone_raw[i],
                            world_T_camera_j_raw=self._world_T_camera_raw[idx],
                            max_range_world=max_range_world,
                        )
                except Exception as e:
                    self.get_logger().error(f"add_frame_pair failed for ({i}->{idx}): {e}")

    def _log_random_triangulated_point_debug(
        self,
        keyframe_idx: int,
        tw,
        *,
        frame_i: int,
        world_T_camera_raw: np.ndarray,
        world_T_drone_raw: np.ndarray,
        world_T_camera_j_raw: np.ndarray,
        max_range_world: float | None,
    ) -> None:
        from pipeline.triangulation_debug import (
            format_triangulation_debug_line,
            sample_random_integrate_point,
        )

        sampled = sample_random_integrate_point(
            tw,
            world_T_camera_raw=world_T_camera_raw,
            world_T_drone_raw=world_T_drone_raw,
            world_T_camera_j_raw=world_T_camera_j_raw,
            max_range_world=max_range_world,
        )
        if sampled is None:
            return
        col_k, X_cam, X_drone, X_world = sampled
        self.get_logger().info(
            format_triangulation_debug_line(
                keyframe_idx=keyframe_idx,
                frame_i=frame_i,
                frame_j=tw.frame_j,
                col_k=col_k,
                X_cam=X_cam,
                X_drone=X_drone,
                X_world=X_world,
            )
        )

    def _publish_keyframe_z_marker(
        self,
        idx: int,
        world_T_camera: np.ndarray,
        stamp_sec: int,
        stamp_nsec: int,
    ) -> None:
        pub = self._keyframe_marker_pub
        if pub is None:
            return
        parent = self._keyframe_marker_frame_id or self._camera_pose_debug_parent_frame()
        if not parent:
            return
        from builtin_interfaces.msg import Time as TimeMsg
        from geometry_msgs.msg import Point

        pose = world_T_camera_to_pose_stamped(
            world_T_camera,
            header=Header(stamp=TimeMsg(sec=stamp_sec, nanosec=stamp_nsec), frame_id=parent),
        )
        length = max(0.05, float(self._keyframe_marker_length_m))
        m = Marker()
        m.header = pose.header
        m.ns = "keyframe_camera_z"
        m.id = int(idx)
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose = pose.pose
        m.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=length)]
        m.scale.x = 0.03
        m.color.r = 0.1
        m.color.g = 0.9
        m.color.b = 1.0
        m.color.a = 1.0
        pub.publish(m)

    def _on_tf_timer(self) -> None:
        if self._tf_buffer is None or self._last_image_msg is None:
            return
        if self._tf_use_latest or (
            self._last_image_msg.header.stamp.sec == 0
            and self._last_image_msg.header.stamp.nanosec == 0
        ):
            stamp = Time()
        else:
            stamp = Time.from_msg(self._last_image_msg.header.stamp)
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._camera_frame,
                stamp,
                timeout=Duration(seconds=0.5),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self._base_frame!r} <- {self._camera_frame!r} @ image stamp failed: {e}"
            )
            return
        tr = t.transform.translation
        self.get_logger().info(
            f"camera in base: t=({tr.x:.3f},{tr.y:.3f},{tr.z:.3f}) "
            f"(image stamp {self._last_image_msg.header.stamp.sec}.{self._last_image_msg.header.stamp.nanosec:09d})"
        )

    def _on_sparse_map_timer(self) -> None:
        pub = self._sparse_map_pub
        if pub is None or self._desc_map is None:
            return
        pts_w = self._desc_map.positions_world()
        if pts_w.size == 0:
            return
        if self._sparse_map_frame_id_override:
            fid = self._sparse_map_frame_id_override
        elif self._last_odom is not None:
            fid = self._last_odom.header.frame_id
        else:
            return
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = fid
        tuples = [(float(r[0]), float(r[1]), float(r[2])) for r in pts_w]
        cloud = point_cloud2.create_cloud(hdr, self._sparse_map_fields, tuples)
        pub.publish(cloud)

    def _write_offline_motion_json(self) -> None:
        if self._offline_dataset_dir is None or not self._offline_motion_frames:
            return
        if self._repo_root is None:
            return
        try:
            from data.io_json import save_motion_json

            transforms = [
                np.asarray(fr["T"], dtype=np.float64) for fr in self._offline_motion_frames
            ]
            filenames = [str(fr["filename"]) for fr in self._offline_motion_frames]
            save_motion_json(
                self._offline_dataset_dir / "motion.json",
                transforms,
                filenames=filenames,
            )
        except Exception as e:
            self.get_logger().warn(f"save_motion_json failed: {e}")

    def persist_offline_dataset(self) -> None:
        if self._offline_dataset_dir is None:
            return
        if self._calibration is not None and self._repo_root is not None:
            try:
                from data.io_json import save_calibration_json

                save_calibration_json(
                    self._offline_dataset_dir / "calibration.json", self._calibration
                )
            except Exception as e:
                self.get_logger().warn(f"offline save_calibration_json failed: {e}")
        self._write_offline_motion_json()
        n = len(self._offline_motion_frames)
        self.get_logger().info(
            f"Offline dataset: {self._offline_dataset_dir} ({n} frames)"
        )

    def persist_run(self) -> None:
        if self._persisted:
            return
        self._persisted = True
        if self._calibration is not None and self._repo_root is not None:
            try:
                from data.io_json import save_calibration_json

                save_calibration_json(self._run_dir / "calibration.json", self._calibration)
            except Exception as e:
                self.get_logger().warn(f"save_calibration_json failed: {e}")
        odom = self._last_odom
        frame_id = self._last_image_msg.header.frame_id if self._last_image_msg else ""
        child = odom.child_frame_id if odom else ""
        header_frame = odom.header.frame_id if odom else ""
        save_keyframe_manifest(
            self._run_dir / "position.json",
            frame_id=frame_id,
            keyframe_distance_m=self._d,
            pair_lookback=self._pair_lookback,
            odom_child_frame=child,
            odom_header_frame=header_frame,
            records=self._kf_records,
            feature_method=self._feature_method,
            feature_n_features=self._feature_n,
            descriptor_merge_beta=(
                self._effective_desc_cfg.merge_beta if self._effective_desc_cfg is not None else None
            ),
            descriptor_max_match_distance=(
                self._effective_desc_cfg.max_match_distance
                if self._effective_desc_cfg is not None
                else None
            ),
            descriptor_ratio_second_best=(
                self._effective_desc_cfg.ratio_second_best
                if self._effective_desc_cfg is not None
                else None
            ),
            landmarks_reference_frame=header_frame if self._desc_map is not None else None,
            map_coordinate_frame=header_frame if self._desc_map is not None else None,
            eval_world_T_camera0_flat16=(
                self._eval_world_T_cam0.reshape(16).tolist()
                if self._eval_world_T_cam0 is not None
                else None
            ),
        )
        pts = np.zeros((0, 3), dtype=np.float64)
        if self._desc_map is not None:
            pts = self._desc_map.positions_world()
        save_sparse_map_npz(self._run_dir / "sparse_map.npz", pts)
        if self._eval_world_T_cam0 is not None and self._world_T_camera_raw:
            from pipeline.geometry import invert_se3

            world_T_eval = self._eval_world_T_cam0 @ invert_se3(self._world_T_camera_raw[0])
            save_sparse_map_eval_world_npz(
                self._run_dir / "sparse_map_eval_world.npz", pts, world_T_eval
            )
        if self._desc_map is not None:
            try:
                from pipeline.descriptor_landmark_map import export_landmarks_csv

                export_landmarks_csv(
                    self._run_dir / "descriptor_landmarks.csv", self._desc_map
                )
            except Exception as e:
                self.get_logger().warn(f"export_landmarks_csv failed: {e}")
        self.get_logger().info(
            f"Saved run: {self._run_dir} ({len(self._kf_records)} keyframes, "
            f"{pts.shape[0]} descriptor landmarks in odom world frame)"
        )

    def destroy_node(self) -> None:
        try:
            if self._save_run_on_shutdown:
                self.persist_run()
            if self._export_offline_dataset:
                self.persist_offline_dataset()
        finally:
            super().destroy_node()

    @property
    def last_odom_gt(self) -> Odometry | None:
        return self._last_odom_gt


def main(args: list[str] | None = None) -> None:
    argv = apply_config_to_argv(list(sys.argv[1:] if args is None else args))
    rclpy.init(args=argv)
    node = IncrementalVoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

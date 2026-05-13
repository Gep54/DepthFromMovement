#!/usr/bin/env python3
"""ROS 2 node: live sparse 3D map from monocular keyframes + metric pose fusion + two-view triangulation.

Odometry ``pose`` updates the primary metric track. An optional ``geometry_msgs/PoseStamped`` topic
supplies a second ``world_T_camera`` estimate (e.g. from optical flow upstream); :mod:`pipeline.metric_fusion`
combines tracks before ``IncrementalMap`` consumes poses. If your camera is offset from ``child_frame_id``,
fuse TF into ``cam_to_world`` before this node.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
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
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

from incremental_vo_ros2.support import (
    BufferedFrame,
    ensure_pipeline_on_path,
    odom_position_xyz,
    odom_to_cam_to_world_T,
    pose_stamped_to_world_T_camera,
    ros_image_to_gray,
    save_keyframe_manifest,
    save_sparse_map_npz,
    world_T_camera_to_quaternion_xyzw,
)


def _sensor_data_qos() -> QoSProfile:
    # Cameras usually publish best-effort; depth 5 drops old frames under load instead of blocking.
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


def _odom_qos() -> QoSProfile:
    # Odom streams are usually reliable; if your publisher uses different QoS, match it or use a remap.
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class IncrementalVoNode(Node):
    """Distance-based keyframes, pluggable metric pose fusion, two-view sparse triangulation."""

    def __init__(self) -> None:
        super().__init__("incremental_vo_node", automatically_declare_parameters_from_overrides=True)

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
        self.declare_parameter("output_root", ".")
        self.declare_parameter("max_image_buffer", 400)
        self.declare_parameter("camera_fx", 600.0)
        self.declare_parameter("camera_fy", 600.0)
        self.declare_parameter("camera_cx", -1.0)
        self.declare_parameter("camera_cy", -1.0)
        self.declare_parameter("pair_lookback", 1)

        # Optional TF debug (off by default for rosbag-focused runs).
        self.declare_parameter("base_frame", "uav1/base_link")
        self.declare_parameter("camera_frame", "uav1/stereo/left_optical")
        self.declare_parameter("tf_lookup_period_s", 0.0)
        self.declare_parameter("tf_use_latest_time", False)
        self.declare_parameter("log_image_hz", 0.0)

        # Descriptor-based landmark fusion (replace-if-better descriptors + EMA position update).
        # Sentinel ``-1.0`` means "use DescriptorMapConfig.defaults(method)" / ``None`` for nullable
        # fields, mirroring the EKF-sigma block convention below.
        self.declare_parameter("feature_method", "ORB")
        self.declare_parameter("feature_n_features", 2000)
        self.declare_parameter("descriptor_merge_beta", -1.0)
        self.declare_parameter("descriptor_max_match_distance", -1.0)
        self.declare_parameter("descriptor_ratio_second_best", -1.0)

        # Metric pose fusion (odometry vs optional external ``world_T_camera``).
        self.declare_parameter("fusion_method", "position_blend")
        self.declare_parameter("fusion_position_blend_weight", 0.5)
        self.declare_parameter("provided_pose_topic", "")
        self.declare_parameter("velocity_topic", "")
        self.declare_parameter("ekf_sigma_process_pos", 0.05)
        self.declare_parameter("ekf_sigma_process_vel", 0.5)
        self.declare_parameter("ekf_sigma_odom_position", 0.02)
        self.declare_parameter("ekf_sigma_velocity", 0.3)
        self.declare_parameter("ekf_sigma_vo_position", 2.0)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        odom_main_topic = self.get_parameter("odom_main_topic").get_parameter_value().string_value
        subscribe_gt = self.get_parameter("subscribe_odom_gt").get_parameter_value().bool_value
        odom_gt_topic = self.get_parameter("odom_gt_topic").get_parameter_value().string_value

        self._d = float(self.get_parameter("keyframe_distance_m").get_parameter_value().double_value)
        out_root = Path(self.get_parameter("output_root").get_parameter_value().string_value).expanduser()
        self._max_buf = max(16, int(self.get_parameter("max_image_buffer").value))
        self._pair_lookback = max(1, int(self.get_parameter("pair_lookback").value))
        self._fx = float(self.get_parameter("camera_fx").get_parameter_value().double_value)
        self._fy = float(self.get_parameter("camera_fy").get_parameter_value().double_value)
        self._cx_auto = float(self.get_parameter("camera_cx").get_parameter_value().double_value)
        self._cy_auto = float(self.get_parameter("camera_cy").get_parameter_value().double_value)

        self._base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self._camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        tf_period = self.get_parameter("tf_lookup_period_s").get_parameter_value().double_value
        self._tf_use_latest = (
            self.get_parameter("tf_use_latest_time").get_parameter_value().bool_value
        )
        log_hz = self.get_parameter("log_image_hz").get_parameter_value().double_value
        self._image_log_interval_s = 1.0 / log_hz if log_hz > 0.0 else math.inf
        self._last_image_log_time = self.get_clock().now()

        self._fusion_method_str = (
            self.get_parameter("fusion_method").get_parameter_value().string_value.strip()
        )
        self._fusion_position_blend_w = float(
            self.get_parameter("fusion_position_blend_weight").get_parameter_value().double_value
        )
        self._provided_pose_topic = (
            self.get_parameter("provided_pose_topic").get_parameter_value().string_value.strip()
        )
        self._velocity_topic = (
            self.get_parameter("velocity_topic").get_parameter_value().string_value.strip()
        )
        self._ekf_sigma_process_pos = float(
            self.get_parameter("ekf_sigma_process_pos").get_parameter_value().double_value
        )
        self._ekf_sigma_process_vel = float(
            self.get_parameter("ekf_sigma_process_vel").get_parameter_value().double_value
        )
        self._ekf_sigma_odom_position = float(
            self.get_parameter("ekf_sigma_odom_position").get_parameter_value().double_value
        )
        self._ekf_sigma_velocity = float(
            self.get_parameter("ekf_sigma_velocity").get_parameter_value().double_value
        )
        self._ekf_sigma_vo_position = float(
            self.get_parameter("ekf_sigma_vo_position").get_parameter_value().double_value
        )

        self._feature_method = (
            self.get_parameter("feature_method").get_parameter_value().string_value.strip().upper()
        )
        if self._feature_method not in ("ORB", "SIFT"):
            self.get_logger().warning(
                f"feature_method={self._feature_method!r} not in {{ORB,SIFT}}; falling back to ORB."
            )
            self._feature_method = "ORB"
        self._feature_n = max(1, int(self.get_parameter("feature_n_features").value))
        self._descriptor_merge_beta = float(
            self.get_parameter("descriptor_merge_beta").get_parameter_value().double_value
        )
        self._descriptor_max_match_distance = float(
            self.get_parameter("descriptor_max_match_distance").get_parameter_value().double_value
        )
        self._descriptor_ratio_second_best = float(
            self.get_parameter("descriptor_ratio_second_best").get_parameter_value().double_value
        )

        self._pose_fusion = None
        if repo is not None:
            try:
                from pipeline.metric_fusion import create_metric_pose_fusion

                self._pose_fusion = create_metric_pose_fusion(
                    self._fusion_method_str,
                    position_blend_weight=self._fusion_position_blend_w,
                    ekf_sigma_process_pos=self._ekf_sigma_process_pos,
                    ekf_sigma_process_vel=self._ekf_sigma_process_vel,
                    ekf_sigma_odom_position=self._ekf_sigma_odom_position,
                    ekf_sigma_velocity=self._ekf_sigma_velocity,
                    ekf_sigma_vo_position=self._ekf_sigma_vo_position,
                )
            except ValueError as e:
                self.get_logger().warning(f"{e}; falling back to fusion_method=odom_only.")
                self._fusion_method_str = "odom_only"
                self._pose_fusion = create_metric_pose_fusion(
                    "odom_only",
                    position_blend_weight=self._fusion_position_blend_w,
                    ekf_sigma_process_pos=self._ekf_sigma_process_pos,
                    ekf_sigma_process_vel=self._ekf_sigma_process_vel,
                    ekf_sigma_odom_position=self._ekf_sigma_odom_position,
                    ekf_sigma_velocity=self._ekf_sigma_velocity,
                    ekf_sigma_vo_position=self._ekf_sigma_vo_position,
                )

        stamp_tag = time.strftime("%Y%m%d_%H%M%S")
        self._run_dir = (out_root.resolve() / "ros2_runs" / f"run_{stamp_tag}").resolve()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "images").mkdir(exist_ok=True)

        self._K: np.ndarray | None = None
        self._last_odom: Odometry | None = None
        self._last_image_msg: Image | None = None
        self._buffer: list[BufferedFrame] = []
        self._world_T_camera: list[np.ndarray] = []
        self._gray_kf: list[np.ndarray] = []
        self._kf_records: list[dict] = []
        self._last_kf_pos: np.ndarray | None = None
        self._inc_map = None
        self._desc_map = None
        self._world_T_camera_0: np.ndarray | None = None
        self._effective_desc_cfg = None
        self._persisted = False
        self._tf_buffer: Buffer | None = None
        self._tf_listener: TransformListener | None = None

        self.create_subscription(Image, image_topic, self._on_image, _sensor_data_qos())
        self.create_subscription(Odometry, odom_main_topic, self._on_odom_main, _odom_qos())
        if self._provided_pose_topic:
            self.create_subscription(
                PoseStamped,
                self._provided_pose_topic,
                self._on_provided_pose,
                _odom_qos(),
            )
        if self._velocity_topic:
            self.create_subscription(
                TwistStamped,
                self._velocity_topic,
                self._on_body_velocity,
                _odom_qos(),
            )
        self._last_odom_gt: Odometry | None = None
        if subscribe_gt:
            self.create_subscription(Odometry, odom_gt_topic, self._on_odom_gt, _odom_qos())

        # Only spin a TF listener when periodic lookups are enabled (avoids /tf subscription and
        # TF_OLD_DATA noise during rosbag playback when extrinsics are not needed).
        if tf_period > 0.0:
            self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
            self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
            self.create_timer(tf_period, self._on_tf_timer)

        self.get_logger().info(
            f"Run directory: {self._run_dir} | keyframe_distance_m={self._d} | "
            f"pair_lookback={self._pair_lookback} | "
            f"image={image_topic!r} odom={odom_main_topic!r}"
            + (f" | odom_gt={odom_gt_topic!r}" if subscribe_gt else "")
            + (
                f" | fusion={self._fusion_method_str!r}"
                + (
                    f" provided_pose={self._provided_pose_topic!r}"
                    if self._provided_pose_topic
                    else ""
                )
                + (f" velocity={self._velocity_topic!r}" if self._velocity_topic else "")
            )
            + (
                f" | TF debug {self._base_frame!r}<-{self._camera_frame!r} every {tf_period}s"
                if tf_period > 0.0
                else " | TF listener disabled (tf_lookup_period_s=0)"
            )
        )

    def _ensure_K(self, msg: Image) -> None:
        if self._K is not None:
            return
        w, h = int(msg.width), int(msg.height)
        cx = w * 0.5 if self._cx_auto < 0 else self._cx_auto
        cy = h * 0.5 if self._cy_auto < 0 else self._cy_auto
        self._K = np.array(
            [[self._fx, 0.0, cx], [0.0, self._fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._init_map_if_possible()

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
            f"merge_beta={desc_cfg.merge_beta} "
            f"max_match_distance={desc_cfg.max_match_distance} "
            f"ratio_second_best={desc_cfg.ratio_second_best}"
        )

    def _fused_cam_to_world_and_pos(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pose_fusion is not None:
            T = self._pose_fusion.fused_world_T_camera()
            return T, T[:3, 3].copy()
        assert self._last_odom is not None
        T = odom_to_cam_to_world_T(self._last_odom)
        return T, odom_position_xyz(self._last_odom)

    def _current_metric_position(self) -> np.ndarray:
        if self._pose_fusion is not None:
            return self._pose_fusion.fused_position_xyz()
        assert self._last_odom is not None
        return odom_position_xyz(self._last_odom)

    def _on_image(self, msg: Image) -> None:
        self._last_image_msg = msg
        self._ensure_K(msg)
        if self._last_odom is None:
            return
        gray = ros_image_to_gray(msg)
        if gray is None:
            self.get_logger().warn(f"Unsupported image encoding {msg.encoding!r}; skip frame.")
            return
        T, pos = self._fused_cam_to_world_and_pos()
        qx, qy, qz, qw = world_T_camera_to_quaternion_xyzw(T)
        bf = BufferedFrame(
            stamp_sec=msg.header.stamp.sec,
            stamp_nsec=msg.header.stamp.nanosec,
            gray=gray.copy(),
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

    def _on_odom_main(self, msg: Odometry) -> None:
        self._last_odom = msg
        if self._pose_fusion is not None:
            T = odom_to_cam_to_world_T(msg)
            self._pose_fusion.push_odom_world_T_camera(
                T, (msg.header.stamp.sec, msg.header.stamp.nanosec)
            )
        self._try_keyframe_selection()

    def _on_provided_pose(self, msg: PoseStamped) -> None:
        if self._pose_fusion is None:
            return
        T = pose_stamped_to_world_T_camera(msg)
        self._pose_fusion.push_provided_world_T_camera(
            T, (msg.header.stamp.sec, msg.header.stamp.nanosec)
        )

    def _on_body_velocity(self, msg: TwistStamped) -> None:
        if self._pose_fusion is None:
            return
        v_b = np.array(
            [
                float(msg.twist.linear.x),
                float(msg.twist.linear.y),
                float(msg.twist.linear.z),
            ],
            dtype=np.float64,
        )
        self._pose_fusion.push_body_velocity(
            v_b, (msg.header.stamp.sec, msg.header.stamp.nanosec)
        )

    def _on_odom_gt(self, msg: Odometry) -> None:
        self._last_odom_gt = msg

    def _try_keyframe_selection(self) -> None:
        if self._last_odom is None or self._K is None or not self._buffer:
            return
        if self._inc_map is None:
            self._init_map_if_possible()
        if self._inc_map is None:
            return

        # First keyframe: take the earliest buffered frame once odom + intrinsics exist.
        if self._last_kf_pos is None:
            bf0 = self._buffer.pop(0)
            self._commit_keyframe(bf0, distance_trigger_m=None)
            return

        pos_cur = self._current_metric_position()
        dist = float(np.linalg.norm(pos_cur - self._last_kf_pos))
        if dist < self._d:
            return
        v = pos_cur - self._last_kf_pos
        u = v / (np.linalg.norm(v) + 1e-12)
        target = self._last_kf_pos + self._d * u
        best_i = min(range(len(self._buffer)), key=lambda i: float(np.linalg.norm(self._buffer[i].pos_odom - target)))
        chosen = self._buffer[best_i]
        self._buffer = self._buffer[best_i + 1 :]
        self._commit_keyframe(chosen, distance_trigger_m=dist)

    def _commit_keyframe(self, bf: BufferedFrame, *, distance_trigger_m: float | None) -> None:
        idx = len(self._world_T_camera)
        img_rel = f"images/kf_{idx:05d}.png"
        img_path = self._run_dir / img_rel
        cv2.imwrite(str(img_path), bf.gray)

        self._world_T_camera.append(bf.cam_to_world.copy())
        self._gray_kf.append(bf.gray)
        if idx == 0:
            self._world_T_camera_0 = bf.cam_to_world.copy()

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
            "image_path": img_rel.replace("\\", "/"),
            "distance_from_previous_keyframe_m": distance_trigger_m,
        }
        self._kf_records.append(rec)
        self._last_kf_pos = bf.pos_odom.copy()

        msg = (
            f"Keyframe {idx}: saved {img_rel} | pos=({bf.pos_odom[0]:.3f},{bf.pos_odom[1]:.3f},{bf.pos_odom[2]:.3f})"
        )
        if distance_trigger_m is not None:
            msg += f" | odom spacing ~{distance_trigger_m:.3f} m (threshold {self._d})"
        self.get_logger().info(msg)

        if idx >= 1 and self._inc_map is not None:
            tw_consecutive = None
            for off in range(1, min(self._pair_lookback, idx) + 1):
                i = idx - off
                try:
                    tw = self._inc_map.add_frame_pair(
                        i, idx, self._gray_kf[i], self._gray_kf[idx]
                    )
                    if (
                        tw.scale_ok
                        and self._desc_map is not None
                        and self._world_T_camera_0 is not None
                    ):
                        try:
                            self._desc_map.integrate(tw, self._world_T_camera_0)
                        except Exception as ex:
                            self.get_logger().warn(
                                f"DescriptorLandmarkMap.integrate failed for ({i}->{idx}): {ex}"
                            )
                    n_desc = len(self._desc_map.landmarks) if self._desc_map is not None else 0
                    self.get_logger().info(
                        f"Two-view {i}->{idx}: triangulated cols={tw.X_world_h.shape[1]} "
                        f"descriptor_landmarks_total={n_desc} reproj={tw.reproj!r}"
                    )
                    if off == 1:
                        tw_consecutive = tw
                except Exception as e:
                    self.get_logger().error(f"add_frame_pair failed for ({i}->{idx}): {e}")
            if tw_consecutive is not None and self._pose_fusion is not None and hasattr(
                self._pose_fusion, "ingest_vo_keyframe_increment"
            ):
                Wi = self._world_T_camera[idx - 1]
                Wj = self._world_T_camera[idx]
                stamp = (bf.stamp_sec, bf.stamp_nsec)
                try:
                    self._pose_fusion.ingest_vo_keyframe_increment(
                        Wi, Wj, tw_consecutive.R_est, tw_consecutive.t_est, stamp
                    )
                except Exception as ex:
                    self.get_logger().warn(f"ingest_vo_keyframe_increment: {ex}")

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

    def persist_run(self) -> None:
        if self._persisted:
            return
        self._persisted = True
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
            fusion_method=self._fusion_method_str,
            provided_pose_topic=self._provided_pose_topic or None,
            velocity_topic=self._velocity_topic or None,
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
            landmarks_reference_frame="camera_0" if self._desc_map is not None else None,
        )
        pts = np.zeros((0, 3), dtype=np.float64)
        if self._desc_map is not None:
            pts = self._desc_map.positions_cam0()
        save_sparse_map_npz(self._run_dir / "sparse_map.npz", pts)
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
            f"{pts.shape[0]} descriptor landmarks in cam0 frame)"
        )

    def destroy_node(self) -> None:
        try:
            self.persist_run()
        finally:
            super().destroy_node()

    @property
    def last_odom_gt(self) -> Odometry | None:
        return self._last_odom_gt


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
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

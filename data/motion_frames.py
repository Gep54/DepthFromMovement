"""Reframe ``motion.json`` poses into the pipeline world / camera frames."""

from __future__ import annotations

import json
from pathlib import Path

from data.schema import MotionSpec
from data.tf_static_io import (
    RACE_FCU_FRAME,
    RACE_PIPELINE_CAMERA_FRAME,
    RACE_PIPELINE_WORLD_FRAME,
    load_tf_static_echo,
    reframe_world_T_sensor_poses,
    resolve_tf_static_path,
    with_race_odom_world_aliases,
)


def _load_frame_convention(dataset_root: Path) -> dict[str, str]:
    path = dataset_root / "frame_convention.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def apply_motion_frame_conversion(
    world_T_camera: list[np.ndarray],
    motion: MotionSpec,
    dataset_root: Path,
) -> list[np.ndarray]:
    """
    If ``motion`` declares a source world frame different from the pipeline target,
    apply static TF from ``tf_static`` (dataset or repo ``echo_tf_static``).

    Convention keys may also live in ``frame_convention.json`` beside ``motion.json``.
    """
    conv = _load_frame_convention(dataset_root)
    src_world = motion.world_frame or conv.get("motion_world_frame")
    dst_world = (
        motion.target_world_frame
        or conv.get("pipeline_world_frame")
        or RACE_PIPELINE_WORLD_FRAME
    )
    if not src_world or src_world == dst_world:
        return world_T_camera

    src_sensor = motion.pose_frame or conv.get("pose_frame") or RACE_FCU_FRAME
    dst_sensor = motion.camera_frame or conv.get("camera_frame") or RACE_PIPELINE_CAMERA_FRAME

    tf_path = motion.tf_static_file
    if tf_path is not None:
        path = Path(tf_path)
        if not path.is_file():
            path = dataset_root / tf_path
    else:
        path = resolve_tf_static_path(dataset_root)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"motion.json world_frame={src_world!r} requires tf_static "
            f"(dataset tf_static.echo or repo echo_tf_static)"
        )

    edges = with_race_odom_world_aliases(load_tf_static_echo(path))
    return reframe_world_T_sensor_poses(
        world_T_camera,
        edges,
        src_world=src_world,
        dst_world=dst_world,
        src_sensor=src_sensor,
        dst_sensor=dst_sensor,
    )

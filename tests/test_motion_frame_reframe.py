"""Static TF reframe for motion.json (fixed_origin → gps_baro + fcu → rgb)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.io_json import load_motion_json
from data.motion_frames import apply_motion_frame_conversion
from data.schema import MotionSpec
from data.tf_static_io import (
    RACE_FCU_FRAME,
    RACE_FIXED_ORIGIN_FRAME,
    RACE_PIPELINE_CAMERA_FRAME,
    RACE_PIPELINE_WORLD_FRAME,
    default_race_tf_static_path,
    load_tf_static_echo,
    lookup_transform,
    reframe_world_T_sensor_poses,
    with_race_odom_world_aliases,
)


@pytest.fixture()
def race_edges() -> dict:
    path = default_race_tf_static_path()
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return with_race_odom_world_aliases(load_tf_static_echo(path))


def test_load_tf_static_echo_parses_fcu_rgb(race_edges) -> None:
    assert (RACE_FCU_FRAME, RACE_PIPELINE_CAMERA_FRAME) in race_edges
    T = race_edges[(RACE_FCU_FRAME, RACE_PIPELINE_CAMERA_FRAME)]
    assert np.isclose(T[0, 3], 0.14, atol=1e-6)


def test_lookup_fixed_to_gps_via_local(race_edges) -> None:
    T = lookup_transform(race_edges, RACE_PIPELINE_WORLD_FRAME, RACE_FIXED_ORIGIN_FRAME)
    assert T.shape == (4, 4)
    # fixed→local is ~identity; gps≈local ⇒ gps_T_fixed ≈ I
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-3)
    assert np.linalg.norm(T[:3, 3]) < 0.01


def test_reframe_identity_sensor_unchanged(race_edges) -> None:
    W = [np.eye(4)]
    out = reframe_world_T_sensor_poses(
        W,
        race_edges,
        src_world=RACE_FIXED_ORIGIN_FRAME,
        dst_world=RACE_PIPELINE_WORLD_FRAME,
        src_sensor=RACE_PIPELINE_CAMERA_FRAME,
        dst_sensor=RACE_PIPELINE_CAMERA_FRAME,
    )
    assert np.allclose(out[0][:3, :3], W[0][:3, :3], atol=1e-3)
    assert np.linalg.norm(out[0][:3, 3] - W[0][:3, 3]) < 0.01


def test_apply_motion_frame_conversion_skips_when_no_world_frame(tmp_path: Path) -> None:
    motion = MotionSpec(
        pose_convention="world_T_camera",
        representation="absolute",
        transforms=[np.eye(4)],
    )
    out = apply_motion_frame_conversion([np.eye(4)], motion, tmp_path)
    assert np.allclose(out[0], np.eye(4))


def test_apply_motion_frame_conversion_from_frame_convention(tmp_path: Path) -> None:
    path = default_race_tf_static_path()
    if not path.is_file():
        pytest.skip("echo_tf_static missing")
    (tmp_path / "frame_convention.json").write_text(
        json.dumps(
            {
                "motion_world_frame": RACE_FIXED_ORIGIN_FRAME,
                "pipeline_world_frame": RACE_PIPELINE_WORLD_FRAME,
                "pose_frame": RACE_FCU_FRAME,
                "camera_frame": RACE_PIPELINE_CAMERA_FRAME,
            }
        ),
        encoding="utf-8",
    )
    motion = MotionSpec(
        pose_convention="world_T_camera",
        representation="absolute",
        transforms=[np.eye(4)],
    )
    edges = with_race_odom_world_aliases(load_tf_static_echo(path))
    out = apply_motion_frame_conversion([np.eye(4)], motion, tmp_path)
    expected = reframe_world_T_sensor_poses(
        [np.eye(4)],
        edges,
        src_world=RACE_FIXED_ORIGIN_FRAME,
        dst_world=RACE_PIPELINE_WORLD_FRAME,
        src_sensor=RACE_FCU_FRAME,
        dst_sensor=RACE_PIPELINE_CAMERA_FRAME,
    )[0]
    assert np.allclose(out[0], expected, atol=1e-6)

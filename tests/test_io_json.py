"""Tests for data.io_json motion/calibration round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.dataset import load_dataset
from data.io_json import load_motion_json, save_motion_json, world_T_camera_from_motion


def test_save_motion_json_roundtrip(tmp_path: Path) -> None:
    transforms = []
    filenames = []
    for i in range(3):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 0.1 * i
        transforms.append(T)
        filenames.append(f"frame_{i:05d}.png")

    path = tmp_path / "motion.json"
    save_motion_json(path, transforms, filenames=filenames)

    spec = load_motion_json(path)
    assert spec.pose_convention == "world_T_camera"
    assert spec.representation == "absolute"
    assert len(spec.transforms) == 3
    W = world_T_camera_from_motion(spec)
    for i, T in enumerate(transforms):
        assert np.allclose(W[i], T)

    raw = path.read_text(encoding="utf-8")
    assert '"filename": "frame_00000.png"' in raw


def test_save_motion_json_load_dataset(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    import cv2

    for i in range(2):
        im = np.zeros((48, 64, 3), np.uint8)
        cv2.imwrite(str(img_dir / f"frame_{i:05d}.png"), im)

    K = [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]
    import json

    with (tmp_path / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump({"K": K, "image_size": [64, 48]}, f)

    Ts = [np.eye(4) for _ in range(2)]
    Ts[1][0, 3] = 0.2
    save_motion_json(
        tmp_path / "motion.json",
        Ts,
        filenames=[f"frame_{i:05d}.png" for i in range(2)],
    )

    ds = load_dataset(tmp_path)
    assert len(ds.image_paths) == 2
    assert len(ds.world_T_camera) == 2
    assert np.allclose(ds.world_T_camera[0], np.eye(4))

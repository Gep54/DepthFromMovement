from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def mini_dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "seq"
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    for k in range(3):
        im = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(im, (40 + k * 15, 60), 10, (200, 100, 50), -1)
        cv2.imwrite(str(img_dir / f"frame_{k:03d}.png"), im)
    K = [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]]
    with (root / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump({"K": K, "image_size": [160, 120]}, f)
    frames = []
    for k in range(3):
        t = 0.15 * k
        T = np.eye(4)
        T[0, 3] = t
        frames.append({"T": T.tolist()})
    with (root / "motion.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pose_convention": "world_T_camera",
                "representation": "absolute",
                "frames": frames,
            },
            f,
        )
    return root

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.dataset import load_dataset


def test_load_mini_dataset(mini_dataset_dir: Path) -> None:
    ds = load_dataset(mini_dataset_dir)
    assert len(ds.image_paths) == 3
    assert len(ds.world_T_camera) == 3
    assert ds.calibration.K.shape == (3, 3)
    assert ds.feature_config.method == "ORB"
    assert ds.feature_config.n_features == 2000


def test_features_json_overrides_defaults(mini_dataset_dir: Path) -> None:
    with (mini_dataset_dir / "features.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "SIFT",
                "n_features": 800,
                "ratio_test": 0.7,
                "cross_check": True,
            },
            f,
        )
    ds = load_dataset(mini_dataset_dir)
    assert ds.feature_config.method == "SIFT"
    assert ds.feature_config.n_features == 800
    assert ds.feature_config.ratio_test == 0.7
    assert ds.feature_config.cross_check is True


def test_dataset_provided_motion_fusion_position_blend(mini_dataset_dir: Path) -> None:
    """Optional ``provided_motion.json`` + ``fusion.json`` fuse into ``world_T_camera``."""
    with (mini_dataset_dir / "motion.json").open(encoding="utf-8") as f:
        motion_raw = json.load(f)
    prov = json.loads(json.dumps(motion_raw))
    for fr in prov["frames"]:
        T = np.asarray(fr["T"], dtype=np.float64)
        T[0, 3] = float(T[0, 3]) + 10.0
        fr["T"] = T.tolist()
    with (mini_dataset_dir / "provided_motion.json").open("w", encoding="utf-8") as f:
        json.dump(prov, f)
    with (mini_dataset_dir / "fusion.json").open("w", encoding="utf-8") as f:
        json.dump({"method": "position_blend", "position_blend_weight": 0.5}, f)

    ds = load_dataset(mini_dataset_dir)
    np.testing.assert_allclose(ds.world_T_camera[0][:3, 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ds.world_T_camera[1][:3, 3], [0.15, 0.0, 0.0], atol=1e-9)

from __future__ import annotations

import json
from pathlib import Path

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

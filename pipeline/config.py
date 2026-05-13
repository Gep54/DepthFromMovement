from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class FeatureConfig:
    method: Literal["ORB", "SIFT"] = "ORB"
    n_features: int = 2000
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    sift_contrast_thresh: float = 0.04
    ratio_test: float = 0.75
    cross_check: bool = False

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def clamp_motion_confidence(value: float) -> float:
    """Clamp user input to [0, 1]."""
    return float(max(0.0, min(1.0, value)))


@dataclass
class FeatureConfig:
    method: Literal["ORB", "SIFT"] = "ORB"
    n_features: int = 2000
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    sift_contrast_thresh: float = 0.04
    ratio_test: float = 0.75
    cross_check: bool = False

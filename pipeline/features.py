from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pipeline.config import FeatureConfig


@dataclass
class FrameFeatures:
    """ORB/SIFT keypoints and descriptors for one grayscale frame (shared across pairs)."""

    keypoints: list[Any]
    descriptors: np.ndarray | None


def compute_frame_features_cache(images_gray: Sequence[np.ndarray], cfg: FeatureConfig) -> list[FrameFeatures]:
    """Run ``detect_and_compute`` once per frame; reuse entries when matching multiple pairs."""
    return [FrameFeatures(*detect_and_compute(g, cfg)) for g in images_gray]


def _make_detector(cfg: FeatureConfig) -> tuple[Any, Any]:
    if cfg.method == "ORB":
        det = cv2.ORB_create(
            nfeatures=cfg.n_features,
            scaleFactor=cfg.orb_scale_factor,
            nlevels=cfg.orb_n_levels,
        )
        norm = cv2.NORM_HAMMING
        return det, norm
    det = cv2.SIFT_create(
        nfeatures=cfg.n_features,
        contrastThreshold=cfg.sift_contrast_thresh,
    )
    return det, cv2.NORM_L2


def detect_and_compute(
    image_gray: np.ndarray,
    cfg: FeatureConfig,
    mask: np.ndarray | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    det, _ = _make_detector(cfg)
    kps, desc = det.detectAndCompute(image_gray, mask)
    if desc is None:
        return [], None
    return kps, desc

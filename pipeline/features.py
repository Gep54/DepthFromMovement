from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.config import FeatureConfig


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

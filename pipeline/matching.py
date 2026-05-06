from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.config import FeatureConfig


def match_pair_points(
    kps1: list[Any],
    kps2: list[Any],
    desc1: np.ndarray,
    desc2: np.ndarray,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    """Return pts1, pts2 (N,2) float32 in pixel coordinates and kept matches."""
    if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), []
    norm = cv2.NORM_HAMMING if cfg.method == "ORB" else cv2.NORM_L2
    if cfg.cross_check:
        bf = cv2.BFMatcher(norm, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda m: m.distance)
    else:
        bf = cv2.BFMatcher(norm, crossCheck=False)
        knn = bf.knnMatch(desc1, desc2, k=2)
        matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < cfg.ratio_test * n.distance:
                matches.append(m)
    if not matches:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), []
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2, matches

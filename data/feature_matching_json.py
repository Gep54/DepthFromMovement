"""Optional dataset-local feature / matcher settings (``features.json``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from pipeline.config import FeatureConfig


def load_feature_matching_json(path: Path) -> FeatureConfig:
    """
    Load :class:`~pipeline.config.FeatureConfig` from JSON.

    Expected keys (all optional; omitted keys use :class:`FeatureConfig` defaults):

    - ``method``: ``\"ORB\"`` | ``\"SIFT\"``
    - ``n_features``: int
    - ``orb_scale_factor``: float
    - ``orb_n_levels``: int
    - ``sift_contrast_thresh``: float
    - ``ratio_test``: float (Lowe ratio test for KNN matching)
    - ``cross_check``: bool (BFMatcher cross-check instead of ratio test)
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    defaults = FeatureConfig()
    method_key = raw.get("method", defaults.method)
    if method_key not in ("ORB", "SIFT"):
        raise ValueError(f"features.json: method must be 'ORB' or 'SIFT', got {method_key!r}")
    method = cast(Literal["ORB", "SIFT"], method_key)

    return FeatureConfig(
        method=method,
        n_features=int(raw.get("n_features", defaults.n_features)),
        orb_scale_factor=float(raw.get("orb_scale_factor", defaults.orb_scale_factor)),
        orb_n_levels=int(raw.get("orb_n_levels", defaults.orb_n_levels)),
        sift_contrast_thresh=float(raw.get("sift_contrast_thresh", defaults.sift_contrast_thresh)),
        ratio_test=float(raw.get("ratio_test", defaults.ratio_test)),
        cross_check=bool(raw.get("cross_check", defaults.cross_check)),
    )

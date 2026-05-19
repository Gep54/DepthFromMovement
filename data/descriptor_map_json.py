"""Optional ``descriptor_map.json`` next to ``calibration.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from pipeline.descriptor_landmark_map import DescriptorMapConfig


def load_descriptor_map_json(path: Path, method: Literal["ORB", "SIFT"]) -> DescriptorMapConfig:
    """
    Load :class:`~pipeline.descriptor_landmark_map.DescriptorMapConfig`.

    Keys (all optional): ``fusion_enabled``, ``merge_beta`` (number or null), ``max_match_distance``,
    ``ratio_second_best`` (number or null), ``spatial_merge_baseline_factor`` (default 0.25),
    ``spatial_merge_radius_m`` (fixed metres; overrides baseline factor when set).
    ``merge_beta: null`` or omitted means mean-equivalent EMA (``1/(n+1)`` per update).
    """
    if not path.is_file():
        return DescriptorMapConfig.defaults(method)

    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    base = DescriptorMapConfig.defaults(method)
    fusion_enabled = bool(raw.get("fusion_enabled", base.fusion_enabled))
    merge_beta: float | None
    if "merge_beta" not in raw:
        merge_beta = base.merge_beta
    elif raw["merge_beta"] is None:
        merge_beta = None
    else:
        merge_beta = float(raw["merge_beta"])

    ratio_raw = raw.get("ratio_second_best", base.ratio_second_best)
    ratio_second_best: float | None
    if ratio_raw is None:
        ratio_second_best = None
    else:
        ratio_second_best = float(ratio_raw)

    md_key = raw.get("max_match_distance", base.max_match_distance)

    spatial_raw = raw.get("spatial_merge_radius_m", base.spatial_merge_radius_m)
    spatial_merge_radius_m: float | None
    if spatial_raw is None:
        spatial_merge_radius_m = None
    else:
        spatial_merge_radius_m = float(spatial_raw)

    factor_raw = raw.get("spatial_merge_baseline_factor", base.spatial_merge_baseline_factor)
    spatial_merge_baseline_factor = float(factor_raw)

    return DescriptorMapConfig(
        method=cast(Literal["ORB", "SIFT"], method),
        fusion_enabled=fusion_enabled,
        merge_beta=merge_beta,
        max_match_distance=float(md_key),
        ratio_second_best=ratio_second_best,
        spatial_merge_baseline_factor=spatial_merge_baseline_factor,
        spatial_merge_radius_m=spatial_merge_radius_m,
    )

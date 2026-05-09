from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

"""Ordered slugs align with exported pipeline visuals (raw → metrics)."""

STEP_ORDER: tuple[str, ...] = (
    "raw_input",
    "keypoints",
    "matches",
    "epilines",
    "inlier_outlier",
    "triangulation",
    "estimated_depth",
    "depth_error",
)

_STEP_INDEX = {name: i + 1 for i, name in enumerate(STEP_ORDER)}


class PipelineRecorder:
    """
    Writes one PNG per pipeline stage under ``<run_root>/steps/``.
    Filenames are ``{idx:02d}_{slug}.png`` for stable ordering.
    """

    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root)
        self.steps_dir = self.run_root / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, slug: str) -> Path:
        if slug not in _STEP_INDEX:
            raise KeyError(f"unknown step {slug!r}; expected one of {sorted(_STEP_INDEX)}")
        idx = _STEP_INDEX[slug]
        return self.steps_dir / f"{idx:02d}_{slug}.png"

    def write(self, slug: str, image_bgr: np.ndarray) -> Path:
        path = self.path_for(slug)
        if image_bgr.dtype != np.uint8:
            raise TypeError("image_bgr must be uint8")
        ok = cv2.imwrite(str(path), image_bgr)
        if not ok:
            raise RuntimeError(f"failed to write {path}")
        return path

    def record(self, slug: str, payload: Mapping[str, Any]) -> Path:
        """
        Accept a payload dict; requires key ``image_bgr`` (uint8 BGR) unless
        ``slug`` is handled specially later. Extra keys are ignored.
        """
        if "image_bgr" not in payload:
            raise KeyError(f"record({slug!r}) expects 'image_bgr' in payload")
        return self.write(slug, payload["image_bgr"])

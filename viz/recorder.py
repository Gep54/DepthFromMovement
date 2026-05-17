from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

"""Ordered slugs for exported pipeline visuals."""

SINGLE_STEP_ORDER: tuple[str, ...] = (
    "original",
    "grayscale",
    "edges",
    "descriptors",
)

PAIR_STEP_ORDER: tuple[str, ...] = (
    "raw_input",
    "matches",
    "epipolar_outliers_epilines",
    "match_classifications",
    "inliers",
)

GEOMETRY_STEP_ORDER: tuple[str, ...] = (
    "triangulation",
    "estimated_depth",
    "depth_error",
)

# Legacy flat order (deprecated); kept for imports that reference STEP_ORDER.
STEP_ORDER: tuple[str, ...] = SINGLE_STEP_ORDER + PAIR_STEP_ORDER + GEOMETRY_STEP_ORDER

_SUBDIR_ORDERS: dict[str, tuple[str, ...]] = {
    "single": SINGLE_STEP_ORDER,
    "pair": PAIR_STEP_ORDER,
    "geometry": GEOMETRY_STEP_ORDER,
}


def _index_for(subdir: str, slug: str) -> int:
    order = _SUBDIR_ORDERS[subdir]
    if slug not in order:
        raise KeyError(f"unknown step {slug!r} for subdir {subdir!r}; expected one of {order}")
    return order.index(slug) + 1


class PipelineRecorder:
    """
    Writes one PNG per pipeline stage under ``<run_root>/steps/<subdir>/``.
    Filenames are ``{idx:02d}_{slug}.png`` for stable ordering.
    """

    def __init__(self, run_root: str | Path, *, subdir: str = "") -> None:
        self.run_root = Path(run_root)
        self.subdir = subdir.strip("/\\")
        if self.subdir and self.subdir not in _SUBDIR_ORDERS:
            raise ValueError(f"unknown subdir {self.subdir!r}; expected one of {sorted(_SUBDIR_ORDERS)}")
        self.steps_dir = self.run_root / "steps"
        if self.subdir:
            self.steps_dir = self.steps_dir / self.subdir
        self.steps_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, slug: str) -> Path:
        if not self.subdir:
            raise ValueError("path_for(slug) requires a subdir-specific PipelineRecorder")
        idx = _index_for(self.subdir, slug)
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


def ensure_step_pngs_exist(
    run_dir: str | Path,
    *,
    include_geometry: bool = True,
) -> list[Path]:
    """Verify illustration (+ optional geometry) stage files are present."""
    root = Path(run_dir)
    paths: list[Path] = []
    for sub, order in _SUBDIR_ORDERS.items():
        if sub == "geometry" and not include_geometry:
            continue
        rec = PipelineRecorder(root, subdir=sub)
        for slug in order:
            p = rec.path_for(slug)
            if not p.is_file():
                raise FileNotFoundError(f"missing step PNG: {p}")
            paths.append(p)
    return paths


def ensure_all_step_pngs_exist(run_dir: str | Path, *, include_geometry: bool = True) -> list[Path]:
    """Alias for :func:`ensure_step_pngs_exist` (backward-compatible name)."""
    return ensure_step_pngs_exist(run_dir, include_geometry=include_geometry)

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
    "match_classifications",
    "inliers",
)

GEOMETRY_STEP_ORDER: tuple[str, ...] = (
    "triangulation",
    "estimated_depth",
    "depth_error",
)

# Default ``dfm-export-steps`` output (minimal per pair).
MINIMAL_PAIR_STEP_ORDER: tuple[str, ...] = (
    "matches",
    "match_classifications",
    "inliers",
)

DETAIL_PAIR_STEP_ORDER: tuple[str, ...] = (
    "rejected_epipolar",
    "rejected_cheiral",
)

MINIMAL_GEOMETRY_STEP_ORDER: tuple[str, ...] = ("estimated_depth",)

# Legacy flat order (deprecated); kept for imports that reference STEP_ORDER.
STEP_ORDER: tuple[str, ...] = SINGLE_STEP_ORDER + PAIR_STEP_ORDER + GEOMETRY_STEP_ORDER

_SUBDIR_ORDERS: dict[str, tuple[str, ...]] = {
    "single": SINGLE_STEP_ORDER,
    "pair": PAIR_STEP_ORDER,
    "geometry": GEOMETRY_STEP_ORDER,
}


def _index_for(order: tuple[str, ...], slug: str) -> int:
    if slug not in order:
        raise KeyError(f"unknown step {slug!r}; expected one of {order}")
    return order.index(slug) + 1


class PipelineRecorder:
    """
    Writes one PNG per pipeline stage under ``<run_root>/steps/<subdir>/``.
    Filenames are ``{idx:02d}_{slug}.png`` for stable ordering.
    """

    def __init__(
        self,
        run_root: str | Path,
        *,
        subdir: str = "",
        slug_order: tuple[str, ...] | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.subdir = subdir.strip("/\\")
        if self.subdir and self.subdir not in _SUBDIR_ORDERS:
            raise ValueError(f"unknown subdir {self.subdir!r}; expected one of {sorted(_SUBDIR_ORDERS)}")
        if slug_order is not None:
            self._slug_order = slug_order
        elif self.subdir:
            self._slug_order = _SUBDIR_ORDERS[self.subdir]
        else:
            self._slug_order = ()
        self.steps_dir = self.run_root / "steps"
        if self.subdir:
            self.steps_dir = self.steps_dir / self.subdir
        self.steps_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, slug: str) -> Path:
        if not self.subdir:
            raise ValueError("path_for(slug) requires a subdir-specific PipelineRecorder")
        idx = _index_for(self._slug_order, slug)
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


def pair_slug_orders(*, full_steps: bool = False, detail_log: bool = False) -> tuple[str, ...]:
    """Ordered pair slugs for the active export profile."""
    if full_steps:
        order: tuple[str, ...] = PAIR_STEP_ORDER
    else:
        order = MINIMAL_PAIR_STEP_ORDER
    if detail_log:
        order = order + DETAIL_PAIR_STEP_ORDER
    return order


def geometry_slug_order(*, full_steps: bool = False) -> tuple[str, ...]:
    return GEOMETRY_STEP_ORDER if full_steps else MINIMAL_GEOMETRY_STEP_ORDER


def ensure_step_pngs_exist(
    run_dir: str | Path,
    *,
    include_geometry: bool = True,
    full_steps: bool = False,
    detail_log: bool = False,
) -> list[Path]:
    """Verify exported stage PNGs for the active profile."""
    root = Path(run_dir)
    paths: list[Path] = []
    if full_steps:
        rec_single = PipelineRecorder(root, subdir="single")
        for slug in SINGLE_STEP_ORDER:
            p = rec_single.path_for(slug)
            if not p.is_file():
                raise FileNotFoundError(f"missing step PNG: {p}")
            paths.append(p)

    pair_order = pair_slug_orders(full_steps=full_steps, detail_log=detail_log)
    rec_pair = PipelineRecorder(root, subdir="pair", slug_order=pair_order)
    for slug in pair_order:
        p = rec_pair.path_for(slug)
        if not p.is_file():
            raise FileNotFoundError(f"missing step PNG: {p}")
        paths.append(p)

    if include_geometry:
        geom_order = geometry_slug_order(full_steps=full_steps)
        rec_geom = PipelineRecorder(root, subdir="geometry", slug_order=geom_order)
        for slug in geom_order:
            p = rec_geom.path_for(slug)
            if not p.is_file():
                raise FileNotFoundError(f"missing step PNG: {p}")
            paths.append(p)
    return paths


def ensure_all_step_pngs_exist(run_dir: str | Path, *, include_geometry: bool = True) -> list[Path]:
    """Alias for :func:`ensure_step_pngs_exist` (backward-compatible name)."""
    return ensure_step_pngs_exist(run_dir, include_geometry=include_geometry)

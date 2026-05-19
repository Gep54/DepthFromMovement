from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from viz.overlays import (
    EPIPOLAR_HIGHLIGHT_COLORS,
    draw_matches_with_bilateral_epilines,
    symmetric_epipolar_distances,
)
from viz.pdf_util import write_bgr_pdf


@dataclass(frozen=True)
class EpipolarPairView:
    """Undistorted images and matches for one frame pair."""

    frame_i: int
    frame_j: int
    und_i: np.ndarray
    und_j: np.ndarray
    pts1: np.ndarray
    pts2: np.ndarray
    F: np.ndarray


def _label_banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (24, 24, 24), -1)
    cv2.putText(
        out,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return out


def _resize_panel(panel: np.ndarray, target_width: int) -> np.ndarray:
    h, w = panel.shape[:2]
    if w == target_width:
        return panel
    scale = target_width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(panel, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _stack_panels_vertical(panels: list[np.ndarray], *, target_width: int = 1280) -> np.ndarray:
    if not panels:
        raise ValueError("need at least one panel")
    resized = [_resize_panel(p, target_width) for p in panels]
    return np.vstack(resized)


def _page_all_matches(view: EpipolarPairView) -> np.ndarray:
    page = draw_matches_with_bilateral_epilines(
        view.und_i,
        view.und_j,
        view.pts1,
        view.pts2,
        view.F,
        default_color=(0, 255, 255),
        line_thickness=1,
    )
    return _label_banner(page, f"frames {view.frame_i:03d}-{view.frame_j:03d}  all matches")


def _page_highlighted_matches(
    entries: list[tuple[float, int, int]],
    views: list[EpipolarPairView],
    *,
    title: str,
) -> np.ndarray:
    """One PDF page: up to five correspondences, colour-coded."""
    panels: list[np.ndarray] = []
    for rank, (dist, vi, mk) in enumerate(entries):
        view = views[vi]
        color = EPIPOLAR_HIGHLIGHT_COLORS[rank % len(EPIPOLAR_HIGHLIGHT_COLORS)]
        panel = draw_matches_with_bilateral_epilines(
            view.und_i,
            view.und_j,
            view.pts1,
            view.pts2,
            view.F,
            match_indices=np.array([mk], dtype=np.int32),
            colors=[color],
            line_thickness=2,
            point_radius=4,
        )
        panels.append(
            _label_banner(
                panel,
                f"#{rank + 1}  dist={dist:.3f}px  frames {view.frame_i:03d}-{view.frame_j:03d}",
            )
        )
    body = _stack_panels_vertical(panels)
    return _label_banner(body, title)


def _ranked_correspondences(views: list[EpipolarPairView]) -> list[tuple[float, int, int]]:
    ranked: list[tuple[float, int, int]] = []
    for vi, view in enumerate(views):
        dists = symmetric_epipolar_distances(view.pts1, view.pts2, view.F)
        for mk, d in enumerate(dists):
            ranked.append((float(d), vi, int(mk)))
    ranked.sort(key=lambda t: t[0])
    return ranked


def export_epipolar_pdf_bundle(run_dir: str | Path, views: list[EpipolarPairView]) -> Path:
    """
    Write ``<run_dir>/epipolar/`` with three PDFs:

    - ``all_pairs_epilines.pdf`` — every correspondence with epilines on both images
    - ``worst_epipolar_5.pdf`` — five largest symmetric epipolar distances (colour-coded)
    - ``best_epipolar_5.pdf`` — five smallest distances (colour-coded)
    """
    if not views:
        raise ValueError("export_epipolar_pdf_bundle requires at least one frame pair view")

    out_dir = Path(run_dir) / "epipolar"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_all = [_page_all_matches(v) for v in views]
    write_bgr_pdf(out_dir / "all_pairs_epilines.pdf", pages_all)

    ranked = _ranked_correspondences(views)
    n_pick = min(5, len(ranked))
    best = ranked[:n_pick]
    worst = list(reversed(ranked[-n_pick:])) if n_pick else []

    write_bgr_pdf(
        out_dir / "worst_epipolar_5.pdf",
        [_page_highlighted_matches(worst, views, title="worst epipolar constraint (5 pairs)")],
    )
    write_bgr_pdf(
        out_dir / "best_epipolar_5.pdf",
        [_page_highlighted_matches(best, views, title="best epipolar constraint (5 pairs)")],
    )
    return out_dir

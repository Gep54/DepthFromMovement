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
from viz.recorder import EPIPOLAR_STEP_ORDER


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


def _ranked_correspondences(view: EpipolarPairView) -> list[tuple[float, int]]:
    """Sorted ``(symmetric epipolar distance px, match index)`` ascending (best first)."""
    dists = symmetric_epipolar_distances(view.pts1, view.pts2, view.F)
    ranked = [(float(d), int(mk)) for mk, d in enumerate(dists)]
    ranked.sort(key=lambda t: t[0])
    return ranked


def _page_highlighted_matches(
    view: EpipolarPairView,
    entries: list[tuple[float, int]],
    *,
    title: str,
) -> np.ndarray:
    """One PNG: up to five correspondences stacked vertically, colour-coded."""
    panels: list[np.ndarray] = []
    for rank, (dist, mk) in enumerate(entries):
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
    if not panels:
        empty = np.hstack([view.und_i, view.und_j])
        return _label_banner(empty, f"{title}  (no matches)")
    body = _stack_panels_vertical(panels)
    return _label_banner(body, title)


def export_epipolar_pair_pngs(pair_run_dir: str | Path, view: EpipolarPairView) -> Path:
    """
    Write per-pair epipolar figures under ``<pair_run_dir>/steps/epipolar/``:

    - ``01_all_matches.png`` — every correspondence with epilines
    - ``02_best_epipolar_5.png`` — five smallest symmetric epipolar distances (stacked)
    - ``03_worst_epipolar_5.png`` — five largest distances (stacked)
    """
    from viz.recorder import PipelineRecorder

    rec = PipelineRecorder(pair_run_dir, subdir="epipolar", slug_order=EPIPOLAR_STEP_ORDER)
    rec.write("all_matches", _page_all_matches(view))

    ranked = _ranked_correspondences(view)
    n_pick = min(5, len(ranked))
    best = ranked[:n_pick]
    worst = list(reversed(ranked[-n_pick:])) if n_pick else []

    rec.write(
        "best_epipolar_5",
        _page_highlighted_matches(
            view,
            best,
            title=f"best epipolar constraint (top {len(best)})  {view.frame_i:03d}-{view.frame_j:03d}",
        ),
    )
    rec.write(
        "worst_epipolar_5",
        _page_highlighted_matches(
            view,
            worst,
            title=f"worst epipolar constraint (bottom {len(worst)})  {view.frame_i:03d}-{view.frame_j:03d}",
        ),
    )
    return rec.steps_dir


def export_epipolar_pdf_bundle(run_dir: str | Path, views: list[EpipolarPairView]) -> Path:
    """Deprecated: use :func:`export_epipolar_pair_pngs` per pair. Writes PNGs under ``epipolar/``."""
    if not views:
        raise ValueError("export_epipolar_pdf_bundle requires at least one frame pair view")
    out_dir = Path(run_dir) / "epipolar"
    out_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        pair_sub = out_dir / f"{view.frame_i:03d}_{view.frame_j:03d}"
        export_epipolar_pair_pngs(pair_sub, view)
    return out_dir

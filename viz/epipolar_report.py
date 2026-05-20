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

# BGR colours for ``01_all_matches`` (epilines vs match chords vs keypoints).
EPIPOLAR_ALL_EPI_LINE_COLOR = (0, 128, 255)  # orange
EPIPOLAR_ALL_MATCH_LINE_COLOR = (255, 255, 0)  # cyan
EPIPOLAR_ALL_POINT_COLOR = (0, 255, 0)  # green


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


def _page_all_matches(view: EpipolarPairView) -> np.ndarray:
    page = draw_matches_with_bilateral_epilines(
        view.und_i,
        view.und_j,
        view.pts1,
        view.pts2,
        view.F,
        epiline_color=EPIPOLAR_ALL_EPI_LINE_COLOR,
        match_line_color=EPIPOLAR_ALL_MATCH_LINE_COLOR,
        point_color=EPIPOLAR_ALL_POINT_COLOR,
        line_thickness=1,
        point_radius=3,
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
    """One side-by-side image pair with up to five correspondences colour-coded."""
    if not entries:
        empty = np.hstack([view.und_i, view.und_j])
        return _label_banner(empty, f"{title}  (no matches)")

    indices = np.array([mk for _, mk in entries], dtype=np.int32)
    colors = [EPIPOLAR_HIGHLIGHT_COLORS[r % len(EPIPOLAR_HIGHLIGHT_COLORS)] for r in range(len(entries))]
    panel = draw_matches_with_bilateral_epilines(
        view.und_i,
        view.und_j,
        view.pts1,
        view.pts2,
        view.F,
        match_indices=indices,
        colors=colors,
        line_thickness=2,
        point_radius=4,
    )
    dist_bits = ", ".join(f"#{r + 1}={d:.2f}px" for r, (d, _) in enumerate(entries))
    return _label_banner(panel, f"{title}  |  {dist_bits}")


def export_epipolar_pair_pngs(pair_run_dir: str | Path, view: EpipolarPairView) -> Path:
    """
    Write per-pair epipolar figures under ``<pair_run_dir>/steps/epipolar/``:

    - ``01_all_matches.png`` — every correspondence; orange epilines, cyan chords, green points
    - ``02_best_epipolar_5.png`` — five best distances on one image pair
    - ``03_worst_epipolar_5.png`` — five worst distances on one image pair
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
            title=f"best epipolar (top {len(best)})  {view.frame_i:03d}-{view.frame_j:03d}",
        ),
    )
    rec.write(
        "worst_epipolar_5",
        _page_highlighted_matches(
            view,
            worst,
            title=f"worst epipolar (bottom {len(worst)})  {view.frame_i:03d}-{view.frame_j:03d}",
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

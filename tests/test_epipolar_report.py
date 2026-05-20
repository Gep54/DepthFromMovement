from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from viz.epipolar_report import (
    EPIPOLAR_STEP_ORDER,
    EpipolarPairView,
    export_epipolar_pair_pngs,
    export_epipolar_pdf_bundle,
)
from viz.overlays import symmetric_epipolar_distances
from viz.pdf_util import write_bgr_pdf


def test_symmetric_epipolar_distances_zero_on_perfect_matches() -> None:
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    tx = np.array(
        [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]],
        dtype=np.float64,
    )
    E = tx @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

    pts1 = np.array([[100.0, 120.0], [400.0, 300.0]], dtype=np.float64)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    pts2 = []
    for a, b, c in lines2:
        y = int(-c / b) if abs(b) > 1e-6 else 120.0
        pts2.append([pts1[len(pts2), 0], float(y)])
    pts2 = np.array(pts2, dtype=np.float64)
    d = symmetric_epipolar_distances(pts1, pts2, F)
    assert d.shape == (2,)
    assert np.all(d < 1e-3)


def test_write_bgr_pdf_roundtrip(tmp_path: Path) -> None:
    page = np.zeros((80, 120, 3), dtype=np.uint8)
    page[:, :, 1] = 200
    out = tmp_path / "t.pdf"
    write_bgr_pdf(out, [page, page])
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"%%EOF" in data


def test_export_epipolar_pair_pngs(tmp_path: Path) -> None:
    img = np.full((60, 80, 3), 40, dtype=np.uint8)
    F = np.eye(3, dtype=np.float64)
    F[0, 2] = -0.001
    pts1 = np.array([[20.0, 30.0], [50.0, 40.0]], dtype=np.float64)
    pts2 = np.array([[25.0, 30.0], [55.0, 42.0]], dtype=np.float64)
    view = EpipolarPairView(0, 1, img, img, pts1, pts2, F)
    steps = export_epipolar_pair_pngs(tmp_path / "pair", view)
    assert steps.name == "epipolar"
    for slug in EPIPOLAR_STEP_ORDER:
        paths = list(steps.glob(f"*_{slug}.png"))
        assert len(paths) == 1, slug
        assert paths[0].stat().st_size > 0


def test_export_epipolar_pdf_bundle_writes_per_pair_png_dirs(tmp_path: Path) -> None:
    img = np.full((60, 80, 3), 40, dtype=np.uint8)
    F = np.eye(3, dtype=np.float64)
    pts1 = np.array([[20.0, 30.0]], dtype=np.float64)
    pts2 = np.array([[25.0, 30.0]], dtype=np.float64)
    view = EpipolarPairView(0, 1, img, img, pts1, pts2, F)
    out_dir = export_epipolar_pdf_bundle(tmp_path / "run", [view])
    pair_epi = out_dir / "000_001" / "steps" / "epipolar"
    assert pair_epi.is_dir()
    assert len(list(pair_epi.glob("*.png"))) == len(EPIPOLAR_STEP_ORDER)

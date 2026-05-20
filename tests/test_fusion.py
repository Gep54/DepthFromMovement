from __future__ import annotations

import numpy as np

from pipeline.fusion import FusedLandmarkMap
from pipeline.map import TwoViewResult


def _tw(
    fi: int,
    fj: int,
    pts1: np.ndarray,
    pts2: np.ndarray,
    X_world_h: np.ndarray,
    cheiral: np.ndarray,
) -> TwoViewResult:
    n = pts1.shape[0]
    return TwoViewResult(
        frame_i=fi,
        frame_j=fj,
        pts1=pts1,
        pts2=pts2,
        inlier_mask=np.ones((n, 1), dtype=np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=True,
        X_cam1_h=X_world_h,
        X_world_h=X_world_h,
        cheiral_mask=cheiral,
        reproj={},
    )


def test_fusion_merges_when_shared_frame_pixels_overlap() -> None:
    fused = FusedLandmarkMap(merge_px=5.0)
    pts1 = np.array([[100.0, 100.0]], dtype=np.float32)
    pts2 = np.array([[200.0, 195.0]], dtype=np.float32)
    X = np.zeros((4, 1), dtype=np.float64)
    X[:3, 0] = [0.0, 0.0, 2.0]
    X[3, 0] = 1.0
    fused.integrate_two_view_result(_tw(0, 1, pts1, pts2, X, np.array([True])))

    pts1b = np.array([[202.0, 197.0]], dtype=np.float32)
    pts2b = np.array([[150.0, 150.0]], dtype=np.float32)
    Xb = np.zeros((4, 1), dtype=np.float64)
    Xb[:3, 0] = [0.05, 0.0, 2.1]
    Xb[3, 0] = 1.0
    fused.integrate_two_view_result(_tw(1, 2, pts1b, pts2b, Xb, np.array([True])))

    assert len(fused.landmarks) == 1
    assert fused.landmarks[0].n_updates == 2
    assert 1 in fused.landmarks[0].observations


def test_fusion_separate_when_far_apart() -> None:
    fused = FusedLandmarkMap(merge_px=2.0)
    X = np.zeros((4, 1), dtype=np.float64)
    X[:3, 0] = [0.0, 0.0, 1.0]
    X[3, 0] = 1.0
    fused.integrate_two_view_result(
        _tw(0, 1, np.array([[10.0, 10.0]], np.float32), np.array([[20.0, 20.0]], np.float32), X, np.array([True]))
    )
    fused.integrate_two_view_result(
        _tw(1, 2, np.array([[50.0, 50.0]], np.float32), np.array([[60.0, 60.0]], np.float32), X.copy(), np.array([True]))
    )
    assert len(fused.landmarks) == 2

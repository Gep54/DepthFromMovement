from __future__ import annotations

import numpy as np

from pipeline.triangulation import CHEIRAL_MIN_Z, cheiral_mask_cam_frames, triangulate_cam1_frame


def test_cheiral_min_z_constant() -> None:
    assert CHEIRAL_MIN_Z == -0.01


def test_cheiral_mask_threshold_and_both_cameras() -> None:
    R = np.eye(3)
    t = np.zeros((3, 1))
    at_thresh = np.array([[0.0], [0.0], [CHEIRAL_MIN_Z]])
    assert not bool(cheiral_mask_cam_frames(at_thresh, R, t)[0])
    above = np.array([[0.0], [0.0], [CHEIRAL_MIN_Z + 0.001]])
    assert bool(cheiral_mask_cam_frames(above, R, t)[0])

    # In front of cam1 (Z>0) but behind cam2 (Z<0) for a 180-deg yaw-like rotation.
    R_flip = np.diag([-1.0, 1.0, -1.0])
    X = np.array([[-1.0], [0.0], [1.0]])
    assert not bool(cheiral_mask_cam_frames(X, R_flip, t)[0])


def test_triangulate_cam1_frame_uses_cheiral_threshold() -> None:
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    pts1 = np.array([[320.0, 240.0]], dtype=np.float32)
    pts2 = np.array([[330.0, 240.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float64)
    t = np.array([[0.1], [0.0], [0.0]], dtype=np.float64)
    X_h, mask = triangulate_cam1_frame(pts1, pts2, K, R, t)
    assert mask.dtype == bool
    if mask.any():
        z1 = X_h[2, mask] / X_h[3, mask]
        assert np.all(z1 > CHEIRAL_MIN_Z)

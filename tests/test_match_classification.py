from __future__ import annotations

import numpy as np

from pipeline.map import TwoViewResult
from viz.match_classification import audit_record, classify_match_rejections, classification_counts


def test_classify_empty_pair() -> None:
    tw = TwoViewResult(
        frame_i=0,
        frame_j=1,
        pts1=np.zeros((0, 2), np.float32),
        pts2=np.zeros((0, 2), np.float32),
        inlier_mask=np.zeros((0, 1), np.uint8),
        E=None,
        R_est=None,
        t_est=None,
        scale=1.0,
        scale_ok=False,
        X_cam1_h=np.zeros((4, 0), np.float64),
        X_world_h=np.zeros((4, 0), np.float64),
        cheiral_mask=np.zeros((0,), bool),
        reproj={},
    )
    cls = classify_match_rejections(
        tw,
        np.eye(3),
        np.eye(4),
        np.eye(4),
    )
    assert classification_counts(cls) == {"epipolar": 0, "cheiral": 0, "reproj": 0, "inlier": 0}
    rec = audit_record(0, 1, cls)
    assert rec["has_all_rejection_types"] is False

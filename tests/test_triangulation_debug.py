from __future__ import annotations

import numpy as np

from pipeline.descriptor_landmark_map import point_camera_to_drone_to_world
from pipeline.geometry import invert_se3
from pipeline.map import TwoViewResult
from pipeline.triangulation_debug import (
    format_triangulation_debug_line,
    sample_random_integrate_point,
    triangulation_point_coords,
    valid_integrate_indices,
    x_cam_column,
)


def _make_tw(
    X_cam_cols: list[np.ndarray],
    *,
    cheiral: list[bool] | None = None,
) -> TwoViewResult:
    n = len(X_cam_cols)
    X_cam_h = np.zeros((4, n), dtype=np.float64)
    cheiral_mask = np.ones(n, dtype=bool) if cheiral is None else np.array(cheiral, dtype=bool)
    for k, X in enumerate(X_cam_cols):
        X_cam_h[:3, k] = X
        X_cam_h[3, k] = 1.0
    X_world_h = X_cam_h.copy()
    return TwoViewResult(
        frame_i=0,
        frame_j=1,
        pts1=np.zeros((n, 2)),
        pts2=np.zeros((n, 2)),
        inlier_mask=np.ones((n, 1), np.uint8),
        E=np.eye(3),
        R_est=np.eye(3),
        t_est=np.array([1.0, 0.0, 0.0]),
        scale=1.0,
        scale_ok=True,
        X_world_h=X_world_h,
        X_cam_h=X_cam_h,
        cheiral_mask=cheiral_mask,
        reproj={},
        descriptors=np.zeros((n, 32), dtype=np.uint8),
    )


def test_valid_integrate_indices_filters_cheiral_finite_range() -> None:
    W_drone = np.eye(4, dtype=np.float64)
    W_drone[:3, 3] = [0.0, 0.0, 0.0]
    W_cam = np.eye(4, dtype=np.float64)
    W_cam[:3, 3] = [0.0, 0.0, 0.0]
    tw = _make_tw(
        [
            np.array([1.0, 0.0, 2.0]),
            np.array([np.nan, 0.0, 2.0]),
            np.array([50.0, 0.0, 2.0]),
            np.array([1.0, 0.0, 3.0]),
        ],
        cheiral=[True, True, True, False],
    )
    valid = valid_integrate_indices(
        tw,
        world_T_camera_raw=W_cam,
        world_T_drone_raw=W_drone,
        world_T_camera_j_raw=W_cam,
        max_range_world=10.0,
    )
    assert valid == [0]


def test_triangulation_point_coords_round_trip() -> None:
    W_drone = np.eye(4, dtype=np.float64)
    W_drone[:3, 3] = [10.0, -2.0, 1.0]
    drone_T_cam = np.eye(4, dtype=np.float64)
    drone_T_cam[:3, 3] = [0.2, 0.0, -0.1]
    W_cam = W_drone @ invert_se3(drone_T_cam)
    X_cam = np.array([1.0, 0.5, 4.0], dtype=np.float64)
    tw = _make_tw([X_cam])
    X_cam_out, X_drone, X_world = triangulation_point_coords(
        tw,
        0,
        world_T_camera_raw=W_cam,
        world_T_drone_raw=W_drone,
    )
    np.testing.assert_allclose(X_cam_out, X_cam)
    X_world_direct = point_camera_to_drone_to_world(X_cam, W_cam, W_drone)
    np.testing.assert_allclose(X_world, X_world_direct, rtol=0, atol=1e-12)
    X_drone_expected = (invert_se3(W_drone) @ np.array([*X_world, 1.0]))[:3]
    np.testing.assert_allclose(X_drone, X_drone_expected, rtol=0, atol=1e-12)


def test_sample_random_integrate_point_deterministic_with_rng() -> None:
    W = np.eye(4, dtype=np.float64)
    tw = _make_tw([np.array([1.0, 0.0, 2.0]), np.array([2.0, 0.0, 3.0])])
    rng = np.random.default_rng(0)
    out1 = sample_random_integrate_point(
        tw,
        world_T_camera_raw=W,
        world_T_drone_raw=W,
        world_T_camera_j_raw=W,
        max_range_world=None,
        rng=rng,
    )
    rng2 = np.random.default_rng(0)
    out2 = sample_random_integrate_point(
        tw,
        world_T_camera_raw=W,
        world_T_drone_raw=W,
        world_T_camera_j_raw=W,
        max_range_world=None,
        rng=rng2,
    )
    assert out1 is not None and out2 is not None
    assert out1[0] == out2[0]


def test_format_triangulation_debug_line() -> None:
    line = format_triangulation_debug_line(
        keyframe_idx=5,
        frame_i=4,
        frame_j=5,
        col_k=2,
        X_cam=np.array([1.0, 2.0, 3.0]),
        X_drone=np.array([4.0, 5.0, 6.0]),
        X_world=np.array([7.0, 8.0, 9.0]),
    )
    assert "kf=5" in line
    assert "pair=4->5" in line
    assert "k=2" in line
    assert "cam=[1.0000, 2.0000, 3.0000]" in line


def test_x_cam_column_skips_non_finite() -> None:
    tw = _make_tw([np.array([np.nan, 1.0, 2.0])])
    assert x_cam_column(tw, 0) is None

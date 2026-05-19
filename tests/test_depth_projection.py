from __future__ import annotations

import numpy as np

from viz.overlays import project_world_points_to_camera_uv_z, render_sparse_depth_pixels


def test_project_world_points_includes_negative_camera_z() -> None:
    """Depth viz uses triangulated 3D as-is; no positive-Z cheiral re-check."""
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    world_T_camera = np.eye(4, dtype=np.float64)
    # Point 2 m in front of camera along +Z in world == camera frame here.
    X_front = np.array([[0.0], [0.0], [2.0]], dtype=np.float64)
    # Point 2 m behind camera (negative Z in camera frame after world transform).
    X_back = np.array([[0.0], [0.0], [-2.0]], dtype=np.float64)
    X_world_h = np.ones((4, 2), dtype=np.float64)
    X_world_h[:3, 0] = X_front[:, 0]
    X_world_h[:3, 1] = X_back[:, 0]
    mask = np.ones(2, dtype=bool)

    uv, z = project_world_points_to_camera_uv_z(X_world_h, mask, K, world_T_camera)
    assert uv.shape == (2, 2)
    assert z.shape == (2,)
    assert z[0] > 0
    assert z[1] < 0

    panel = render_sparse_depth_pixels(480, 640, uv, z, halo_radius=3)
    assert panel.shape == (480, 640, 3)
    assert int(panel.sum()) > 0

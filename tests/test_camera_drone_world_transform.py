from __future__ import annotations

import numpy as np

from pipeline.descriptor_landmark_map import point_camera_to_drone_to_world
from pipeline.geometry import invert_se3


def test_camera_drone_world_chain_matches_single_camera_to_world() -> None:
    W_drone = np.eye(4, dtype=np.float64)
    W_drone[:3, 3] = [10.0, -2.0, 1.0]
    drone_T_cam = np.eye(4, dtype=np.float64)
    drone_T_cam[:3, 3] = [0.2, 0.0, -0.1]
    W_cam = W_drone @ invert_se3(drone_T_cam)
    X_cam = np.array([1.0, 0.5, 4.0], dtype=np.float64)
    X_chain = point_camera_to_drone_to_world(X_cam, W_cam, W_drone)
    X_direct = (W_cam @ np.array([X_cam[0], X_cam[1], X_cam[2], 1.0]))[:3]
    np.testing.assert_allclose(X_chain, X_direct, rtol=0, atol=1e-12)

from __future__ import annotations

import numpy as np

from pipeline.geometry import canonicalize_world_T_camera_to_first, relative_motion_from_world_poses


def _random_se3(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = q
    T[:3, 3] = rng.standard_normal(3)
    return T


def test_canonicalize_first_pose_is_identity() -> None:
    W0 = np.eye(4, dtype=np.float64)
    W0[0, 3] = 2.5
    W1 = np.eye(4, dtype=np.float64)
    W1[1, 3] = 0.4
    out = canonicalize_world_T_camera_to_first([W0, W1])
    np.testing.assert_allclose(out[0], np.eye(4), atol=1e-12)


def test_canonicalize_preserves_relative_motions() -> None:
    rng = np.random.default_rng(0)
    raw = [_random_se3(rng) for _ in range(6)]
    canon = canonicalize_world_T_camera_to_first(raw)
    for i in range(len(raw)):
        for j in range(len(raw)):
            Ri, ti = relative_motion_from_world_poses(raw[i], raw[j])
            Rc, tc = relative_motion_from_world_poses(canon[i], canon[j])
            np.testing.assert_allclose(Ri, Rc, atol=1e-10)
            np.testing.assert_allclose(ti, tc, atol=1e-10)

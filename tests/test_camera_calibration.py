"""Tests for CameraInfo / intrinsics conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.camera_calibration import calibration_from_intrinsics, distortion_model_supported
from data.io_json import load_calibration_json, save_calibration_json
from data.schema import Calibration


def test_uav1_plumb_bob_zero_distortion():
    k_flat = [
        320.00000000000006,
        0.0,
        320.0,
        0.0,
        320.00000000000006,
        240.0,
        0.0,
        0.0,
        1.0,
    ]
    cal = calibration_from_intrinsics(
        K_flat=k_flat,
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        width=640,
        height=480,
        distortion_model="plumb_bob",
    )
    assert cal.image_size == (640, 480)
    assert cal.dist_coeffs is None
    assert cal.K.shape == (3, 3)
    np.testing.assert_allclose(cal.K[0, 0], 320.0)
    np.testing.assert_allclose(cal.K[1, 1], 320.0)
    np.testing.assert_allclose(cal.K[0, 2], 320.0)
    np.testing.assert_allclose(cal.K[1, 2], 240.0)


def test_plumb_bob_nonzero_distortion():
    d = [0.1, -0.05, 0.001, 0.002, 0.01]
    cal = calibration_from_intrinsics(
        K_flat=[400.0, 0, 320, 0, 400, 240, 0, 0, 1],
        D=d,
        width=640,
        height=480,
        distortion_model="plumb_bob",
    )
    assert cal.dist_coeffs is not None
    assert cal.dist_coeffs.shape == (5,)
    np.testing.assert_allclose(cal.dist_coeffs, d)


def test_invalid_k_length():
    with pytest.raises(ValueError, match="9 elements"):
        calibration_from_intrinsics(K_flat=[1, 2, 3], D=None, width=640, height=480)


def test_unknown_distortion_model_zeros_out_d():
    cal = calibration_from_intrinsics(
        K_flat=[400.0, 0, 320, 0, 400, 240, 0, 0, 1],
        D=[0.1, 0, 0, 0, 0],
        width=640,
        height=480,
        distortion_model="equidistant",
    )
    assert cal.dist_coeffs is None
    assert not distortion_model_supported("equidistant")


def test_save_load_roundtrip(tmp_path: Path):
    cal = Calibration(
        K=np.eye(3),
        dist_coeffs=np.array([0.1, 0.0, 0.0, 0.0, 0.0]),
        image_size=(640, 480),
    )
    path = tmp_path / "calibration.json"
    save_calibration_json(path, cal)
    loaded = load_calibration_json(path)
    np.testing.assert_allclose(loaded.K, cal.K)
    assert loaded.image_size == cal.image_size
    np.testing.assert_allclose(loaded.dist_coeffs, cal.dist_coeffs)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "dist_coeffs" in raw
    assert raw["image_size"] == [640, 480]

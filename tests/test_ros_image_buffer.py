from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from data.camera_calibration import calibration_from_intrinsics  # noqa: E402
from incremental_vo_ros2.image_buffer import (  # noqa: E402
    copy_image_msg,
    image_msg_to_gray_undistorted,
    should_buffer_image,
)


@dataclass
class FakeImage:
    height: int = 0
    width: int = 0
    encoding: str = ""
    is_bigendian: int = 0
    step: int = 0
    data: list[int] = field(default_factory=list)
    header: object | None = None


def _mono8_image(h: int = 10, w: int = 8) -> FakeImage:
    return FakeImage(
        height=h,
        width=w,
        encoding="mono8",
        step=w,
        data=list(np.arange(h * w, dtype=np.uint8)),
    )


def test_should_buffer_image_first_keyframe() -> None:
    assert should_buffer_image(None, 0.8, has_last_keyframe=False) is True
    assert should_buffer_image(0.0, 0.8, has_last_keyframe=False) is True


def test_should_buffer_image_fraction_gate() -> None:
    assert should_buffer_image(0.79, 0.8, has_last_keyframe=True) is False
    assert should_buffer_image(0.80, 0.8, has_last_keyframe=True) is True
    assert should_buffer_image(1.2, 0.8, has_last_keyframe=True) is True
    assert should_buffer_image(None, 0.8, has_last_keyframe=True) is False


def test_copy_image_msg_independent() -> None:
    src = _mono8_image()
    dup = copy_image_msg(src)
    src.data[0] = 255
    assert dup.data[0] != 255
    assert dup.height == src.height and dup.width == src.width


def test_image_msg_to_gray_undistorted_mono8() -> None:
    msg = _mono8_image()
    cal = calibration_from_intrinsics(
        K_flat=[400.0, 0, 4, 0, 400, 5, 0, 0, 1],
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        width=8,
        height=10,
        distortion_model="plumb_bob",
    )
    gray, k_eff = image_msg_to_gray_undistorted(msg, cal)
    assert gray is not None
    assert gray.shape == (10, 8)
    assert gray.dtype == np.uint8
    assert k_eff is None


def test_image_msg_to_gray_undistorted_unsupported() -> None:
    msg = _mono8_image()
    msg.encoding = "32FC1"
    gray, k_eff = image_msg_to_gray_undistorted(msg, None)
    assert gray is None
    assert k_eff is None

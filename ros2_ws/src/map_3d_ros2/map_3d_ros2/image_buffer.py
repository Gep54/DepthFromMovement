"""Keyframe image buffering helpers (no ROS message imports — testable without rclpy)."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "copy_image_msg",
    "image_msg_to_gray_undistorted",
    "ros_image_to_gray",
    "should_buffer_image",
]


def should_buffer_image(
    fraction: float | None,
    start_fraction: float,
    *,
    has_last_keyframe: bool,
) -> bool:
    """Whether to retain a raw image for keyframe candidacy.

    ``fraction`` is ``‖pos − last_kf‖ / motion_threshold_m``; ``None`` means no keyframe yet.
    """
    if not has_last_keyframe:
        return True
    if fraction is None:
        return False
    return float(fraction) >= float(start_fraction)


def copy_image_msg(msg):
    """Deep-copy an image message (``sensor_msgs/Image`` or compatible duck type)."""
    out = type(msg)()
    out.header = msg.header
    out.height = int(msg.height)
    out.width = int(msg.width)
    out.encoding = msg.encoding
    out.is_bigendian = msg.is_bigendian
    out.step = int(msg.step)
    out.data = list(msg.data)
    return out


def _image_data_uint8(msg) -> np.ndarray:
    raw = msg.data
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return np.frombuffer(raw, dtype=np.uint8)
    return np.asarray(raw, dtype=np.uint8)


def ros_image_to_gray(msg) -> np.ndarray | None:
    """Decode image message to single-channel uint8; returns None if encoding unsupported."""
    h, w = int(msg.height), int(msg.width)
    arr = _image_data_uint8(msg)
    if msg.encoding in ("mono8", "8UC1"):
        if arr.size != h * w:
            return None
        return arr.reshape((h, w))
    if msg.encoding in ("bgr8", "8UC3"):
        if arr.size != h * w * 3:
            return None
        bgr = arr.reshape((h, w, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if msg.encoding in ("rgb8", "rgba8"):
        step = 4 if msg.encoding == "rgba8" else 3
        if arr.size != h * w * step:
            return None
        rgb = arr.reshape((h, w, step))[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return None


def image_msg_to_gray_undistorted(
    msg, cal
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Decode → grayscale; undistort when ``cal`` has non-zero distortion.

    Returns ``(gray, k_eff)``; ``k_eff`` is ``None`` when distortion is negligible.
    """
    gray = ros_image_to_gray(msg)
    if gray is None:
        return None, None
    if cal is None:
        return gray, None
    if cal.dist_coeffs is None or np.linalg.norm(cal.dist_coeffs) < 1e-9:
        return gray, None
    if cal.image_size is None:
        return gray, None
    w, h = cal.image_size
    new_K, _ = cv2.getOptimalNewCameraMatrix(cal.K, cal.dist_coeffs, (w, h), alpha=0)
    new_K = np.asarray(new_K, dtype=np.float64)
    und = cv2.undistort(gray, cal.K, cal.dist_coeffs, None, newK=new_K)
    return und, new_K

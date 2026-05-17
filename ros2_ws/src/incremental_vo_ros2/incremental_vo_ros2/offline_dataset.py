"""Offline dataset layout helpers (no ROS imports)."""

from __future__ import annotations

__all__ = ["offline_dataset_image_basename"]


def offline_dataset_image_basename(prefix: str, index: int) -> str:
    """Numbered image filename for offline datasets (e.g. ``frame_00000.png``)."""
    safe = prefix.strip() or "frame"
    return f"{safe}_{index:05d}.png"

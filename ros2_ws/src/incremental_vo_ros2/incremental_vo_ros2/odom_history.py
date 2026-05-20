"""Odometry ring buffer and stamp lookup (no ROS message imports in helpers)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


def odom_stamp_nsec(msg: Any) -> int:
    """Odometry header time as integer nanoseconds."""
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


@dataclass
class _OdomSample:
    stamp_nsec: int
    msg: Any


class OdomHistory:
    """Ring buffer of odometry messages for nearest-stamp lookup (image sync)."""

    def __init__(self, maxlen: int = 800) -> None:
        self._samples: deque[_OdomSample] = deque(maxlen=max(16, int(maxlen)))

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def push(self, msg: Any) -> None:
        self._samples.append(_OdomSample(stamp_nsec=odom_stamp_nsec(msg), msg=msg))

    def nearest(self, stamp_sec: int, stamp_nsec: int) -> Any | None:
        """Return the odometry sample closest to ``(stamp_sec, stamp_nsec)``."""
        if not self._samples:
            return None
        target = int(stamp_sec) * 1_000_000_000 + int(stamp_nsec)
        best = min(self._samples, key=lambda s: abs(s.stamp_nsec - target))
        return best.msg

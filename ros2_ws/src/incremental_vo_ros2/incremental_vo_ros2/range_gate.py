"""Re-export sparse-map range helpers from ``pipeline`` (no ROS imports)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.range_gate import consecutive_keyframe_baseline_m, max_sparse_range_m

__all__ = ["consecutive_keyframe_baseline_m", "max_sparse_range_m"]

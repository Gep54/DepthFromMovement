from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_fusion_config() -> dict[str, Any]:
    return {"method": "position_blend", "position_blend_weight": 0.5}


def load_fusion_json(path: Path) -> dict[str, Any]:
    """Load optional ``fusion.json``; unknown keys ignored."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = default_fusion_config()
    if "method" in raw:
        out["method"] = str(raw["method"])
    if "position_blend_weight" in raw:
        out["position_blend_weight"] = float(raw["position_blend_weight"])
    return out

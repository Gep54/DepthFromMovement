from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import Calibration, MotionSpec


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_calibration_json(path: Path) -> Calibration:
    raw = _read_json(path)
    if "K" not in raw:
        raise KeyError("calibration.json must contain 'K' (3x3 intrinsics matrix)")
    K = np.asarray(raw["K"], dtype=np.float64)
    dist = raw.get("dist_coeffs") or raw.get("distortion")
    dist_arr = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    size = raw.get("image_size")
    image_size: tuple[int, int] | None = None
    if size is not None:
        if len(size) != 2:
            raise ValueError("image_size must be [width, height]")
        image_size = (int(size[0]), int(size[1]))
    return Calibration(K=K, dist_coeffs=dist_arr, image_size=image_size)


def save_calibration_json(path: Path, cal: Calibration) -> None:
    """Write ``calibration.json`` in the same schema as :func:`load_calibration_json`."""
    payload: dict[str, Any] = {"K": cal.K.tolist()}
    if cal.dist_coeffs is not None and cal.dist_coeffs.size > 0:
        payload["dist_coeffs"] = cal.dist_coeffs.reshape(-1).tolist()
    if cal.image_size is not None:
        payload["image_size"] = [int(cal.image_size[0]), int(cal.image_size[1])]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_motion_json(path: Path) -> MotionSpec:
    raw = _read_json(path)
    pose_convention = raw.get("pose_convention", "world_T_camera")
    if pose_convention not in ("world_T_camera", "camera_T_world"):
        raise ValueError("pose_convention must be 'world_T_camera' or 'camera_T_world'")
    representation = raw.get("representation", "absolute")
    if representation not in ("absolute", "relative_to_prev"):
        raise ValueError("representation must be 'absolute' or 'relative_to_prev'")
    frames = raw.get("frames")
    if frames is None:
        raise KeyError("motion.json must contain 'frames' list")
    transforms: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        if isinstance(fr, dict) and "T" in fr:
            transforms.append(np.asarray(fr["T"], dtype=np.float64))
        else:
            transforms.append(np.asarray(fr, dtype=np.float64))
    return MotionSpec(
        pose_convention=pose_convention,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        transforms=transforms,
    )


def _inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def world_T_camera_from_motion(spec: MotionSpec) -> list[np.ndarray]:
    """Canonical world_T_camera (4x4) per frame for triangulation / odometry alignment."""
    Tlist = spec.transforms
    if spec.representation == "absolute":
        if spec.pose_convention == "world_T_camera":
            return [np.array(x, copy=True) for x in Tlist]
        return [_inv_se3(np.asarray(x, dtype=np.float64)) for x in Tlist]
    if spec.pose_convention == "world_T_camera":
        out = [np.array(Tlist[0], copy=True)]
        for i in range(1, len(Tlist)):
            out.append(out[-1] @ np.asarray(Tlist[i], dtype=np.float64))
        return out
    out = [_inv_se3(np.asarray(Tlist[0], dtype=np.float64))]
    for i in range(1, len(Tlist)):
        out.append(out[-1] @ np.asarray(Tlist[i], dtype=np.float64))
    return out

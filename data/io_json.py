from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from data.schema import Calibration, MotionSpec

PoseConvention = Literal["world_T_camera", "camera_T_world"]
MotionRepresentation = Literal["absolute", "relative_to_prev"]


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
    def _opt_str(key: str) -> str | None:
        v = raw.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    return MotionSpec(
        pose_convention=pose_convention,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        transforms=transforms,
        world_frame=_opt_str("world_frame"),
        target_world_frame=_opt_str("target_world_frame"),
        pose_frame=_opt_str("pose_frame"),
        camera_frame=_opt_str("camera_frame"),
        tf_static_file=_opt_str("tf_static_file"),
    )


def save_motion_json(
    path: Path,
    transforms: Sequence[np.ndarray],
    *,
    pose_convention: PoseConvention = "world_T_camera",
    representation: MotionRepresentation = "absolute",
    filenames: Sequence[str] | None = None,
    world_frame: str | None = None,
    target_world_frame: str | None = None,
    pose_frame: str | None = None,
    camera_frame: str | None = None,
) -> None:
    """Write ``motion.json`` in the same schema as :func:`load_motion_json`."""
    if filenames is not None and len(filenames) != len(transforms):
        raise ValueError(
            f"filenames length {len(filenames)} != transforms length {len(transforms)}"
        )
    frames: list[dict[str, Any]] = []
    for i, T in enumerate(transforms):
        Ta = np.asarray(T, dtype=np.float64)
        if Ta.shape == (3, 4):
            row = np.eye(4, dtype=np.float64)
            row[:3, :4] = Ta
            Ta = row
        fr: dict[str, Any] = {"index": i, "T": Ta.tolist()}
        if filenames is not None:
            fr["filename"] = filenames[i]
        frames.append(fr)
    payload: dict[str, Any] = {
        "pose_convention": pose_convention,
        "representation": representation,
        "frames": frames,
    }
    if world_frame:
        payload["world_frame"] = world_frame
    if target_world_frame:
        payload["target_world_frame"] = target_world_frame
    if pose_frame:
        payload["pose_frame"] = pose_frame
    if camera_frame:
        payload["camera_frame"] = camera_frame
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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

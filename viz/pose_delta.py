"""Per-pair camera pose difference figure and JSON for export runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.geometry import relative_motion_from_world_poses

POSE_DELTA_PNG = "pose_delta.png"
POSE_DELTA_JSON = "pose_delta.json"


def _camera_center_world(T: np.ndarray) -> np.ndarray:
    return np.asarray(T[:3, 3], dtype=np.float64).ravel()


def _camera_forward_xy(T: np.ndarray, length_m: float = 0.08) -> np.ndarray:
    """World XY tip of a short arrow along camera +Z (optical axis)."""
    R = np.asarray(T[:3, :3], dtype=np.float64)
    origin = _camera_center_world(T)
    forward = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(forward[:2]))
    if n < 1e-9:
        forward = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        n = float(np.linalg.norm(forward[:2]))
    if n < 1e-9:
        return origin[:2]
    scale = length_m / n
    tip = origin[:2] + forward[:2] * scale
    return tip


def _rotation_magnitude_deg(R: np.ndarray) -> float:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return float(np.linalg.norm(rvec.ravel()) * 180.0 / np.pi)


def compute_pair_pose_delta(
    world_T_i: np.ndarray,
    world_T_j: np.ndarray,
    *,
    frame_i: int,
    frame_j: int,
) -> dict[str, Any]:
    """Numeric summary of pose difference between frames ``i`` and ``j``."""
    R_rel, t_rel = relative_motion_from_world_poses(world_T_i, world_T_j)
    ci = _camera_center_world(world_T_i)
    cj = _camera_center_world(world_T_j)
    delta_w = cj - ci
    baseline = float(np.linalg.norm(t_rel))
    return {
        "frame_i": int(frame_i),
        "frame_j": int(frame_j),
        "camera_i_position_world_m": ci.tolist(),
        "camera_j_position_world_m": cj.tolist(),
        "translation_world_m": delta_w.tolist(),
        "translation_cam_i_m": np.asarray(t_rel, dtype=np.float64).ravel().tolist(),
        "baseline_m": baseline,
        "rotation_cam_i_to_cam_j_deg": _rotation_magnitude_deg(R_rel),
        "rotation_matrix_cam_i_to_cam_j": np.asarray(R_rel, dtype=np.float64).tolist(),
    }


def render_pair_pose_delta(
    summary: dict[str, Any],
    world_T_i: np.ndarray,
    world_T_j: np.ndarray,
    *,
    width: int = 1024,
    height: int = 512,
) -> np.ndarray:
    """BGR figure: top-down XY plot (left) and numeric summary (right)."""
    plot_w = height
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    plot = canvas[:, :plot_w]
    text_panel = canvas[:, plot_w:]

    ci = _camera_center_world(world_T_i)
    cj = _camera_center_world(world_T_j)
    tips = np.array([_camera_forward_xy(world_T_i), _camera_forward_xy(world_T_j)], dtype=np.float64)
    pts_xy = np.vstack([ci[:2], cj[:2], tips])
    mn = np.min(pts_xy, axis=0)
    mx = np.max(pts_xy, axis=0)
    span = np.maximum(mx - mn, 1e-6)
    margin = 0.35
    mn = mn - span * margin
    mx = mx + span * margin
    span = mx - mn

    def proj(p: np.ndarray) -> tuple[int, int]:
        x = int((p[0] - mn[0]) / span[0] * (plot_w - 40) + 20)
        y = int((1.0 - (p[1] - mn[1]) / span[1]) * (height - 40) + 20)
        return x, y

    p_i = proj(ci[:2])
    p_j = proj(cj[:2])
    cv2.arrowedLine(plot, p_i, p_j, (40, 40, 220), 3, tipLength=0.12, line_type=cv2.LINE_AA)
    for label, T, color in (
        (f"i={summary['frame_i']}", world_T_i, (200, 80, 80)),
        (f"j={summary['frame_j']}", world_T_j, (80, 80, 200)),
    ):
        p0 = proj(_camera_center_world(T)[:2])
        p1 = proj(_camera_forward_xy(T))
        cv2.arrowedLine(plot, p0, p1, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.circle(plot, p0, 10, color, -1, cv2.LINE_AA)
        cv2.putText(plot, label, (p0[0] + 12, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(plot, label, (p0[0] + 12, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    cv2.putText(
        plot,
        "top-down XY (world)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        plot,
        "red/blue dots = cam i/j; arrows = +Z; thick = delta",
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    def fmt3(v: list[float]) -> str:
        return f"[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]"

    tw = summary["translation_world_m"]
    tc = summary["translation_cam_i_m"]
    lines = [
        f"Pair {summary['frame_i']} -> {summary['frame_j']}",
        "",
        f"Baseline (cam i frame): {summary['baseline_m']:.4f} m",
        f"Rotation i->j: {summary['rotation_cam_i_to_cam_j_deg']:.2f} deg",
        "",
        f"Cam i world: {fmt3(summary['camera_i_position_world_m'])}",
        f"Cam j world: {fmt3(summary['camera_j_position_world_m'])}",
        "",
        f"Delta world (j-i): {fmt3(tw)}",
        f"Delta cam i (OpenCV): {fmt3(tc)}",
    ]
    y0 = 36
    for k, line in enumerate(lines):
        cv2.putText(
            text_panel,
            line,
            (16, y0 + k * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def export_pair_pose_delta(
    pair_run_dir: str | Path,
    world_T_i: np.ndarray,
    world_T_j: np.ndarray,
    *,
    frame_i: int,
    frame_j: int,
) -> tuple[Path, Path]:
    """Write ``pose_delta.png`` and ``pose_delta.json`` under ``pair_run_dir``."""
    root = Path(pair_run_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = compute_pair_pose_delta(world_T_i, world_T_j, frame_i=frame_i, frame_j=frame_j)
    png_path = root / POSE_DELTA_PNG
    json_path = root / POSE_DELTA_JSON
    img = render_pair_pose_delta(summary, world_T_i, world_T_j)
    if not cv2.imwrite(str(png_path), img):
        raise RuntimeError(f"failed to write {png_path}")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return png_path, json_path

"""Per-pair camera pose difference figure and JSON for export runs."""

from __future__ import annotations

import json
import math
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


def _nice_scale_length_m(target_m: float) -> float:
    """Round ``target_m`` to a readable scale-bar length (1–2–5 decades)."""
    if target_m <= 0.0:
        return 0.1
    exp = math.floor(math.log10(target_m))
    base = target_m / (10.0**exp)
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0**exp))


def _tick_step_m(span_m: float) -> float:
    return _nice_scale_length_m(max(span_m / 4.0, 1e-6))


class _WorldXYPlot:
    """Equal-aspect top-down plot: world +X right, world +Y up on screen."""

    def __init__(
        self,
        plot: np.ndarray,
        points_xy: np.ndarray,
        *,
        pad: int = 56,
    ) -> None:
        h, w = plot.shape[:2]
        self._plot = plot
        self._pad = pad
        self._w = w
        self._h = h

        mn = np.min(points_xy, axis=0)
        mx = np.max(points_xy, axis=0)
        center = 0.5 * (mn + mx)
        half = 0.5 * max(float(mx[0] - mn[0]), float(mx[1] - mn[1]), 1e-6)
        half *= 1.28
        self.mn = center - half
        self.mx = center + half
        self.span = float(self.mx[0] - self.mn[0])

        inner_w = max(1, w - 2 * pad)
        inner_h = max(1, h - 2 * pad)
        self.px_per_m = min(inner_w / self.span, inner_h / self.span)

    def proj(self, p: np.ndarray) -> tuple[int, int]:
        p = np.asarray(p, dtype=np.float64).reshape(2)
        x = int(self._pad + (p[0] - self.mn[0]) * self.px_per_m)
        y = int(self._h - self._pad - (p[1] - self.mn[1]) * self.px_per_m)
        return x, y

    def draw_axes(self, wf: str) -> None:
        """World +X / +Y reference arrows at the plot corner (metric, not camera frame)."""
        ox, oy = self.mn[0] + 0.06 * self.span, self.mn[1] + 0.06 * self.span
        axis_len = max(0.12 * self.span, 0.05)
        o = np.array([ox, oy], dtype=np.float64)
        px_x = self.proj(o + np.array([axis_len, 0.0]))
        px_y = self.proj(o + np.array([0.0, axis_len]))
        p0 = self.proj(o)
        col_axis = (70, 70, 70)
        cv2.arrowedLine(self._plot, p0, px_x, col_axis, 2, tipLength=0.18, line_type=cv2.LINE_AA)
        cv2.arrowedLine(self._plot, p0, px_y, col_axis, 2, tipLength=0.18, line_type=cv2.LINE_AA)
        cv2.putText(
            self._plot,
            "+X",
            (px_x[0] + 4, px_x[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col_axis,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            self._plot,
            "+Y",
            (px_y[0] + 4, px_y[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col_axis,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            self._plot,
            f"axes: {wf}",
            (p0[0], p0[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    def draw_grid_ticks(self) -> None:
        step = _tick_step_m(self.span)
        x0 = math.ceil(self.mn[0] / step) * step
        y0 = math.ceil(self.mn[1] / step) * step
        col_grid = (220, 220, 220)
        col_tick = (110, 110, 110)
        x = x0
        while x <= self.mx[0] + 1e-9:
            p0 = self.proj(np.array([x, self.mn[1]]))
            p1 = self.proj(np.array([x, self.mx[1]]))
            cv2.line(self._plot, p0, p1, col_grid, 1, cv2.LINE_AA)
            cv2.putText(
                self._plot,
                f"{x:.2f}",
                (p0[0] - 18, self._h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                col_tick,
                1,
                cv2.LINE_AA,
            )
            x += step
        y = y0
        while y <= self.mx[1] + 1e-9:
            p0 = self.proj(np.array([self.mn[0], y]))
            p1 = self.proj(np.array([self.mx[0], y]))
            cv2.line(self._plot, p0, p1, col_grid, 1, cv2.LINE_AA)
            cv2.putText(
                self._plot,
                f"{y:.2f}",
                (8, p0[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                col_tick,
                1,
                cv2.LINE_AA,
            )
            y += step

    def draw_scale_bar(self) -> None:
        target_px = 100.0
        bar_m = _nice_scale_length_m(target_px / self.px_per_m)
        bar_px = int(bar_m * self.px_per_m)
        x0 = self._w - self._pad - bar_px - 20
        y0 = self._h - 28
        x1 = x0 + bar_px
        cv2.line(self._plot, (x0, y0), (x1, y0), (30, 30, 30), 3, cv2.LINE_AA)
        cv2.line(self._plot, (x0, y0 - 6), (x0, y0 + 6), (30, 30, 30), 2, cv2.LINE_AA)
        cv2.line(self._plot, (x1, y0 - 6), (x1, y0 + 6), (30, 30, 30), 2, cv2.LINE_AA)
        label = f"{bar_m:g} m"
        cv2.putText(
            self._plot,
            label,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )


def compute_pair_pose_delta(
    world_T_i: np.ndarray,
    world_T_j: np.ndarray,
    *,
    frame_i: int,
    frame_j: int,
    world_frame: str | None = None,
) -> dict[str, Any]:
    """Numeric summary of pose difference between frames ``i`` and ``j``."""
    R_rel, t_rel = relative_motion_from_world_poses(world_T_i, world_T_j)
    ci = _camera_center_world(world_T_i)
    cj = _camera_center_world(world_T_j)
    delta_w = cj - ci
    baseline = float(np.linalg.norm(t_rel))
    Ri = np.asarray(world_T_i[:3, :3], dtype=np.float64)
    forward_w = Ri[:, 2]
    dw_norm = float(np.linalg.norm(delta_w))
    if dw_norm > 1e-9 and np.linalg.norm(forward_w) > 1e-9:
        motion_vs_optical_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(forward_w @ (delta_w / dw_norm)), -1.0, 1.0)
                )
            )
        )
    else:
        motion_vs_optical_deg = None
    out: dict[str, Any] = {
        "frame_i": int(frame_i),
        "frame_j": int(frame_j),
        "camera_i_position_world_m": ci.tolist(),
        "camera_j_position_world_m": cj.tolist(),
        "translation_world_m": delta_w.tolist(),
        "translation_cam_i_m": np.asarray(t_rel, dtype=np.float64).ravel().tolist(),
        "baseline_m": baseline,
        "rotation_cam_i_to_cam_j_deg": _rotation_magnitude_deg(R_rel),
        "rotation_matrix_cam_i_to_cam_j": np.asarray(R_rel, dtype=np.float64).tolist(),
        "forward_axis_world_m": forward_w.tolist(),
        "motion_vs_optical_axis_deg": motion_vs_optical_deg,
    }
    if world_frame:
        out["world_frame"] = world_frame
    return out


def render_pair_pose_delta(
    summary: dict[str, Any],
    world_T_i: np.ndarray,
    world_T_j: np.ndarray,
    *,
    width: int = 1024,
    height: int = 512,
) -> np.ndarray:
    """BGR figure: top-down world XY (left) and numeric summary (right)."""
    plot_w = height
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    plot = canvas[:, :plot_w]
    text_panel = canvas[:, plot_w:]

    ci = _camera_center_world(world_T_i)
    cj = _camera_center_world(world_T_j)
    delta_w = np.asarray(summary["translation_world_m"], dtype=np.float64)
    tips = np.array([_camera_forward_xy(world_T_i), _camera_forward_xy(world_T_j)], dtype=np.float64)
    pts_xy = np.vstack([ci[:2], cj[:2], ci[:2] + delta_w[:2], tips])

    wf = str(summary.get("world_frame", "world"))
    xy = _WorldXYPlot(plot, pts_xy)
    xy.draw_grid_ticks()
    xy.draw_axes(wf)
    xy.draw_scale_bar()

    p_i = xy.proj(ci[:2])
    p_j = xy.proj(cj[:2])
    dw_norm = float(np.linalg.norm(delta_w))
    motion_color = (30, 140, 30)  # green: world translation j-i
    cv2.arrowedLine(plot, p_i, p_j, motion_color, 4, tipLength=0.15, line_type=cv2.LINE_AA)
    mid = ((p_i[0] + p_j[0]) // 2, (p_i[1] + p_j[1]) // 2)
    cv2.putText(
        plot,
        f"Delta t world ({dw_norm:.3f} m)",
        (mid[0] + 10, mid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        motion_color,
        2,
        cv2.LINE_AA,
    )

    for label, T, color in (
        (f"i={summary['frame_i']}", world_T_i, (60, 60, 220)),
        (f"j={summary['frame_j']}", world_T_j, (220, 60, 60)),
    ):
        p0 = xy.proj(_camera_center_world(T)[:2])
        p1 = xy.proj(_camera_forward_xy(T))
        cv2.arrowedLine(plot, p0, p1, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.circle(plot, p0, 9, color, -1, cv2.LINE_AA)
        cv2.putText(
            plot,
            label,
            (p0[0] + 12, p0[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            plot,
            label,
            (p0[0] + 12, p0[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        plot,
        f"top-down world XY  ({wf})",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        plot,
        "green = translation in world (j-i); thin = camera +Z",
        (12, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        plot,
        "units: metres",
        (12, height - 34),
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
        f"World frame: {wf}",
        "",
        f"|Delta t| world: {dw_norm:.4f} m",
        f"Delta t world (j-i): {fmt3(tw)}",
        f"Delta t cam i (OpenCV): {fmt3(tc)}",
        "",
        f"Baseline ||t_cam||: {summary['baseline_m']:.4f} m",
        f"Rotation i->j: {summary['rotation_cam_i_to_cam_j_deg']:.2f} deg",
        "",
        f"Cam i world: {fmt3(summary['camera_i_position_world_m'])}",
        f"Cam j world: {fmt3(summary['camera_j_position_world_m'])}",
    ]
    if summary.get("motion_vs_optical_axis_deg") is not None:
        lines.append(f"Angle(motion, cam i +Z): {summary['motion_vs_optical_axis_deg']:.2f} deg")
    y0 = 36
    for k, line in enumerate(lines):
        cv2.putText(
            text_panel,
            line,
            (16, y0 + k * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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
    world_frame: str | None = None,
) -> tuple[Path, Path]:
    """Write ``pose_delta.png`` and ``pose_delta.json`` under ``pair_run_dir``."""
    root = Path(pair_run_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = compute_pair_pose_delta(
        world_T_i, world_T_j, frame_i=frame_i, frame_j=frame_j, world_frame=world_frame
    )
    png_path = root / POSE_DELTA_PNG
    json_path = root / POSE_DELTA_JSON
    img = render_pair_pose_delta(summary, world_T_i, world_T_j)
    if not cv2.imwrite(str(png_path), img):
        raise RuntimeError(f"failed to write {png_path}")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return png_path, json_path

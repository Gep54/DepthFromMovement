from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy.interpolate import griddata

from pipeline.geometry import invert_se3

_BGR_RED_NEAR = np.array([0.0, 0.0, 255.0], dtype=np.float64)
_BGR_BLUE_FAR = np.array([255.0, 0.0, 0.0], dtype=np.float64)


def depth_to_bgr_red_near_blue_far(z_m: np.ndarray | float, lo_m: float, hi_m: float) -> np.ndarray:
    """
    Linear BGR: **closest** depth → saturated red, **farthest** → saturated blue.

    ``z_m`` may be a scalar or any shaped array; returns shape ``(..., 3)`` uint8 BGR.
    """
    z = np.asarray(z_m, dtype=np.float64)
    t = (z - lo_m) / (hi_m - lo_m + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    base = (1.0 - t)[..., np.newaxis] * _BGR_RED_NEAR + t[..., np.newaxis] * _BGR_BLUE_FAR
    return np.clip(np.round(base), 0, 255).astype(np.uint8)


def blend_bgr_toward_white(bgr: np.ndarray, white_frac: float) -> np.ndarray:
    """``white_frac`` in ``[0, 1]``: blend toward white."""
    w = np.array([255.0, 255.0, 255.0], dtype=np.float64)
    x = np.asarray(bgr, dtype=np.float64).reshape(3)
    f = float(np.clip(white_frac, 0.0, 1.0))
    out = (1.0 - f) * x + f * w
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def draw_depth_point_halo(
    canvas: np.ndarray,
    u: int,
    v: int,
    z_m: float,
    lo_m: float,
    hi_m: float,
    *,
    outer_radius: int,
) -> None:
    """Filled circles from outer (lighter) to inner (full depth colour)."""
    base = depth_to_bgr_red_near_blue_far(z_m, lo_m, hi_m).reshape(3)
    h, w = canvas.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return
    if outer_radius <= 0:
        canvas[v, u] = base
        return
    mx = max(int(outer_radius), 1)
    for r in range(mx, -1, -1):
        wf = (r / mx) ** 1.12
        wf = min(0.92, wf)
        col = blend_bgr_toward_white(base, wf)
        cv2.circle(canvas, (u, v), r, col.tolist(), -1, lineType=cv2.LINE_AA)


def project_world_points_to_camera_uv_z(
    X_world_h: np.ndarray,
    cheiral_mask: np.ndarray,
    K: np.ndarray,
    world_T_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project triangulated homogeneous world points into the camera frame.
    Returns pixel coords ``uv`` (N, 2) as float (u, v) and positive depths ``z_cam`` (N,) in metres.
    """
    X = X_world_h[:3, :]
    valid = cheiral_mask & np.all(np.isfinite(X), axis=0)
    if not np.any(valid):
        return np.zeros((0, 2), np.float64), np.zeros(0, np.float64)
    X = X[:, valid]
    Tcw = invert_se3(world_T_camera)
    R = Tcw[:3, :3]
    t = Tcw[:3, 3].reshape(3, 1)
    Xc = R @ X + t
    z = Xc[2, :]
    front = z > 1e-6
    Xc = Xc[:, front]
    z = z[front]
    if z.size == 0:
        return np.zeros((0, 2), np.float64), np.zeros(0, np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (Xc[0, :] / z) + cx
    v = fy * (Xc[1, :] / z) + cy
    uv = np.stack([u, v], axis=1)
    return uv.astype(np.float64), z.astype(np.float64)


def depth_colormap_range_m(
    z_cam_m: np.ndarray,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float]:
    z = z_cam_m[np.isfinite(z_cam_m) & (z_cam_m > 0)]
    if z.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(z, percentile)
    if hi <= lo + 1e-9:
        hi = lo + 1e-3
    return float(lo), float(hi)


def splat_min_depth_per_pixel(
    height: int,
    width: int,
    uv: np.ndarray,
    z_cam_m: np.ndarray,
) -> np.ndarray:
    """Sparse depth sheet ``z[v,u]`` with minimum depth winning at duplicated pixels."""
    zz = np.full((height, width), np.nan, dtype=np.float64)
    for k in range(len(z_cam_m)):
        zz_k = float(z_cam_m[k])
        if not np.isfinite(zz_k) or zz_k <= 0:
            continue
        u, v = int(round(float(uv[k, 0]))), int(round(float(uv[k, 1])))
        if not (0 <= u < width and 0 <= v < height):
            continue
        if not np.isfinite(zz[v, u]) or zz_k < zz[v, u]:
            zz[v, u] = zz_k
    return zz


def render_sparse_depth_pixels(
    height: int,
    width: int,
    uv: np.ndarray,
    z_cam_m: np.ndarray,
    *,
    z_lo_m: float | None = None,
    z_hi_m: float | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
    halo_radius: int = 10,
    background: Literal["dark", "photo"] = "dark",
    bgr_background: np.ndarray | None = None,
) -> np.ndarray:
    """
    Colour each depth sample **red = near**, **blue = far**, with a soft halo (lighter tints
    of the same hue outward). Use ``halo_radius=0`` for a single-pixel dot only.
    """
    if z_cam_m.size == 0:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(canvas, "no depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        return canvas
    if z_lo_m is not None and z_hi_m is not None:
        lo, hi = float(z_lo_m), float(z_hi_m)
    else:
        lo, hi = depth_colormap_range_m(z_cam_m, percentile)
    if background == "photo" and bgr_background is not None and bgr_background.shape[:2] == (height, width):
        canvas = (bgr_background.astype(np.float32) * 0.35).astype(np.uint8)
    else:
        canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    hr = max(0, int(halo_radius))
    for k in range(len(z_cam_m)):
        z = float(z_cam_m[k])
        if not np.isfinite(z) or z <= 0:
            continue
        u, v = int(round(float(uv[k, 0]))), int(round(float(uv[k, 1])))
        draw_depth_point_halo(canvas, u, v, z, lo, hi, outer_radius=hr)
    cv2.putText(
        canvas,
        f"Z {lo:.3f}..{hi:.3f} m  red=near blue=far",
        (10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return canvas


def render_dense_depth_colormap(
    height: int,
    width: int,
    uv: np.ndarray,
    z_cam_m: np.ndarray,
    *,
    z_lo_m: float | None = None,
    z_hi_m: float | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
    interp: Literal["linear", "nearest"] = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate sparse depths to a full raster (linear with nearest-NaN fill), then colour with
    **red=near, blue=far** (same linear mapping as sparse halos).

    Returns ``(bgr, z_map_m)`` where ``z_map_m`` is NaN outside the sampled hull / extrapolation.
    """
    zz = splat_min_depth_per_pixel(height, width, uv, z_cam_m)
    finite = np.isfinite(zz)
    if not np.any(finite):
        empty = np.zeros((height, width, 3), dtype=np.uint8)
        return empty, zz
    ys, xs = np.where(finite)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    values = zz[finite]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    method = interp if interp == "nearest" else "linear"
    zi = griddata(points, values, (grid_x, grid_y), method=method)
    if method == "linear":
        zn = griddata(points, values, (grid_x, grid_y), method="nearest")
        zi = np.where(np.isnan(zi), zn, zi)
    if z_lo_m is not None and z_hi_m is not None:
        lo, hi = float(z_lo_m), float(z_hi_m)
    else:
        lo, hi = depth_colormap_range_m(values, percentile)
    valid = np.isfinite(zi) & (zi > 0)
    color = np.full((height, width, 3), 12, dtype=np.uint8)
    if np.any(valid):
        rgb_layer = depth_to_bgr_red_near_blue_far(zi, lo, hi)
        color = np.where(valid[..., np.newaxis], rgb_layer, color)
    cv2.putText(
        color,
        f"dense Z {lo:.3f}..{hi:.3f} m  red=near blue=far",
        (10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return color, zi


def blend_photo_depth_colormap(
    bgr: np.ndarray,
    depth_bgr: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float = 0.58,
) -> np.ndarray:
    """Alpha-blend colour depth overlay onto the photo where ``valid_mask`` is true."""
    out = bgr.astype(np.float32)
    d = depth_bgr.astype(np.float32)
    m = valid_mask[..., np.newaxis].astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    blended = out * (1.0 - m * a) + d * (m * a)
    return np.clip(blended, 0, 255).astype(np.uint8)


def estimated_depth_visualization(
    bgr: np.ndarray,
    uv: np.ndarray,
    z_cam_m: np.ndarray,
    *,
    halo_radius: int = 10,
    dense_interp: Literal["linear", "nearest"] = "linear",
    blend_alpha: float = 0.52,
    z_percentile: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Composite image: **sparse per-pixel** depth colours | **dense** interpolated map |
    dense overlay blended on the photo (three panels horizontally).

    Encoding: **red = nearest**, **blue = farthest** (camera-frame depth). Sparse dots use a light halo.
    """
    h, w = bgr.shape[:2]
    lo, hi = depth_colormap_range_m(z_cam_m, z_percentile)
    sparse = render_sparse_depth_pixels(
        h,
        w,
        uv,
        z_cam_m,
        z_lo_m=lo,
        z_hi_m=hi,
        halo_radius=halo_radius,
        background="dark",
    )
    dense_bgr, zi = render_dense_depth_colormap(
        h, w, uv, z_cam_m, z_lo_m=lo, z_hi_m=hi, interp=dense_interp
    )
    valid = np.isfinite(zi) & (zi > 0)
    fused = blend_photo_depth_colormap(bgr, dense_bgr, valid, alpha=blend_alpha)
    panels = [sparse, dense_bgr, fused]
    return np.hstack(panels)


def draw_keypoints(bgr: np.ndarray, pts: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = bgr.copy()
    for p in pts.reshape(-1, 2):
        cv2.circle(out, (int(p[0]), int(p[1])), 3, color, 1, lineType=cv2.LINE_AA)
    return out


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    color=(0, 255, 255),
) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1 : w1 + w2] = img2
    for a, b in zip(pts1, pts2):
        p1 = (int(a[0]), int(a[1]))
        p2 = (int(b[0]) + w1, int(b[1]))
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(out, p1, 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p2, 2, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def draw_inlier_outlier_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1 : w1 + w2] = img2
    m = inlier_mask.reshape(-1).astype(bool)
    for k, (a, b) in enumerate(zip(pts1, pts2)):
        col = (0, 255, 0) if m[k] else (0, 0, 255)
        p1 = (int(a[0]), int(a[1]))
        p2 = (int(b[0]) + w1, int(b[1]))
        cv2.line(out, p1, p2, col, 1, cv2.LINE_AA)
    return out


def draw_epilines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    which: str = "second",
) -> np.ndarray:
    """Draw epilines on img2 induced by pts1 (which='second') or swap if 'first'."""
    lines = None
    if which == "second":
        lines = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        canvas = img2.copy()
        h, w = canvas.shape[:2]
        for a, b, c in lines[: min(len(lines), 80)]:
            x0, x1 = 0, w
            if abs(b) < 1e-6:
                continue
            y0 = int(-(a * x0 + c) / b)
            y1 = int(-(a * x1 + c) / b)
            cv2.line(canvas, (x0, y0), (x1, y1), (255, 128, 0), 1, cv2.LINE_AA)
        side = canvas
    else:
        lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        canvas = img1.copy()
        h, w = canvas.shape[:2]
        for a, b, c in lines[: min(len(lines), 80)]:
            x0, x1 = 0, w
            if abs(b) < 1e-6:
                continue
            y0 = int(-(a * x0 + c) / b)
            y1 = int(-(a * x1 + c) / b)
            cv2.line(canvas, (x0, y0), (x1, y1), (255, 128, 0), 1, cv2.LINE_AA)
        side = canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1 if which == "second" else canvas
    out[:h2, w1 : w1 + w2] = canvas if which == "second" else img2
    return out


def render_depth_histogram_panel(
    depths_before: np.ndarray,
    depths_after: np.ndarray | None,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    db = depths_before[np.isfinite(depths_before) & (depths_before > 0)]
    if db.size == 0:
        cv2.putText(panel, "no depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        return panel
    hist_b, _ = np.histogram(db, bins=40, range=(np.percentile(db, 2), np.percentile(db, 98)))
    hist_b = np.float32(hist_b) / (np.max(hist_b) + 1e-6)
    bw = (width - 80) // 40
    x0 = 40
    for i, h in enumerate(hist_b):
        x = x0 + i * (bw + 2)
        y2 = height - 40
        y1 = int(y2 - h * (height - 80))
        cv2.rectangle(panel, (x, y1), (x + bw, y2), (180, 80, 80), -1)
    cv2.putText(panel, "depth (before scale)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    if depths_after is not None:
        da = depths_after[np.isfinite(depths_after) & (depths_after > 0)]
        if da.size:
            hist_a, _ = np.histogram(da, bins=40, range=(np.percentile(da, 2), np.percentile(da, 98)))
            hist_a = np.float32(hist_a) / (np.max(hist_a) + 1e-6)
            x0 = 40
            for i, h in enumerate(hist_a):
                x = x0 + i * (bw + 2)
                y2 = height // 2 - 20
                y1 = int(y2 - h * (height // 2 - 60))
                cv2.rectangle(panel, (x, y1), (x + bw, y2), (80, 180, 80), -1)
            cv2.putText(
                panel,
                "depth (after scale)",
                (20, height // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
    return panel


def render_trajectory_topdown(
    world_T_camera: list[np.ndarray],
    gt_world_T_camera: list[np.ndarray] | None = None,
    size: int = 512,
    margin: float = 0.1,
    highlight_frame_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    xy = np.array([T[:2, 3] for T in world_T_camera], dtype=np.float64)
    if len(xy) == 0:
        return canvas
    all_xy = [xy]
    if gt_world_T_camera is not None and len(gt_world_T_camera) == len(world_T_camera):
        gxy = np.array([T[:2, 3] for T in gt_world_T_camera], dtype=np.float64)
        all_xy.append(gxy)
    stacked = np.vstack(all_xy)
    mn = np.min(stacked, axis=0)
    mx = np.max(stacked, axis=0)
    span = np.maximum(mx - mn, 1e-6)
    mn = mn - span * margin
    mx = mx + span * margin
    span = mx - mn

    def proj(p: np.ndarray) -> tuple[int, int]:
        x = int((p[0] - mn[0]) / span[0] * (size - 1))
        y = int((1.0 - (p[1] - mn[1]) / span[1]) * (size - 1))
        return x, y

    for i in range(1, len(xy)):
        p0 = proj(xy[i - 1])
        p1 = proj(xy[i])
        cv2.line(canvas, p0, p1, (200, 80, 80), 2, cv2.LINE_AA)
    for i, p in enumerate(xy):
        cv2.circle(canvas, proj(p), 3, (0, 0, 200), -1, cv2.LINE_AA)
    if gt_world_T_camera is not None and len(gt_world_T_camera) == len(world_T_camera):
        gxy = np.array([T[:2, 3] for T in gt_world_T_camera], dtype=np.float64)
        for i in range(1, len(gxy)):
            p0 = proj(gxy[i - 1])
            p1 = proj(gxy[i])
            cv2.line(canvas, p0, p1, (80, 180, 80), 2, cv2.LINE_AA)
    if highlight_frame_indices:
        for hi in highlight_frame_indices:
            if 0 <= hi < len(xy):
                cv2.circle(canvas, proj(xy[hi]), 14, (255, 0, 255), 3, cv2.LINE_AA)
    return canvas


def project_points_topdown(X: np.ndarray, size: int = 512, margin: float = 0.1) -> np.ndarray:
    """X is (N,3) world points; draw XY scatter on white."""
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    if len(X) == 0:
        return canvas
    xy = X[:, :2].astype(np.float64)
    mn = np.min(xy, axis=0)
    mx = np.max(xy, axis=0)
    span = np.maximum(mx - mn, 1e-6)
    mn = mn - span * margin
    mx = mx + span * margin
    span = mx - mn

    def proj(p: np.ndarray) -> tuple[int, int]:
        x = int((p[0] - mn[0]) / span[0] * (size - 1))
        y = int((1.0 - (p[1] - mn[1]) / span[1]) * (size - 1))
        return x, y

    for p in xy:
        cv2.circle(canvas, proj(p), 2, (120, 120, 220), -1, cv2.LINE_AA)
    return canvas


def sparse_depth_error_heatmap(
    bgr: np.ndarray,
    uv: np.ndarray,
    pred_z: np.ndarray,
    gt_z: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Color circles at integer uv by signed relative error (pred-gt)/gt."""
    out = bgr.copy()
    if valid is None:
        valid = np.ones(len(uv), dtype=bool)
    errs = []
    for k in range(len(uv)):
        if not valid[k]:
            continue
        u, v = int(uv[k, 0]), int(uv[k, 1])
        if u < 0 or v < 0 or u >= out.shape[1] or v >= out.shape[0]:
            continue
        gz = float(gt_z[k])
        pz = float(pred_z[k])
        if gz <= 1e-6 or not np.isfinite(gz) or not np.isfinite(pz):
            continue
        e = (pz - gz) / gz
        errs.append(e)
        c = 0.0 if abs(e) > 0.5 else abs(e) / 0.5
        if e >= 0:
            color = (0, int(255 * c), int(255 * (1 - c)))
        else:
            color = (int(255 * c), int(255 * (1 - c)), 0)
        cv2.circle(out, (u, v), 4, color, -1, cv2.LINE_AA)
    return out

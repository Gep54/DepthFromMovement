from __future__ import annotations

from typing import Literal, Sequence

import cv2
import numpy as np
from scipy.interpolate import griddata

from pipeline.geometry import invert_se3
# Smallest |Z| used only to avoid division by zero when projecting (not a cheirality gate).
_PROJ_MIN_ABS_Z = 1e-12

# Sparse halos on the dark panel stay compact; photo overlay uses a larger disk for visibility.
DEFAULT_DEPTH_HALO_RADIUS_SPARSE_PX = 5
DEPTH_HALO_PHOTO_RADIUS_MULTIPLIER = 10
DEFAULT_DEPTH_HALO_RADIUS_PHOTO_PX = max(
    1, int(DEFAULT_DEPTH_HALO_RADIUS_SPARSE_PX * DEPTH_HALO_PHOTO_RADIUS_MULTIPLIER)
)

# Multi-stop BGR colormap: near (small Z) → far (large Z). No saturated red (avoids clash with error cues).
_DEPTH_BGR_STOPS = np.array(
    [
        [0, 255, 255],
        [0, 252, 210],
        [40, 245, 160],
        [90, 228, 110],
        [160, 215, 70],
        [240, 190, 50],
        [255, 140, 90],
        [255, 70, 150],
        [240, 110, 230],
        [210, 165, 255],
    ],
    dtype=np.float64,
)


def depth_to_bgr_colormap(z_m: np.ndarray | float, lo_m: float, hi_m: float) -> np.ndarray:
    """
    Piecewise-linear BGR map: **near** yellow → green → blue → **far** pink (no saturated red).

    Near depth ``lo_m`` uses the first stop; far ``hi_m`` uses the last. Scalar or array ``z_m``.
    Returns uint8 BGR shape ``(..., 3)`` for arrays, or ``(3,)`` for scalars.
    """
    z = np.asarray(z_m, dtype=np.float64)
    scalar_in = z.ndim == 0
    shape_z = z.shape
    zf = z.reshape(-1)
    t = (zf - lo_m) / (hi_m - lo_m + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    stops = _DEPTH_BGR_STOPS
    nseg = stops.shape[0] - 1
    s = t * nseg
    i = np.minimum(np.floor(s).astype(np.int32), nseg - 1)
    f = s - i
    f = np.clip(f, 0.0, 1.0)
    c0 = stops[i]
    c1 = stops[i + 1]
    out = (1.0 - f[:, np.newaxis]) * c0 + f[:, np.newaxis] * c1
    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    if scalar_in:
        return out_u8[0]
    return out_u8.reshape(shape_z + (3,))


def depth_to_bgr_red_near_blue_far(z_m: np.ndarray | float, lo_m: float, hi_m: float) -> np.ndarray:
    """Backward-compatible name; uses :func:`depth_to_bgr_colormap`."""
    return depth_to_bgr_colormap(z_m, lo_m, hi_m)


def _put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float = 0.42,
    color: tuple[int, int, int] = (248, 248, 248),
    outline: tuple[int, int, int] = (22, 22, 22),
    thickness: int = 1,
) -> None:
    x, y = org
    for ox, oy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(
            img,
            text,
            (x + ox, y + oy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            outline,
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_depth_scale_bar_bottom_right(img: np.ndarray, lo_m: float, hi_m: float) -> None:
    """
    Vertical colour strip **bottom-right**: **top** = farthest ``hi_m``, **bottom** = nearest ``lo_m``,
    matching :func:`depth_to_bgr_colormap`. Labels max / mid / min depth in metres to the left of the bar.
    """
    mid_m = (lo_m + hi_m) * 0.5
    h, w = img.shape[:2]
    margin = 10
    bar_w = 88
    bar_h = max(96, int(h * 0.30))
    x2 = w - margin
    x1 = x2 - bar_w
    y2 = h - margin
    y1 = max(margin + 40, y2 - bar_h)
    bar_h = y2 - y1
    for row in range(bar_h):
        frac = row / max(bar_h - 1, 1)
        z_at = hi_m * (1.0 - frac) + lo_m * frac
        pix = depth_to_bgr_colormap(z_at, lo_m, hi_m)
        pr = np.asarray(pix, dtype=np.int32).reshape(-1)
        col = (int(pr[0]), int(pr[1]), int(pr[2]))
        cv2.line(img, (x1, y1 + row), (x2 - 1, y1 + row), col, 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (72, 72, 72), 1, cv2.LINE_AA)
    fs = 0.88
    thick = 8
    labels: list[tuple[str, int]] = [
        (f"{hi_m:.2f} m", y1 - 8),
        (f"{mid_m:.2f} m", y1 + bar_h // 2 + 8),
        (f"{lo_m:.2f} m", y2 + 26),
    ]
    pad = 10
    for lab, yy in labels:
        tw, _th = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0]
        x_lab = x1 - pad - tw
        x_lab = max(margin, x_lab)
        _put_text_outline(img, lab, (x_lab, yy), font_scale=fs, thickness=thick)


def blend_sparse_depth_halos_on_photo(
    bgr: np.ndarray,
    uv: np.ndarray,
    z_cam_m: np.ndarray,
    lo_m: float,
    hi_m: float,
    *,
    halo_radius: int = DEFAULT_DEPTH_HALO_RADIUS_PHOTO_PX,
    peak_alpha: float = 0.62,
) -> np.ndarray:
    """
    Blend depth halos **only** near each sample (disk ``halo_radius``); photo stays untouched elsewhere.
    Uses the same radial tint falloff as :func:`draw_depth_point_halo`.
    """
    out = bgr.astype(np.float64).copy()
    hh, ww = bgr.shape[:2]
    hr = max(0, int(halo_radius))
    mx = max(hr, 1)
    pa = float(np.clip(peak_alpha, 0.0, 1.0))
    for k in range(len(z_cam_m)):
        z = float(z_cam_m[k])
        if not np.isfinite(z):
            continue
        uc = int(round(float(uv[k, 0])))
        vc = int(round(float(uv[k, 1])))
        base = depth_to_bgr_colormap(z, lo_m, hi_m).astype(np.float64).reshape(3)
        v0, v1 = max(0, vc - hr), min(hh, vc + hr + 1)
        u0, u1 = max(0, uc - hr), min(ww, uc + hr + 1)
        for vv in range(v0, v1):
            dy = vv - vc
            for uu in range(u0, u1):
                dx = uu - uc
                dist_sq = dx * dx + dy * dy
                if dist_sq > hr * hr:
                    continue
                dist = float(np.sqrt(dist_sq))
                wf = (dist / mx) ** 1.12
                wf = float(min(0.92, wf))
                col = blend_bgr_toward_white(base, wf).astype(np.float64)
                a = pa * (1.0 - wf / 0.92) if wf > 1e-9 else pa
                a = float(np.clip(a, 0.0, pa))
                out[vv, uu] = out[vv, uu] * (1.0 - a) + col * a
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


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
    base = depth_to_bgr_colormap(z_m, lo_m, hi_m).reshape(3)
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

    Uses triangulation output as-is: only finite 3D coordinates are required.
    ``cheiral_mask`` is accepted for API compatibility but not used to reject points
    (failed cheiral columns are expected to be non-finite in ``X_world_h``).
    No positive-depth or cheirality re-check is applied here.
    Returns pixel coords ``uv`` (N, 2) and camera-frame depths ``z_cam`` (N,) in metres.
    """
    del cheiral_mask  # mask is redundant with NaN storage when cheirality is enabled upstream
    X = X_world_h[:3, :]
    valid = np.all(np.isfinite(X), axis=0)
    if not np.any(valid):
        return np.zeros((0, 2), np.float64), np.zeros(0, np.float64)
    X = X[:, valid]
    Tcw = invert_se3(world_T_camera)
    R = Tcw[:3, :3]
    t = Tcw[:3, 3].reshape(3, 1)
    Xc = R @ X + t
    z = Xc[2, :]
    proj_ok = np.isfinite(z) & (np.abs(z) > _PROJ_MIN_ABS_Z)
    Xc = Xc[:, proj_ok]
    z = z[proj_ok]
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
    z = z_cam_m[np.isfinite(z_cam_m)]
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
        if not np.isfinite(zz_k):
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
    halo_radius: int = DEFAULT_DEPTH_HALO_RADIUS_SPARSE_PX,
    background: Literal["dark", "photo"] = "dark",
    bgr_background: np.ndarray | None = None,
    draw_scale_bar: bool = True,
) -> np.ndarray:
    """
    Colour each depth sample using :func:`depth_to_bgr_colormap` (near yellow → far pink),
    with a soft halo (lighter tints outward). Use ``halo_radius=0`` for a single-pixel dot only.
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
        if not np.isfinite(z):
            continue
        u, v = int(round(float(uv[k, 0]))), int(round(float(uv[k, 1])))
        draw_depth_point_halo(canvas, u, v, z, lo, hi, outer_radius=hr)
    if draw_scale_bar:
        draw_depth_scale_bar_bottom_right(canvas, lo, hi)
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
    draw_scale_bar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate sparse depths to a full raster (linear with nearest-NaN fill), then colour with
    :func:`depth_to_bgr_colormap`.

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
    valid = np.isfinite(zi)
    color = np.full((height, width, 3), 12, dtype=np.uint8)
    if np.any(valid):
        rgb_layer = depth_to_bgr_colormap(zi, lo, hi)
        color = np.where(valid[..., np.newaxis], rgb_layer, color)
    if draw_scale_bar:
        draw_depth_scale_bar_bottom_right(color, lo, hi)
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
    halo_radius: int = DEFAULT_DEPTH_HALO_RADIUS_SPARSE_PX,
    photo_halo_radius: int | None = None,
    show_dense_panel: bool = False,
    dense_interp: Literal["linear", "nearest"] = "linear",
    blend_alpha: float = 0.52,
    halo_peak_alpha: float = 0.62,
    z_percentile: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Composite image: **sparse halos** on dark | optional **dense** interpolated map |
    photo with **local halos only** (no full-frame dense tint unless ``show_dense_panel``).

    ``halo_radius`` applies to the dark sparse panel only; the photo overlay defaults to
    ``halo_radius * DEPTH_HALO_PHOTO_RADIUS_MULTIPLIER`` (override with ``photo_halo_radius``).

    Depth colouring uses :func:`depth_to_bgr_colormap`; each panel draws a bottom-right metre scale.
    """
    h, w = bgr.shape[:2]
    lo, hi = depth_colormap_range_m(z_cam_m, z_percentile)
    r_photo = (
        int(photo_halo_radius)
        if photo_halo_radius is not None
        else max(1, int(halo_radius) * DEPTH_HALO_PHOTO_RADIUS_MULTIPLIER)
    )
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
    photo_halos = blend_sparse_depth_halos_on_photo(
        bgr,
        uv,
        z_cam_m,
        lo,
        hi,
        halo_radius=r_photo,
        peak_alpha=halo_peak_alpha,
    )
    draw_depth_scale_bar_bottom_right(photo_halos, lo, hi)
    if not show_dense_panel:
        return np.hstack([sparse, photo_halos])
    dense_bgr, zi = render_dense_depth_colormap(
        h, w, uv, z_cam_m, z_lo_m=lo, z_hi_m=hi, interp=dense_interp
    )
    valid = np.isfinite(zi)
    fused = blend_photo_depth_colormap(bgr, dense_bgr, valid, alpha=blend_alpha)
    draw_depth_scale_bar_bottom_right(fused, lo, hi)
    return np.hstack([sparse, dense_bgr, fused])


def draw_keypoints(bgr: np.ndarray, pts: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = bgr.copy()
    for p in pts.reshape(-1, 2):
        cv2.circle(out, (int(p[0]), int(p[1])), 3, color, 1, lineType=cv2.LINE_AA)
    return out


def _match_canvas(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, int]:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1 : w1 + w2] = img2
    return out, w1


def grayscale_to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        return gray.copy()
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def canny_edges_bgr(gray: np.ndarray) -> np.ndarray:
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    med = float(np.median(g))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(g, lo, hi)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def draw_rich_keypoints(bgr: np.ndarray, keypoints: list) -> np.ndarray:
    out = bgr.copy()
    if keypoints:
        cv2.drawKeypoints(
            bgr,
            keypoints,
            out,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
    return out


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    color=(0, 255, 255),
) -> np.ndarray:
    out, w1 = _match_canvas(img1, img2)
    for a, b in zip(pts1, pts2):
        p1 = (int(a[0]), int(a[1]))
        p2 = (int(b[0]) + w1, int(b[1]))
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(out, p1, 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p2, 2, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def draw_matches_masked(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    mask: np.ndarray,
    color=(0, 255, 255),
) -> np.ndarray:
    m = mask.reshape(-1).astype(bool)
    if not np.any(m):
        return _match_canvas(img1, img2)[0]
    return draw_matches(img1, img2, pts1[m], pts2[m], color=color)


# BGR colors for match rejection categories (thesis figures).
MATCH_COLOR_EPIPOLAR = (0, 0, 255)
MATCH_COLOR_CHEIRAL = (0, 128, 255)
MATCH_COLOR_INLIER = (0, 255, 0)


def draw_classified_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    *,
    epipolar: np.ndarray,
    cheiral: np.ndarray,
    inlier: np.ndarray,
) -> np.ndarray:
    out, w1 = _match_canvas(img1, img2)
    categories = (
        (epipolar, MATCH_COLOR_EPIPOLAR),
        (cheiral, MATCH_COLOR_CHEIRAL),
        (inlier, MATCH_COLOR_INLIER),
    )
    for mask, col in categories:
        m = mask.reshape(-1).astype(bool)
        for k, (a, b) in enumerate(zip(pts1, pts2)):
            if not m[k]:
                continue
            p1 = (int(a[0]), int(a[1]))
            p2 = (int(b[0]) + w1, int(b[1]))
            cv2.line(out, p1, p2, col, 1, cv2.LINE_AA)
            cv2.circle(out, p1, 2, col, -1, cv2.LINE_AA)
            cv2.circle(out, p2, 2, col, -1, cv2.LINE_AA)
    return out


def _draw_epiline_on_image(canvas: np.ndarray, a: float, b: float, c: float, color, thickness: int = 1) -> None:
    h, w = canvas.shape[:2]
    x0, x1 = 0, w - 1
    if abs(b) < 1e-6:
        return
    y0 = int(-(a * x0 + c) / b)
    y1 = int(-(a * x1 + c) / b)
    cv2.line(canvas, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


# Distinct BGR colours for highlighting up to five correspondence pairs in epipolar PDFs.
EPIPOLAR_HIGHLIGHT_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),
    (0, 200, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 128, 0),
)


def symmetric_epipolar_distances(
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Per-match symmetric point-to-epiline distance in pixels (mean of both images)."""
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if p1.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    d2 = np.abs(lines2[:, 0] * p2[:, 0] + lines2[:, 1] * p2[:, 1] + lines2[:, 2])
    n2 = np.hypot(lines2[:, 0], lines2[:, 1]) + 1e-12
    d1 = np.abs(lines1[:, 0] * p1[:, 0] + lines1[:, 1] * p1[:, 1] + lines1[:, 2])
    n1 = np.hypot(lines1[:, 0], lines1[:, 1]) + 1e-12
    return 0.5 * (d1 / n1 + d2 / n2)


def draw_matches_with_bilateral_epilines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    *,
    match_indices: np.ndarray | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
    default_color: tuple[int, int, int] = (0, 255, 255),
    line_thickness: int = 1,
    point_radius: int = 3,
) -> np.ndarray:
    """
    Side-by-side mosaic with epilines on **both** images and match segments drawn.

    When ``match_indices`` and ``colors`` are set, each listed match uses ``colors[r]``.
  Otherwise every match uses ``default_color``.
    """
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    n = p1.shape[0]
    if n == 0:
        return _match_canvas(img1, img2)[0]

    if match_indices is None:
        idx = np.arange(n, dtype=np.int32)
        use_colors = [default_color] * n
    else:
        idx = np.asarray(match_indices, dtype=np.int32).reshape(-1)
        if colors is None:
            use_colors = [default_color] * len(idx)
        else:
            use_colors = list(colors)

    left = img1.copy()
    right = img2.copy()
    lines2_all = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1_all = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    for rank, k in enumerate(idx):
        col = use_colors[rank % len(use_colors)]
        a, b, c = lines2_all[k]
        _draw_epiline_on_image(right, float(a), float(b), float(c), col, line_thickness)
        a, b, c = lines1_all[k]
        _draw_epiline_on_image(left, float(a), float(b), float(c), col, line_thickness)

    out, w1 = _match_canvas(left, right)
    for rank, k in enumerate(idx):
        col = use_colors[rank % len(use_colors)]
        pt1 = (int(p1[k, 0]), int(p1[k, 1]))
        pt2 = (int(p2[k, 0]) + w1, int(p2[k, 1]))
        cv2.line(out, pt1, pt2, col, line_thickness, cv2.LINE_AA)
        cv2.circle(out, pt1, point_radius, col, -1, cv2.LINE_AA)
        cv2.circle(out, pt2, point_radius, col, -1, cv2.LINE_AA)
    return out


def draw_epipolar_outliers_with_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    line_color=(255, 128, 0),
    match_color=(0, 0, 255),
) -> np.ndarray:
    """Side-by-side: epipolar outliers only, each with its epiline on image 2."""
    outlier = ~inlier_mask.reshape(-1).astype(bool)
    if not np.any(outlier):
        return _match_canvas(img1, img2)[0]
    p1o = pts1[outlier]
    p2o = pts2[outlier]
    lines = cv2.computeCorrespondEpilines(p1o.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    canvas2 = img2.copy()
    for a, b, c in lines:
        _draw_epiline_on_image(canvas2, float(a), float(b), float(c), line_color)
    out, w1 = _match_canvas(img1, canvas2)
    for a, b in zip(p1o, p2o):
        pt1 = (int(a[0]), int(a[1]))
        pt2 = (int(b[0]) + w1, int(b[1]))
        cv2.line(out, pt1, pt2, match_color, 1, cv2.LINE_AA)
        cv2.circle(out, pt1, 3, match_color, -1, cv2.LINE_AA)
        cv2.circle(out, pt2, 3, match_color, -1, cv2.LINE_AA)
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

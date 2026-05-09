from __future__ import annotations

from pathlib import Path

import numpy as np

from data.schema import GTPoseRow


def quat_wxyz_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        raise ValueError("invalid quaternion (zero norm)")
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def load_tum_poses(path: Path) -> list[GTPoseRow]:
    rows: list[GTPoseRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append(
                GTPoseRow(
                    timestamp=float(parts[0]),
                    tx=float(parts[1]),
                    ty=float(parts[2]),
                    tz=float(parts[3]),
                    qx=float(parts[4]),
                    qy=float(parts[5]),
                    qz=float(parts[6]),
                    qw=float(parts[7]),
                )
            )
    return rows


def tum_rows_to_world_T_camera(rows: list[GTPoseRow]) -> list[np.ndarray]:
    """TUM is typically world position + orientation of the sensor; build world_T_camera (4x4)."""
    out: list[np.ndarray] = []
    for r in rows:
        R = quat_wxyz_to_R(r.qw, r.qx, r.qy, r.qz)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [r.tx, r.ty, r.tz]
        out.append(T)
    return out


def load_gt_depth(path: Path) -> np.ndarray | None:
    """Load single-channel depth map (EXR, PNG 16-bit, or any OpenCV-readable float)."""
    import cv2

    if not path.is_file():
        return None
    if path.suffix.lower() == ".exr":
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)

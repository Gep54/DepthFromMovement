from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

PoseConvention = Literal["world_T_camera", "camera_T_world"]
MotionRepresentation = Literal["absolute", "relative_to_prev"]


@dataclass
class Calibration:
    """Camera intrinsics and optional distortion."""

    K: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray | None = None  # (n,) OpenCV ordering
    image_size: tuple[int, int] | None = None  # (width, height)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError("K must be 3x3")
        if self.dist_coeffs is not None:
            self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)


@dataclass
class MotionSpec:
    """Per-frame rigid transforms for the sequence."""

    pose_convention: PoseConvention
    representation: MotionRepresentation
    """absolute: T[i] is pose of frame i. relative_to_prev: T[i] maps frame i-1 -> i (T[0] often identity)."""
    transforms: list[np.ndarray]
    """Each (4, 4) homogeneous SE(3), ``world_T_camera``: maps camera→world, ``X_w = R X_c + t``."""

    def __post_init__(self) -> None:
        cleaned: list[np.ndarray] = []
        for i, T in enumerate(self.transforms):
            Ta = np.asarray(T, dtype=np.float64)
            if Ta.shape == (3, 4):
                H = np.eye(4, dtype=np.float64)
                H[:3, :4] = Ta
                Ta = H
            if Ta.shape != (4, 4):
                raise ValueError(f"Transform {i} must be 4x4 or 3x4, got {Ta.shape}")
            cleaned.append(Ta)
        self.transforms = cleaned


@dataclass
class GTPoseRow:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class DatasetPaths:
    root: Path
    images_dir: Path
    calibration_file: Path
    motion_file: Path
    features_file: Path
    gt_depth_dir: Path | None = None
    gt_poses_file: Path | None = None
    image_glob: str = "*.png"


def default_dataset_paths(root: str | Path) -> DatasetPaths:
    r = Path(root)
    return DatasetPaths(
        root=r,
        images_dir=r / "images",
        calibration_file=r / "calibration.json",
        motion_file=r / "motion.json",
        features_file=r / "features.json",
        gt_depth_dir=r / "gt_depth" if (r / "gt_depth").is_dir() else None,
        gt_poses_file=r / "gt_poses.txt" if (r / "gt_poses.txt").is_file() else None,
    )


def validate_calibration(cal: Calibration, image_paths: Sequence[Path]) -> None:
    if cal.image_size is not None and image_paths:
        import cv2

        im = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
        if im is not None:
            h, w = im.shape[:2]
            ew, eh = cal.image_size
            if (w, h) != (ew, eh):
                raise ValueError(
                    f"calibration image_size {cal.image_size} does not match first image {(w, h)}"
                )


def validate_motion_vs_images(motion: MotionSpec, n_images: int) -> None:
    if len(motion.transforms) != n_images:
        raise ValueError(
            f"motion.json: expected {n_images} transforms (one per image), got {len(motion.transforms)}"
        )


def validate_dataset_consistency(
    cal: Calibration,
    motion: MotionSpec,
    image_paths: Sequence[Path],
) -> None:
    validate_calibration(cal, image_paths)
    validate_motion_vs_images(motion, len(image_paths))

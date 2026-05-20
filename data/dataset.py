from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from data.feature_matching_json import load_feature_matching_json
from data.fusion_json import default_fusion_config, load_fusion_json
from data.gt_io import load_gt_depth, load_tum_poses, tum_rows_to_world_T_camera
from data.io_json import load_calibration_json, load_motion_json, world_T_camera_from_motion
from data.motion_frames import apply_motion_frame_conversion
from pipeline.config import FeatureConfig
from pipeline.metric_fusion import fuse_pose_sequence
from data.schema import (
    Calibration,
    DatasetPaths,
    MotionSpec,
    default_dataset_paths,
    validate_dataset_consistency,
    validate_motion_vs_images,
)


@dataclass
class Dataset:
    paths: DatasetPaths
    image_paths: list[Path]
    calibration: Calibration
    motion: MotionSpec
    world_T_camera: list[np.ndarray]
    """Per-frame ``world_T_camera`` in the metric world frame (e.g. ``gps_baro_origin`` from ``motion.json``)."""
    world_frame: str | None = None
    """World / odometry parent frame label from ``motion.json`` when available."""
    gt_depth_paths: list[Path | None] = field(default_factory=list)
    """Aligned with image_paths; None if missing."""
    gt_world_T_camera: list[np.ndarray] | None = None
    """Optional ground-truth poses in the same absolute world as ``world_T_camera`` when provided."""
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    """Feature detector + matcher settings (from optional ``features.json``)."""

    def __post_init__(self) -> None:
        if not self.gt_depth_paths:
            self.gt_depth_paths = [None] * len(self.image_paths)
        if len(self.gt_depth_paths) != len(self.image_paths):
            raise ValueError("gt_depth_paths length must match images")
        if self.gt_world_T_camera is not None and len(self.gt_world_T_camera) != len(self.image_paths):
            raise ValueError("gt_world_T_camera length must match images when provided")


def _sorted_images(images_dir: Path, glob_pat: str) -> list[Path]:
    paths = sorted(images_dir.glob(glob_pat))
    jpgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    if not paths and jpgs:
        return sorted(set(jpgs))
    return paths


def _align_gt_depth_names(images_dir: Path, depth_dir: Path, image_paths: Sequence[Path]) -> list[Path | None]:
    out: list[Path | None] = []
    for ip in image_paths:
        stem = ip.stem
        for ext in (".exr", ".png", ".tif", ".tiff"):
            cand = depth_dir / f"{stem}{ext}"
            if cand.is_file():
                out.append(cand)
                break
        else:
            out.append(None)
    return out


def load_dataset(
    root: str | Path,
    *,
    paths: DatasetPaths | None = None,
    image_glob: str | None = None,
    validate: bool = True,
) -> Dataset:
    p = paths or default_dataset_paths(root)
    if image_glob:
        p = DatasetPaths(
            root=p.root,
            images_dir=p.images_dir,
            calibration_file=p.calibration_file,
            motion_file=p.motion_file,
            features_file=p.features_file,
            gt_depth_dir=p.gt_depth_dir,
            gt_poses_file=p.gt_poses_file,
            image_glob=image_glob,
            provided_motion_file=p.provided_motion_file,
            fusion_file=p.fusion_file,
        )
    root_p = Path(p.root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root_p}")

    image_roots: list[Path] = []
    if p.images_dir.is_dir():
        image_roots.append(p.images_dir)
    if root_p.resolve() not in {r.resolve() for r in image_roots}:
        image_roots.append(root_p)

    image_paths: list[Path] = []
    for img_root in image_roots:
        image_paths = _sorted_images(img_root, p.image_glob)
        if image_paths:
            break
    if not image_paths:
        searched = ", ".join(str(r) for r in image_roots)
        raise FileNotFoundError(f"no images matching {p.image_glob!r} under {searched}")

    if not p.calibration_file.is_file():
        raise FileNotFoundError(f"calibration.json not found: {p.calibration_file}")
    if not p.motion_file.is_file():
        raise FileNotFoundError(f"motion.json not found: {p.motion_file}")

    calibration = load_calibration_json(p.calibration_file)
    motion = load_motion_json(p.motion_file)
    if validate:
        validate_dataset_consistency(calibration, motion, image_paths)

    world_T_camera = world_T_camera_from_motion(motion)
    world_T_camera = apply_motion_frame_conversion(world_T_camera, motion, root_p)

    fusion_cfg = default_fusion_config()
    if p.fusion_file is not None and p.fusion_file.is_file():
        fusion_cfg = load_fusion_json(p.fusion_file)

    prov_path = p.provided_motion_file
    if prov_path is not None and prov_path.is_file():
        motion_provided = load_motion_json(prov_path)
        if validate:
            validate_motion_vs_images(motion_provided, len(image_paths))
        wt_provided = world_T_camera_from_motion(motion_provided)
        world_T_camera = fuse_pose_sequence(
            world_T_camera,
            wt_provided,
            str(fusion_cfg["method"]),
            position_blend_weight=float(fusion_cfg["position_blend_weight"]),
        )

    feature_config = FeatureConfig()
    if p.features_file.is_file():
        feature_config = load_feature_matching_json(p.features_file)

    gt_depth_paths: list[Path | None] = [None] * len(image_paths)
    if p.gt_depth_dir and p.gt_depth_dir.is_dir():
        gt_depth_paths = _align_gt_depth_names(p.images_dir, p.gt_depth_dir, image_paths)

    gt_poses: list[np.ndarray] | None = None
    if p.gt_poses_file and p.gt_poses_file.is_file():
        rows = load_tum_poses(p.gt_poses_file)
        gt_poses = tum_rows_to_world_T_camera(rows)
        if len(gt_poses) != len(image_paths):
            raise ValueError(
                f"gt_poses.txt has {len(gt_poses)} rows but found {len(image_paths)} images; counts must match"
            )

    world_frame = motion.target_world_frame or motion.world_frame
    world_T_camera = [np.asarray(W, dtype=np.float64).copy() for W in world_T_camera]

    return Dataset(
        paths=p,
        image_paths=image_paths,
        calibration=calibration,
        motion=motion,
        world_T_camera=world_T_camera,
        world_frame=world_frame,
        gt_depth_paths=gt_depth_paths,
        gt_world_T_camera=gt_poses,
        feature_config=feature_config,
    )


def read_image_bgr(path: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return im


def load_gt_depth_for_frame(ds: Dataset, index: int) -> np.ndarray | None:
    if index < 0 or index >= len(ds.gt_depth_paths):
        return None
    dp = ds.gt_depth_paths[index]
    if dp is None:
        return None
    return load_gt_depth(dp)

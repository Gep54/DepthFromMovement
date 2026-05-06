"""
Generate TestData/motion.json with one absolute world_T_camera pose per image.

Uses a synthetic forward translation along +X (small baseline between frames) so the
two-view pipeline is not degenerate when you do not yet have real odometry.

Usage (from repo root):
  python scripts/init_test_dataset_motion.py TestData
  python scripts/init_test_dataset_motion.py TestData --step 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _list_images(images_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    out: list[Path] = []
    for pat in exts:
        out.extend(images_dir.glob(pat))
    return sorted({p.resolve() for p in out}, key=lambda p: p.name.lower())


def _image_roots(dataset_root: Path) -> list[Path]:
    roots = []
    sub = dataset_root / "images"
    if sub.is_dir():
        roots.append(sub)
    if dataset_root.resolve() not in {r.resolve() for r in roots}:
        roots.append(dataset_root)
    return roots


def main() -> None:
    ap = argparse.ArgumentParser(description="Write motion.json aligned with images/ count.")
    ap.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path("TestData"),
        help="Folder containing images/ (default: TestData)",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Translation along world X between consecutive frames (default 0.02)",
    )
    ap.add_argument(
        "--pose-convention",
        choices=("world_T_camera", "camera_T_world"),
        default="world_T_camera",
    )
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    imgs: list[Path] = []
    for cand in _image_roots(root):
        imgs = _list_images(cand)
        if imgs:
            break
    if not imgs:
        raise SystemExit(
            f"no images found under {root / 'images'} or {root} (png/jpg/jpeg/webp)"
        )

    frames: list[dict] = []
    for i in range(len(imgs)):
        T = np.eye(4, dtype=float)
        T[0, 3] = args.step * float(i)
        frames.append({"index": i, "T": T.tolist()})

    motion = {
        "pose_convention": args.pose_convention,
        "representation": "absolute",
        "frames": frames,
    }
    out_path = root / "motion.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(motion, f, indent=2)
        f.write("\n")

    print(f"wrote {out_path} with {len(frames)} poses for {len(imgs)} images")


if __name__ == "__main__":
    main()

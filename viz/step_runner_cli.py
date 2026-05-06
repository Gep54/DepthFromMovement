from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import load_dataset
from pipeline.config import MotionMode
from viz.step_runner import (
    ensure_all_step_pngs_exist,
    ensure_sequence_outputs_exist,
    export_all_stages,
    export_sequence_consecutive_pairs,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Export pipeline step PNGs for a dataset folder.")
    p.add_argument("dataset_root", type=Path, help="Folder with images/, calibration.json, motion.json")
    p.add_argument("--run-dir", type=Path, default=Path("runs") / "export", help="Output run directory")
    p.add_argument(
        "--sequence",
        action="store_true",
        help="Export all consecutive pairs under run_dir/pairs/, fused landmarks in summary/",
    )
    p.add_argument(
        "--fuse-merge-px",
        type=float,
        default=4.0,
        help="Pixel radius for landmark fusion when --sequence (default 4)",
    )
    p.add_argument("--i", type=int, default=0, help="First frame index (single-pair mode only)")
    p.add_argument("--j", type=int, default=1, help="Second frame index (single-pair mode only)")
    p.add_argument(
        "--motion-mode",
        choices=("known_pose", "estimate_essential"),
        default="known_pose",
    )
    args = p.parse_args()
    ds = load_dataset(args.dataset_root)
    mm: MotionMode = args.motion_mode  # type: ignore[assignment]
    if args.sequence:
        export_sequence_consecutive_pairs(
            ds,
            args.run_dir,
            motion_mode=mm,
            fuse_merge_px=args.fuse_merge_px,
        )
        ensure_sequence_outputs_exist(args.run_dir, len(ds.image_paths))
    else:
        export_all_stages(ds, args.run_dir, i=args.i, j=args.j, motion_mode=mm)
        ensure_all_step_pngs_exist(args.run_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import load_dataset
from pipeline.map import MapConfig
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
        help="Export multi-baseline pairs under run_dir/pairs/, fused landmarks in summary/",
    )
    p.add_argument(
        "--fuse-merge-px",
        type=float,
        default=4.0,
        help="Pixel radius for landmark fusion when --sequence (default 4)",
    )
    p.add_argument(
        "--pair-lookback",
        type=int,
        default=10,
        help="With --sequence: pair each frame j with j-1..j-W (default 10). Use 1 for consecutive-only.",
    )
    p.add_argument("--i", type=int, default=0, help="First frame index (single-pair mode only)")
    p.add_argument("--j", type=int, default=1, help="Second frame index (single-pair mode only)")
    p.add_argument(
        "--no-geometry-stages",
        action="store_true",
        help="Skip triangulation / estimated depth / depth error under steps/geometry/",
    )
    p.add_argument(
        "--reproj-outlier-px",
        type=float,
        default=3.0,
        help="Reprojection threshold (px) for match rejection colouring (default 3)",
    )
    p.add_argument(
        "--rejection-audit",
        type=Path,
        default=None,
        help="Path for rejection_audit.jsonl (default: <run-dir>/rejection_audit.jsonl)",
    )
    p.add_argument(
        "--triangulation-motion-source",
        choices=("vision_scale", "odometry_pose"),
        default="vision_scale",
        help="How odometry enters two-view triangulation (default vision_scale)",
    )
    p.add_argument(
        "--max-range-baseline-factor",
        type=float,
        default=0.0,
        help="Drop triangulated points beyond factor * max(baseline_m, 1e-3) in cam1 (<=0 disables)",
    )
    args = p.parse_args()
    ds = load_dataset(args.dataset_root)
    map_cfg = MapConfig(
        triangulation_motion_source=args.triangulation_motion_source,
        max_range_baseline_factor=args.max_range_baseline_factor,
    )
    include_geometry = not args.no_geometry_stages
    audit_path = args.rejection_audit
    if args.sequence:
        export_sequence_consecutive_pairs(
            ds,
            args.run_dir,
            fuse_merge_px=args.fuse_merge_px,
            pair_lookback=args.pair_lookback,
            map_cfg=map_cfg,
            include_geometry=include_geometry,
            reproj_thresh_px=args.reproj_outlier_px,
            rejection_audit_path=audit_path,
        )
        ensure_sequence_outputs_exist(
            args.run_dir,
            len(ds.image_paths),
            pair_lookback=args.pair_lookback,
            include_geometry=include_geometry,
        )
    else:
        export_all_stages(
            ds,
            args.run_dir,
            i=args.i,
            j=args.j,
            map_cfg=map_cfg,
            include_geometry=include_geometry,
            reproj_thresh_px=args.reproj_outlier_px,
            rejection_audit_path=audit_path,
        )
        ensure_all_step_pngs_exist(args.run_dir, include_geometry=include_geometry)


if __name__ == "__main__":
    main()

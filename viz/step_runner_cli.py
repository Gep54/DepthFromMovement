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
        help="Export multi-baseline pairs under run_dir/pairs/",
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
        help="Skip estimated depth under steps/geometry/ (default writes geometry/01_estimated_depth.png only)",
    )
    p.add_argument(
        "--detail-log",
        action="store_true",
        help="Per pair: add pair/rejected_epipolar and rejected_cheiral PNGs",
    )
    p.add_argument(
        "--full-steps",
        action="store_true",
        help="Export all legacy step PNGs (single/, full pair/, triangulation, depth error)",
    )
    p.add_argument(
        "--epipolar",
        action="store_true",
        help="Per pair: steps/epipolar/ PNGs (all matches, best 5 stacked, worst 5 stacked)",
    )
    p.add_argument(
        "--no-cheiral",
        action="store_true",
        help="Disable cheirality rejection; keep all epipolar inliers (default Z threshold is -0.01 m)",
    )
    p.add_argument(
        "--epipolar-thresh",
        type=float,
        default=3.0,
        metavar="PX",
        help="RANSAC epipolar distance threshold in pixels (default 3.0; larger = looser)",
    )
    p.add_argument(
        "--cheiral-min-z",
        type=float,
        default=-0.01,
        metavar="M",
        help="Cheirality: min camera-frame Z in metres, both views (default -0.01; lower = looser)",
    )
    p.add_argument(
        "--rejection-audit",
        type=Path,
        default=None,
        help="Path for rejection_audit.jsonl (default: <run-dir>/rejection_audit.jsonl)",
    )
    args = p.parse_args()
    ds = load_dataset(args.dataset_root)
    include_geometry = not args.no_geometry_stages
    map_cfg = MapConfig(
        check_cheiral=not args.no_cheiral,
        ransac_epipolar_thresh=float(args.epipolar_thresh),
        cheiral_min_z=float(args.cheiral_min_z),
    )
    audit_path = args.rejection_audit
    full_steps = args.full_steps
    detail_log = args.detail_log
    export_epipolar = args.epipolar
    if args.sequence:
        export_sequence_consecutive_pairs(
            ds,
            args.run_dir,
            pair_lookback=args.pair_lookback,
            include_geometry=include_geometry,
            rejection_audit_path=audit_path,
            map_cfg=map_cfg,
            full_steps=full_steps,
            detail_log=detail_log,
            export_epipolar=export_epipolar,
        )
        ensure_sequence_outputs_exist(
            args.run_dir,
            len(ds.image_paths),
            pair_lookback=args.pair_lookback,
            include_geometry=include_geometry,
            full_steps=full_steps,
            detail_log=detail_log,
        )
    else:
        export_all_stages(
            ds,
            args.run_dir,
            i=args.i,
            j=args.j,
            include_geometry=include_geometry,
            rejection_audit_path=audit_path,
            map_cfg=map_cfg,
            full_steps=full_steps,
            detail_log=detail_log,
            export_epipolar=export_epipolar,
        )
        ensure_all_step_pngs_exist(
            args.run_dir,
            include_geometry=include_geometry,
            full_steps=full_steps,
            detail_log=detail_log,
        )


if __name__ == "__main__":
    main()

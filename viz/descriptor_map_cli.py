from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from data.dataset import load_dataset
from data.descriptor_map_json import load_descriptor_map_json
from pipeline.descriptor_landmark_map import export_landmarks_csv
from viz.descriptor_map_runner import run_descriptor_landmark_pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build descriptor-based sparse landmark map (camera-0 frame) with multi-baseline pairing.",
    )
    p.add_argument("dataset_root", type=Path, help="Folder with calibration.json, motion.json, images/")
    p.add_argument("--run-dir", type=Path, default=Path("runs") / "descriptor_map", help="Output directory")
    p.add_argument("--pair-lookback", type=int, default=10)
    p.add_argument(
        "--merge-beta",
        type=float,
        default=None,
        help="Fixed EMA merge weight in (0,1]; omit to use dataset/config default or mean-equivalent mode",
    )
    p.add_argument(
        "--descriptor-max-dist",
        type=float,
        default=None,
        help="Override max Hamming (ORB) or L2 (SIFT) for associating observations to landmarks",
    )
    p.add_argument(
        "--spatial-merge-radius-m",
        type=float,
        default=None,
        help="3D merge gate radius in cam0 (m); omit to use descriptor_map.json or disable",
    )
    p.add_argument(
        "--descriptor-map-config",
        type=Path,
        default=None,
        help="JSON config path (default: <dataset>/descriptor_map.json if present)",
    )
    p.add_argument("--export-csv", type=Path, default=None, help="Write landmarks CSV to this path")
    p.add_argument("--iter-viz", action="store_true", help="Save PNG snapshot after each processed pair")
    args = p.parse_args()

    ds = load_dataset(args.dataset_root)

    cfg_path = args.descriptor_map_config
    if cfg_path is None:
        cfg_path = Path(args.dataset_root) / "descriptor_map.json"
    desc_cfg = load_descriptor_map_json(Path(cfg_path), ds.feature_config.method)

    if args.merge_beta is not None:
        desc_cfg = replace(desc_cfg, merge_beta=float(args.merge_beta))
    if args.descriptor_max_dist is not None:
        desc_cfg = replace(desc_cfg, max_match_distance=float(args.descriptor_max_dist))
    if args.spatial_merge_radius_m is not None:
        desc_cfg = replace(desc_cfg, spatial_merge_radius_m=float(args.spatial_merge_radius_m))

    desc_map = run_descriptor_landmark_pipeline(
        ds,
        args.run_dir,
        pair_lookback=args.pair_lookback,
        desc_cfg=desc_cfg,
        save_iter_viz=args.iter_viz,
    )

    if args.export_csv is not None:
        export_landmarks_csv(args.export_csv, desc_map)


if __name__ == "__main__":
    main()

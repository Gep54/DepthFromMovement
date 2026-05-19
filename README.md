# DepthFromMovement

**Sparse 3D mapping** from a **monocular camera** plus **metric odometry**. The primary deployment is a **ROS~2 node** that selects keyframes, runs the same two-view triangulation core as the offline tools, and writes run artefacts. **`dfm-export-steps`** replays image folders with **`motion.json`** for figures and batch experiments.

---

## What this codebase does

1. **Load a dataset** — Sorted RGB frames plus **`calibration.json`** (intrinsics) and **`motion.json`** (poses from the drone odometry track). Optional GT depth / poses under documented paths (`data/`).

2. **Two-view pipeline** (ORB/SIFT features → matching → RANSAC on the essential constraint → `recoverPose` → **vision** relative rotation and translation **direction** → scale translation to **odometry** \(\|\mathbf{t}\|\) between the two frames → triangulation in a **camera-0–centric** world frame: **`load_dataset`** left-multiplies every `world_T_camera` (and optional GT poses) by \(\mathrm{inv}(W_0)\) so the first pose is identity; pairwise relative motion is unchanged, so two-view scaling matches the original track). Intrinsics-distorted images may still be **undistorted internally** before detection; **`world_T_camera`** in memory is this canonical map frame (simulation or odometry “world” offsets appear only in raw logs, e.g. ROS **`position.json`** `position` fields).

3. **Exports** — Illustration PNGs under **`steps/single/`** (original, grayscale, Canny edges, rich keypoints) and **`steps/pair/`** (raw mosaic, all matches, epipolar outliers with epilines, multi-colour rejection panel, inliers only). Optional **`steps/geometry/`**: triangulation mosaic, sparse estimated depth, GT depth error. Match colours: **green** = full inlier, **red** = epipolar RANSAC outlier, **orange** = cheiral fail, **magenta** = high reprojection. **`rejection_audit.jsonl`** logs per-pair counts; **`summary/pairs_all_rejection_types.json`** lists pairs where every rejection type appears at least once. Depth colouring: **yellow = near → green → blue → pink = far** (no saturated red, to stay distinct from error highlights).

4. **Sequence mode** — For each frame index \(j\), pairs **\((j-1,j),\,(j-2,j),\,\ldots\)** up to **`pair_lookback`** earlier frames **(default \(10\))**---a breaking change versus older versions that exported only consecutive pairs **`(k,k+1)`**. Use **`--pair-lookback 1`** to restore consecutive-only pairing. Outputs live under **`runs/.../pairs/iii_jjj/steps/`**; landmarks are **fused** across edges (**`fuse-merge-px`**); **`summary/`** contains full trajectory plus fused top-down and fused sparse depth on reference frame 0. **Descriptors are computed once per frame** (`compute_frame_features_cache`) and reused for every pair and for the keypoints figure—single-pair export still detects per call unless you pass a cache from Python.

---

## Dataset layout

Point the CLI at a folder that contains:

| Path | Purpose |
|------|---------|
| **`calibration.json`** | `K` (3×3), optional `dist_coeffs`, optional `image_size` |
| **`motion.json`** | Per-frame **camera→world** pose (`world_T_camera`): \(\mathbf{X}_w = \mathbf{R}\mathbf{X}_c + \mathbf{t}\). Fields: `pose_convention`, `representation` (`absolute` or `relative_to_prev`), `frames[]` with `T` (4×4 or 3×4); optional per-frame **`filename`** (basename in **`images/`**, used by the ROS exporter). **`load_dataset`** rewrites the loaded trajectory so frame **0** is the map origin (same convention; GT poses transformed consistently). |
| **`images/`** | Frames (`*.png` by default); if empty, the loader also searches image files in the dataset root |

Optional: **`features.json`** — detector/matcher settings (ORB vs SIFT, counts, Lowe ratio, cross-check, ORB pyramid / SIFT contrast); omitted keys use library defaults (same as **`FeatureConfig`**). **`descriptor_map.json`** (optional) — knobs for **`dfm-descriptor-map`**: **`merge_beta`** (`null` = incremental arithmetic mean via \(1/(n+1)\)), **`max_match_distance`**, **`ratio_second_best`**, **`spatial_merge_radius_m`** (3D merge gate in world frame; ROS live node uses **`keyframe_distance_m`**). **`gt_depth/`** (depth maps named like image stems), **`gt_poses.txt`** (TUM-style; length must match image count).

Helper (optional): **`scripts/init_test_dataset_motion.py`** — builds `motion.json` from image count when you only need a synthetic baseline. The ROS node can emit the same layout from a live run or rosbag via **`export_offline_dataset`** (see **ROS 2 package** below).

---

## Installation

```bash
pip install -r requirements.txt
# editable install (recommended) — exposes console script dfm-export-steps
pip install -e .
```

Development tests:

```bash
pip install -e ".[dev]"
pytest tests
```

### ROS 2 package (live mapping)

The **`ros2_ws/`** colcon workspace contains **`incremental_vo_ros2`**: a node that subscribes to a monocular image, **`sensor_msgs/CameraInfo`** (default **`/uav1/stereo/left/camera_info`**) for intrinsics, and odometry. The node selects **keyframes** when odometry translation has moved at least **`keyframe_distance_m`** (default **0.5** m), runs **`pipeline.IncrementalMap`** over each new keyframe against up to **`pair_lookback`** prior keyframes for triangulation + reprojection diagnostics, then feeds every successful **`TwoViewResult`** into **`pipeline.DescriptorLandmarkMap`** (same class as **`dfm-descriptor-map`**): after cheirality, each point is transformed **camera → drone (odom child) → world (odom parent)** and stored in that **odom world frame**. Landmarks are associated by **nearest-neighbour descriptor distance**, with a **3D merge gate** at radius **`keyframe_distance_m`** (world-frame `‖ΔX‖`), positions updated with **EMA** (default **`1/(n+1)`**), and **prototype descriptors are replace-if-better**. Poses passed into **`IncrementalMap`** stay **canonical** for two-view geometry; the sparse map and live **`PointCloud2`** use **raw odom world** coordinates. **`position.json`** logs raw metric keyframe poses. On rosbag replay (sim time jumps backward), the map is **cleared** and a **new** **`run_<timestamp>/`** folder is allocated (**`bag_replay_reset_enabled`**, default **true**). Optional **offline dataset export** (`**export_offline_dataset:=true**`) writes a **`load_dataset`**-ready folder under **`output_root/offline_datasets/run_<timestamp>/`** (or **`offline_dataset_root`** if set): **`images/frame_NNNNN.png`**, **`motion.json`** (odometry **`world_T_camera`** with matching **`filename`** per frame), and **`calibration.json`** on shutdown. Use with **`dfm-export-steps <that_folder> --sequence`**. Independent of **`save_run_on_shutdown`** (VO debug artefacts under **`ros2_runs/`**).

On shutdown the node writes **`output_root/ros2_runs/run_<timestamp>/`** when **`save_run_on_shutdown:=true`**: **`images/`**, **`calibration.json`** (latched **`K`**, optional **`dist_coeffs`**, **`image_size`** — same schema as offline datasets), **`position.json`** (keyframe records, effective feature/descriptor knobs, **`landmarks_reference_frame`** / **`map_coordinate_frame`** = odom **`header.frame_id`**, and optional flat **`eval_world_T_camera0`**), **`sparse_map.npz`** (descriptor-merged points in **odom parent world**), optional **`sparse_map_eval_world.npz`** when **`eval_world_T_camera0`** is set (maps odom-world points into the evaluation frame via **`eval @ inv(W0_raw)`**), and **`descriptor_landmarks.csv`** (id, world XYZ, **`n_updates`**, descriptor hex). The process must import **`pipeline.*`** from this repository (the node walks parents until **`pipeline/map.py`**).

**Build (from `ros2_ws/`, with your ROS 2 distro already on `PATH`):**

```bash
colcon build --packages-select incremental_vo_ros2
```

On **Linux** or **macOS** you may add **`--symlink-install`** so Python sources stay linked into the build tree (faster iteration). On **Windows**, skip that flag unless **Developer Mode** is on (Settings → *Privacy & security* → *For developers*): otherwise `colcon` can fail with **WinError 1314** (“client does not hold a required privilege”) when creating symlinks. If a build failed partway, delete `build/incremental_vo_ros2` and `install/incremental_vo_ros2` (or the whole `build/` / `install/` trees) and rebuild before sourcing `install` again.

**Overlay the workspace** (each new terminal):

- **Windows (Command Prompt):** from `ros2_ws`, run `install\setup.bat`.
- **Windows (PowerShell):** from `ros2_ws`, run `.\install\setup.ps1` so the environment applies to the current session (`setup.bat` alone does not persist variables into an already-open PowerShell window).
- **Linux / WSL:** `source install/setup.bash`

When sourcing the overlay, you may see **RTI Connext DDS** warnings about `rtisetenv_x64Win64VS2017.bat` missing. That only means the optional RTI vendor stack is not installed; **ROS 2 still works** with the default middleware (e.g. **Fast DDS**). You can ignore the message unless you explicitly use RTI Connext.

**Run the node:**

```bash
ros2 run incremental_vo_ros2 incremental_vo_node
```

### Parameter precedence

Node parameters resolve in this order (highest wins last):

1. **Code defaults** — values in `declare_parameter(...)` in `incremental_vo_node.py`
2. **`configuration.env`** — dotenv-style `name=value` file (see repo-root [`configuration.env`](configuration.env) and [`ros2_ws/configuration.env.example`](ros2_ws/configuration.env.example))
3. **CLI** — `ros2 run … --ros-args -p name:=value` (or launch-file equivalents)

Config file discovery (first match): `--config-file PATH`, environment variable **`INCREMENTAL_VO_CONFIG`**, then **`./configuration.env`** in the process working directory. When using [`ros2_ws/run_incremental_vo.ps1`](ros2_ws/run_incremental_vo.ps1), pass **`-ConfigFile ..\configuration.env`** to point at the repo-root file.

```bash
# File only (uses ./configuration.env if present in cwd)
ros2 run incremental_vo_ros2 incremental_vo_node

# Explicit config path
ros2 run incremental_vo_ros2 incremental_vo_node --config-file /path/to/configuration.env

# CLI overrides file
ros2 run incremental_vo_ros2 incremental_vo_node --config-file configuration.env --ros-args -p keyframe_distance_m:=0.2
```

Useful **`--ros-args`** parameters include **`-p use_sim_time:=true`** when playing a rosbag with **`ros2 bag play … --clock`**, **`-p output_root:=…`**, **offline dataset export** — **`-p export_offline_dataset:=true`** (writes **`output_root/offline_datasets/run_<timestamp>/`** unless **`-p offline_dataset_root:=/path`**), **`images/frame_NNNNN.png`**, **`motion.json`** with matching **`filename`** fields, **`calibration.json`** on exit; **`-p offline_dataset_image_prefix:=frame`** — **`-p keyframe_distance_m:=0.5`** (keyframe spacing and descriptor **3D merge** radius in cam0), **`-p keyframe_buffer_start_fraction:=0.8`** (only buffer raw images once odometry travel reaches this fraction of **`keyframe_distance_m`**; decode/undistort runs once when a frame is committed as a keyframe), **`-p pair_lookback:=10`**, topic overrides (**`-p image_topic:=…`**, **`-p camera_info_topic:=/uav1/stereo/left/camera_info`**, **`-p odom_main_topic:=…`**). For **`ros2 bag play`**, set **`-p camera_info_qos_durability:=volatile`** (or use [`configuration.race_rosbag.env`](configuration.race_rosbag.env)) so the CameraInfo subscription matches rosbag2’s VOLATILE replay QoS; live cameras typically keep the default **`transient_local`**. Intrinsics come from **`CameraInfo`** by default (**`require_camera_info:=true`**); images are skipped until the first message is latched. Non-zero plumb_bob distortion is undistorted before feature detection (effective **`K`** from **`cv2.getOptimalNewCameraMatrix`**). Legacy **`camera_fx`**, **`camera_fy`**, **`camera_cx`**, **`camera_cy`** apply only when **`require_camera_info:=false`**. Optional simulator odom: **`-p subscribe_odom_gt:=true`**. Optional evaluation-frame export: pass sixteen floats **`-p eval_world_T_camera0:=[r0,r1,…,r15]`** (row-major **`4×4`**, **camera 0 at first keyframe → evaluation world**); when valid and non-zero, **`sparse_map_eval_world.npz`** is written beside **`sparse_map.npz`**. Descriptor-map knobs mirror **`dfm-descriptor-map`** and **`descriptor_map.json`**: **`-p feature_method:=ORB`** (or **`SIFT`**), **`-p feature_n_features:=2000`**, **`-p descriptor_merge_beta:=-1.0`** (sentinel **`-1.0`** = use **`1/(n+1)`** mean-equivalent EMA; positive values fix the EMA β), **`-p descriptor_max_match_distance:=-1.0`** (sentinel = method default: 64 Hamming for ORB, 220 L2 for SIFT), and **`-p descriptor_ratio_second_best:=-1.0`** (sentinel = Lowe ratio off). Sparse-map range gate (live **`PointCloud2`**, in-memory landmarks, shutdown **`sparse_map.npz`**): **`-p sparse_map_max_range_baseline_factor:=100.0`** drops cam0 points with **`‖X_cam0‖ > factor × ‖t_j − t_{j−1}‖`** between consecutive keyframes; **`≤ 0`** disables. Frame-orientation debug: **`-p log_frame_transforms:=true`** logs legible 4×4 **`world_T_drone`**, **`drone_T_camera`**, and **`world_T_camera`** on keyframe 0 (see [`configuration.race_rosbag.env`](configuration.race_rosbag.env)).

The thesis **`pipeline/`** module (imported when building the map) depends on **SciPy**. If you see **`ModuleNotFoundError: scipy`**, install into the **same Python environment** that runs `ros2` (e.g. `pixi add scipy` in your ROS env, or `python -m pip install -r src/incremental_vo_ros2/requirements.txt` using that interpreter), then rebuild or restart the node.

---

## How to run it

### Command-line (dataset replay)

After `pip install -e .`, entry points are **`dfm-export-steps`** (visual step PNGs / sequence export) and **`dfm-descriptor-map`** (descriptor landmark map + CSV). Example module invocations:

```bash
python -m viz.step_runner_cli <dataset_root> [options]
python -m viz.descriptor_map_cli <dataset_root> [options]
```

### Descriptor landmark map (`dfm-descriptor-map`)

Separate from **`dfm-export-steps`**: builds a **sparse 3D map in dataset world coordinates** (first keyframe at the origin after **`load_dataset`**). Uses the **same multi-baseline pairing** as sequence mode (`iter_sequence_pairs`, default **`--pair-lookback 10`**). Each triangulated point carries its ORB/SIFT descriptor from the **first view of the pair**; landmarks are **associated by nearest-neighbour descriptor distance**, then optionally gated in 3D: a descriptor match merges only if the new point lies within a sphere of radius **`spatial_merge_radius_m`** around the existing landmark (world frame). Set via **`descriptor_map.json`** or **`--spatial-merge-radius-m`**; omitted = gate disabled. On the live ROS node the same radius is **`keyframe_distance_m`** (minimum travelled distance before the next keyframe). Positions are updated with **EMA** — default **`merge_beta`** omitted/`null` ⇒ weights **`1/(n+1)`** (same as incremental mean); a numeric **`--merge-beta`** fixes classical EMA. **Prototype descriptors** use **replace-if-better**. **CLI overrides** optional **`descriptor_map.json`**.

```bash
dfm-descriptor-map path/to/dataset --run-dir runs/dmap --export-csv runs/dmap/landmarks.csv --iter-viz
```

| Option | Meaning |
|--------|---------|
| **`--run-dir`** | Output root (writes **`descriptor_map/`** with PNGs inside) |
| **`--pair-lookback`** | Same as sequence export (descriptor map uses multi-baseline pairing) |
| **`--merge-beta`** | Fixed EMA \(\beta\); omit for mean-equivalent behaviour |
| **`--descriptor-max-dist`** | Override NN association threshold (Hamming ORB / L2 SIFT) |
| **`--spatial-merge-radius-m`** | 3D merge gate radius in world frame (m); omit to use JSON or disable |
| **`--descriptor-map-config`** | Path to JSON (default tries **`<dataset>/descriptor_map.json`**) |
| **`--export-csv`** | Landmark table: id, world XYZ, **`n_updates`**, descriptor hex |
| **`--iter-viz`** | Extra PNG per processed pair under **`descriptor_map/iter/`** |

#### Single pair (default)

Processes frames **`i`** and **`j`**; writes **`steps/single/`**, **`steps/pair/`**, and (by default) **`steps/geometry/`**.

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --i 0 --j 1

# Illustration figures only (no triangulation / depth panels)
dfm-export-steps path/to/dataset --run-dir runs/illus --i 0 --j 1 --no-geometry-stages
```

#### Full sequence + fused landmarks

Default **multi-baseline** pairing (each frame with up to **10** prior frames):

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --sequence --fuse-merge-px 4
```

Consecutive pairs only (legacy layout size):

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --sequence --pair-lookback 1
```

| Option | Meaning |
|--------|---------|
| **`dataset_root`** | Folder with `calibration.json`, `motion.json`, and images |
| **`--run-dir`** | Output directory (default: `runs/export`) |
| **`--sequence`** | Enable multi-pair export + landmark merge + `summary/` artefacts |
| **`--fuse-merge-px`** | Pixel radius for merging landmarks on shared frames when `--sequence` (default `4`) |
| **`--pair-lookback`** | With `--sequence`: pair each frame `j` with `j-1…j-W` (default **`10`**). Use **`1`** for consecutive-only. |
| **`--i`**, **`--j`** | Frame indices for **single-pair** mode only (defaults `0`, `1`) |
| **`--no-geometry-stages`** | Skip `steps/geometry/` (triangulation, estimated depth, depth error) |
| **`--reproj-outlier-px`** | Reprojection threshold in pixels for magenta “reproj outlier” class (default `3`) |
| **`--rejection-audit`** | Override path for `rejection_audit.jsonl` (default `<run-dir>/rejection_audit.jsonl`) |

Outputs:

- **Single pair:** `runs/<name>/steps/single/01_original.png` … `04_descriptors.png`; `steps/pair/01_raw_input.png` … `05_inliers.png`; optional `steps/geometry/`; `rejection_audit.jsonl`.
- **Sequence:** `runs/<name>/pairs/iii_jjj/steps/{single,pair,geometry}/…` for every generated pair, plus `summary/trajectory_topdown_full_sequence.png`, `fused_landmarks_topdown.png`, `fused_estimated_depth_ref000.png`, `pairs_all_rejection_types.json`, and `rejection_audit.jsonl` at the run root.

### Python API (automation / notebooks)

```python
from pathlib import Path
from data.dataset import load_dataset
from viz.step_runner import export_all_stages, export_sequence_consecutive_pairs

ds = load_dataset(Path("path/to/dataset"))

export_all_stages(ds, "runs/single", i=0, j=1)

export_sequence_consecutive_pairs(
    ds,
    "runs/sequence",
    fuse_merge_px=4.0,
    pair_lookback=10,
)
```

Lower-level pieces live under **`pipeline/`** (geometry, triangulation, fusion), **`viz/overlays`** (drawing), and **`data/`** (loaders / validation).

---

## Packages

| Package | Role |
|---------|------|
| **`data/`** | Schema, JSON loaders, dataset validation |
| **`pipeline/`** | Features, matching, two-view geometry, triangulation, incremental map, landmark fusion |
| **`viz/`** | Step recorder, overlays, export orchestration, CLI |
| **`ros2_ws/src/incremental_vo_ros2/`** | Optional ROS 2 (Jazzy-compatible) Python node: live or rosbag streams → odometric keyframes → **`IncrementalMap`**; run outputs under **`ros2_runs/`** |

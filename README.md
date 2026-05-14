# DepthFromMovement

**Sparse 3D mapping** from a **monocular camera** plus **metric motion**, with an optional **pluggable fusion** step that combines odometry with a second pose source (ROS topic or offline JSON). The primary deployment is a **ROS~2 node** that selects keyframes, runs the same two-view triangulation core as the offline tools, and writes run artefacts. **`dfm-export-steps`** replays image folders with **`motion.json`** (and optional **`provided_motion.json`**) for figures and batch experiments.

---

## What this codebase does

1. **Metric pose fusion (optional)** — `pipeline.metric_fusion` combines a **primary** track (odometry / `motion.json`) with an optional **provided** track (`geometry_msgs/PoseStamped` in ROS, or `provided_motion.json` offline). Built-in strategies: `odom_only`, `provided_if_available`, `position_blend` (blend translations; rotation from odometry), and **`ekf_pose_velocity`** (ROS only: constant-velocity EKF on world position + velocity; odometry position, `TwistStamped` body linear velocity, and a **keyframe-rate** VO direction measurement from two-view `t_est`). Offline JSON fusion supports only the first three methods (not `ekf_pose_velocity`).

2. **Load a dataset** — Sorted RGB frames plus **`calibration.json`** (intrinsics) and **`motion.json`** (poses). Optional **`provided_motion.json`** (same schema as **`motion.json`**) and **`fusion.json`** (`method`, `position_blend_weight`). Optional GT depth / poses under documented paths (`data/`).

3. **Two-view pipeline** (ORB/SIFT features → matching → RANSAC on the essential constraint → `recoverPose` → **vision** relative rotation and translation **direction** → scale translation to **odometry** \(\|\mathbf{t}\|\) from the fused trajectory between the two frames → triangulation in a **camera-0–centric** world frame: **`load_dataset`** left-multiplies every `world_T_camera` (and optional GT poses) by \(\mathrm{inv}(W_0)\) so the first pose is identity; pairwise relative motion is unchanged, so two-view scaling matches the original track). Intrinsics-distorted images may still be **undistorted internally** before detection; **`world_T_camera`** in memory is this canonical map frame (simulation or odometry “world” offsets appear only in raw logs, e.g. ROS **`position.json`** `position` fields).

4. **Exports** — Step PNGs: raw mosaic, keypoints, matches, epilines, inlier/outlier matches, triangulation mosaic, sparse estimated depth on the reference frame, optional GT depth error. Depth colouring: **yellow = near → green → blue → pink = far** (no saturated red, to stay distinct from error highlights).

5. **Sequence mode** — For each frame index \(j\), pairs **\((j-1,j),\,(j-2,j),\,\ldots\)** up to **`pair_lookback`** earlier frames **(default \(10\))**---a breaking change versus older versions that exported only consecutive pairs **`(k,k+1)`**. Use **`--pair-lookback 1`** to restore consecutive-only pairing. Outputs live under **`runs/.../pairs/iii_jjj/steps/`**; landmarks are **fused** across edges (**`fuse-merge-px`**); **`summary/`** contains full trajectory plus fused top-down and fused sparse depth on reference frame 0. **Descriptors are computed once per frame** (`compute_frame_features_cache`) and reused for every pair and for the keypoints figure—single-pair export still detects per call unless you pass a cache from Python.

---

## Dataset layout

Point the CLI at a folder that contains:

| Path | Purpose |
|------|---------|
| **`calibration.json`** | `K` (3×3), optional `dist_coeffs`, optional `image_size` |
| **`motion.json`** | Per-frame **camera→world** pose (`world_T_camera`): \(\mathbf{X}_w = \mathbf{R}\mathbf{X}_c + \mathbf{t}\). Fields: `pose_convention`, `representation` (`absolute` or `relative_to_prev`), `frames[]` with `T` (4×4 or 3×4). **`load_dataset`** rewrites the loaded trajectory so frame **0** is the map origin (same convention; GT poses transformed consistently). |
| **`provided_motion.json`** (optional) | Second pose track, **same schema** as **`motion.json`**, one transform per image. Fused with the primary track per **`fusion.json`** when both files exist. |
| **`fusion.json`** (optional) | Offline fusion: **`method`** (`odom_only`, `provided_if_available`, `position_blend` only — **not** `ekf_pose_velocity`; default **`position_blend`** when this file is missing but **`provided_motion.json`** exists) and optional **`position_blend_weight`** in \([0,1]\). Ignored if **`provided_motion.json`** is absent. |
| **`images/`** | Frames (`*.png` by default); if empty, the loader also searches image files in the dataset root |

Optional: **`features.json`** — detector/matcher settings (ORB vs SIFT, counts, Lowe ratio, cross-check, ORB pyramid / SIFT contrast); omitted keys use library defaults (same as **`FeatureConfig`**). **`descriptor_map.json`** (optional) — knobs for **`dfm-descriptor-map`**: **`merge_beta`** (`null` = incremental arithmetic mean via \(1/(n+1)\)), **`max_match_distance`**, **`ratio_second_best`**. **`gt_depth/`** (depth maps named like image stems), **`gt_poses.txt`** (TUM-style; length must match image count).

Helper (optional): **`scripts/init_test_dataset_motion.py`** — builds `motion.json` from image count when you only need a synthetic baseline.

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

The **`ros2_ws/`** colcon workspace contains **`incremental_vo_ros2`**: a node that subscribes to a monocular image and fused odometry, optionally to a **`geometry_msgs/PoseStamped`** second pose (**`provided_pose_topic`**) for snapshot fusion methods, and optionally **`geometry_msgs/TwistStamped`** on **`velocity_topic`** when using **`fusion_method:=ekf_pose_velocity`**. Fusion is implemented in **`pipeline.metric_fusion`**; **`ekf_pose_velocity`** runs a **6D constant-velocity EKF** (world position + velocity) with odometry position updates, body-frame velocity rotated by odometry attitude, and after each successful keyframe pair a **soft position** update from two-view translation direction (same rate as keyframes, not per image). Other **`fusion_method`** values snapshot-fuse poses. The node selects **keyframes** when the **fused** translation has moved at least **`keyframe_distance_m`** (default **0.5** m), runs **`pipeline.IncrementalMap`** over each new keyframe against up to **`pair_lookback`** prior keyframes for triangulation + reprojection diagnostics, then feeds every successful **`TwoViewResult`** into **`pipeline.DescriptorLandmarkMap`** (same class as **`dfm-descriptor-map`**) anchored at the first keyframe — landmarks are associated by **nearest-neighbour descriptor distance**, positions updated with **EMA** (default **`1/(n+1)`**), and **prototype descriptors are replace-if-better** so the live 3D map is updated as descriptors improve instead of being re-appended per baseline. Poses passed into **`IncrementalMap`** are **canonical** (first keyframe at the map origin); **`position.json`** still logs **raw** fused metric positions for traceability. On shutdown the node writes **`output_root/ros2_runs/run_<timestamp>/`**: **`images/`**, **`position.json`** (includes **`fusion_method`**, optional topics, effective feature/descriptor knobs, **`landmarks_reference_frame: "camera_0"`**, **`map_coordinate_frame: "camera0"`**, and optional flat **`eval_world_T_camera0`**), **`sparse_map.npz`** (descriptor-fused points in the camera-0 map frame), optional **`sparse_map_eval_world.npz`** when **`eval_world_T_camera0`** is set to a valid non-zero **`4×4`** (row-major sixteen floats, **camera 0 → evaluation world**, same homogeneous convention as **`world_T_camera`**), and **`descriptor_landmarks.csv`** (id, cam0 XYZ, **`n_updates`**, descriptor hex — same schema as **`dfm-descriptor-map`**). The process must import **`pipeline.*`** from this repository (the node walks parents until **`pipeline/map.py`**).

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

Useful **`--ros-args`** parameters include **`-p use_sim_time:=true`** when playing a rosbag with **`ros2 bag play … --clock`**, **`-p output_root:=…`**, **`-p keyframe_distance_m:=0.5`**, **`-p pair_lookback:=10`**, topic overrides (**`-p image_topic:=…`**, **`-p odom_main_topic:=…`**), snapshot fusion (**`-p fusion_method:=odom_only`**, **`-p fusion_position_blend_weight:=0.3`**, **`-p provided_pose_topic:=/my/pose`**), and EKF fusion (**`-p fusion_method:=ekf_pose_velocity`**, **`-p velocity_topic:=/my/body/twist`**, **`-p ekf_sigma_odom_position:=0.02`**, …). Default **`fusion_method`** is **`position_blend`**. Optional simulator odom: **`-p subscribe_odom_gt:=true`**. Optional evaluation-frame export: pass sixteen floats **`-p eval_world_T_camera0:=[r0,r1,…,r15]`** (row-major **`4×4`**, **camera 0 at first keyframe → evaluation world**); when valid and non-zero, **`sparse_map_eval_world.npz`** is written beside **`sparse_map.npz`**. Descriptor-map knobs mirror **`dfm-descriptor-map`** and **`descriptor_map.json`**: **`-p feature_method:=ORB`** (or **`SIFT`**), **`-p feature_n_features:=2000`**, **`-p descriptor_merge_beta:=-1.0`** (sentinel **`-1.0`** = use **`1/(n+1)`** mean-equivalent EMA; positive values fix the EMA β), **`-p descriptor_max_match_distance:=-1.0`** (sentinel = method default: 64 Hamming for ORB, 220 L2 for SIFT), and **`-p descriptor_ratio_second_best:=-1.0`** (sentinel = Lowe ratio off).

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

Separate from **`dfm-export-steps`**: builds a **sparse 3D map in camera-0 coordinates** (origin at the first camera centre). Uses the **same multi-baseline pairing** as sequence mode (`iter_sequence_pairs`, default **`--pair-lookback 10`**). Each triangulated point carries its ORB/SIFT descriptor from the **first view of the pair**; landmarks are **associated by nearest-neighbour descriptor distance**, positions updated with **EMA** — default **`merge_beta`** omitted/`null` ⇒ weights **`1/(n+1)`** (same as incremental mean); a numeric **`--merge-beta`** fixes classical EMA. **Prototype descriptors** use **replace-if-better**. **CLI overrides** optional **`descriptor_map.json`**.

```bash
dfm-descriptor-map path/to/dataset --run-dir runs/dmap --export-csv runs/dmap/landmarks.csv --iter-viz
```

| Option | Meaning |
|--------|---------|
| **`--run-dir`** | Output root (writes **`descriptor_map/`** with PNGs inside) |
| **`--pair-lookback`** | Same as sequence export (descriptor map uses multi-baseline pairing) |
| **`--merge-beta`** | Fixed EMA \(\beta\); omit for mean-equivalent behaviour |
| **`--descriptor-max-dist`** | Override NN association threshold (Hamming ORB / L2 SIFT) |
| **`--descriptor-map-config`** | Path to JSON (default tries **`<dataset>/descriptor_map.json`**) |
| **`--export-csv`** | Landmark table: id, cam0 XYZ, **`n_updates`**, descriptor hex |
| **`--iter-viz`** | Extra PNG per processed pair under **`descriptor_map/iter/`** |

#### Single pair (default)

Processes frames **`i`** and **`j`** only; writes PNGs under **`--run-dir/steps/`**.

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --i 0 --j 1
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
| **`--sequence`** | Enable multi-pair export + fusion + `summary/` artefacts |
| **`--fuse-merge-px`** | Pixel radius for merging landmarks on shared frames when `--sequence` (default `4`) |
| **`--pair-lookback`** | With `--sequence`: pair each frame `j` with `j-1…j-W` (default **`10`**). Use **`1`** for consecutive-only. |
| **`--i`**, **`--j`** | Frame indices for **single-pair** mode only (defaults `0`, `1`) |

Outputs:

- **Single pair:** `runs/<name>/steps/01_raw_input.png` … `08_depth_error.png` (fixed step order).
- **Sequence:** `runs/<name>/pairs/iii_jjj/steps/…` for every generated pair, plus `summary/trajectory_topdown_full_sequence.png`, `fused_landmarks_topdown.png`, `fused_estimated_depth_ref000.png`.

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

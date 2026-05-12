# DepthFromMovement

Bachelor thesis code: **sparse depth / structure from two-view geometry** driven by **known camera motion** (your metric trajectory). Images supply correspondences; **`motion.json`** supplies poses that contribute to relative motion according to **`motion-confidence`** (see below).

---

## What this codebase does

1. **Load a dataset** — Sorted RGB frames plus **`calibration.json`** (intrinsics) and **`motion.json`** (poses). Optional GT depth / poses under documented paths (`data/`).

2. **Two-view pipeline** (ORB/SIFT features → matching → RANSAC on the essential constraint → triangulation). A scalar **`motion_confidence`** \(\alpha\in[0,1]\) blends **relative** rotation and translation from **visual geometry** (\(\alpha\to 0\)) with **odometry-relative** \(R,\mathbf{t}\) from **`motion.json`** (\(\alpha\to 1\)). Intrinsics-distorted images may still be **undistorted internally** before detection; **`world_T_camera`** from the motion file is used to express triangulated points in the dataset world frame (the same anchoring applies when \(\alpha\) is small).

3. **Exports** — Step PNGs: raw mosaic, keypoints, matches, epilines, inlier/outlier matches, triangulation mosaic, sparse estimated depth on the reference frame, optional GT depth error. Depth colouring: **yellow = near → green → blue → pink = far** (no saturated red, to stay distinct from error highlights).

4. **Sequence mode** — For each frame index \(j\), pairs **\((j-1,j),\,(j-2,j),\,\ldots\)** up to **`pair_lookback`** earlier frames **(default \(10\))**---a breaking change versus older versions that exported only consecutive pairs **`(k,k+1)`**. Use **`--pair-lookback 1`** to restore consecutive-only pairing. Outputs live under **`runs/.../pairs/iii_jjj/steps/`**; landmarks are **fused** across edges (**`fuse-merge-px`**); **`summary/`** contains full trajectory plus fused top-down and fused sparse depth on reference frame 0. **Descriptors are computed once per frame** (`compute_frame_features_cache`) and reused for every pair and for the keypoints figure—single-pair export still detects per call unless you pass a cache from Python.

---

## Dataset layout

Point the CLI at a folder that contains:

| Path | Purpose |
|------|---------|
| **`calibration.json`** | `K` (3×3), optional `dist_coeffs`, optional `image_size` |
| **`motion.json`** | Per-frame **camera→world** pose (`world_T_camera`): \(\mathbf{X}_w = \mathbf{R}\mathbf{X}_c + \mathbf{t}\). Fields: `pose_convention`, `representation` (`absolute` or `relative_to_prev`), `frames[]` with `T` (4×4 or 3×4) |
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

### ROS 2 package (optional)

The **`ros2_ws/`** colcon workspace contains **`incremental_vo_ros2`**: a node that subscribes to a monocular image and fused odometry, selects **keyframes** when the platform has moved at least **`keyframe_distance_m`** (default **0.5** m), runs the same **`pipeline.IncrementalMap`** two-view step as the offline exporter between consecutive keyframes, and on shutdown writes artefacts under **`output_root/ros2_runs/run_<timestamp>/`**: **`images/`** (PNG keyframes), **`position.json`**, and **`sparse_map.npz`**. The process must be able to import **`pipeline.*`** from this repository (the node walks parent directories until it finds **`pipeline/map.py`**).

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

Useful **`--ros-args`** parameters include **`-p use_sim_time:=true`** when playing a rosbag with **`ros2 bag play … --clock`**, **`-p output_root:=…`**, **`-p keyframe_distance_m:=0.5`**, and topic overrides (**`-p image_topic:=…`**, **`-p odom_main_topic:=…`**). Optional simulator odom: **`-p subscribe_odom_gt:=true`**.

The thesis **`pipeline/`** module (imported when building the map) depends on **SciPy**. If you see **`ModuleNotFoundError: scipy`**, install into the **same Python environment** that runs `ros2` (e.g. `pixi add scipy` in your ROS env, or `python -m pip install -r src/incremental_vo_ros2/requirements.txt` using that interpreter), then rebuild or restart the node.

---

## How to run it

### Command-line (primary)

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
| **`--motion-confidence`**, **`--pair-lookback`** | Same semantics as sequence export |
| **`--merge-beta`** | Fixed EMA \(\beta\); omit for mean-equivalent behaviour |
| **`--descriptor-max-dist`** | Override NN association threshold (Hamming ORB / L2 SIFT) |
| **`--descriptor-map-config`** | Path to JSON (default tries **`<dataset>/descriptor_map.json`**) |
| **`--export-csv`** | Landmark table: id, cam0 XYZ, **`n_updates`**, descriptor hex |
| **`--iter-viz`** | Extra PNG per processed pair under **`descriptor_map/iter/`** |

#### Single pair (default)

Processes frames **`i`** and **`j`** only; writes PNGs under **`--run-dir/steps/`**.

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --i 0 --j 1 --motion-confidence 1
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
| **`--motion-confidence`** | \(0\) = vision-only relative motion (no odometry alignment of translation); \(1\) = triangulation from odometry relative pose; values in between **SLERP** rotation and **linearly mix** translation with odometry (default **`1`**) |

Outputs:

- **Single pair:** `runs/<name>/steps/01_raw_input.png` … `08_depth_error.png` (fixed step order).
- **Sequence:** `runs/<name>/pairs/iii_jjj/steps/…` for every generated pair, plus `summary/trajectory_topdown_full_sequence.png`, `fused_landmarks_topdown.png`, `fused_estimated_depth_ref000.png`.

### Python API (automation / notebooks)

```python
from pathlib import Path
from data.dataset import load_dataset
from viz.step_runner import export_all_stages, export_sequence_consecutive_pairs

ds = load_dataset(Path("path/to/dataset"))

export_all_stages(ds, "runs/single", i=0, j=1, motion_confidence=1.0)

export_sequence_consecutive_pairs(
    ds,
    "runs/sequence",
    motion_confidence=1.0,
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

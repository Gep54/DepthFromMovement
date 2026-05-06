# DepthFromMovement

Bachelor thesis code: **sparse depth / structure from two-view geometry** driven by **known camera motion** (your metric trajectory). Images supply correspondences; **`motion.json`** supplies poses used for triangulation (unless you explicitly estimate relative pose from matches).

---

## What this codebase does

1. **Load a dataset** — Sorted RGB frames plus **`calibration.json`** (intrinsics) and **`motion.json`** (poses). Optional GT depth / poses under documented paths (`data/`).

2. **Two-view pipeline** (ORB/SIFT features → matching → optional essential-matrix RANSAC):
   - **`known_pose`** (default): relative geometry matches **`motion.json`**; points are triangulated in a common world frame using **camera→world** poses (internally inverted for projection).
   - **`estimate_essential`**: estimates \(\mathbf{E}\) from correspondences, then uses **`motion.json`** only for **metric scale** alignment (`\|\mathbf{t}_o\| / \|\mathbf{t}_v\|`).

3. **Exports** — Step PNGs (raw → matches → epilines → triangulation → depth visuals → trajectory → GT depth error if available). Depth colouring: **red = near**, **blue = far**, sparse points drawn with a soft halo.

4. **Sequence mode** — Runs every consecutive pair `(0,1), (1,2), …`, writes each under **`runs/.../pairs/MMM_NNN/steps/`**, **fuses** landmarks across edges when the **same frame** sees nearby pixels (parameter **`fuse-merge-px`**), and writes **`summary/`** (full trajectory + fused top-down + fused depth on reference frame 0).

---

## Dataset layout

Point the CLI at a folder that contains:

| Path | Purpose |
|------|---------|
| **`calibration.json`** | `K` (3×3), optional `dist_coeffs`, optional `image_size` |
| **`motion.json`** | Per-frame **camera→world** pose (`world_T_camera`): \(\mathbf{X}_w = \mathbf{R}\mathbf{X}_c + \mathbf{t}\). Fields: `pose_convention`, `representation` (`absolute` or `relative_to_prev`), `frames[]` with `T` (4×4 or 3×4) |
| **`images/`** | Frames (`*.png` by default); if empty, the loader also searches image files in the dataset root |

Optional: **`features.json`** — detector/matcher settings (ORB vs SIFT, counts, Lowe ratio, cross-check, ORB pyramid / SIFT contrast); omitted keys use library defaults (same as **`FeatureConfig`**). **`gt_depth/`** (depth maps named like image stems), **`gt_poses.txt`** (TUM-style; length must match image count).

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

---

## How to run it

### Command-line (primary)

After `pip install -e .`, the entry point is **`dfm-export-steps`**. Equivalent module invocation:

```bash
python -m viz.step_runner_cli <dataset_root> [options]
```

#### Single pair (default)

Processes frames **`i`** and **`j`** only; writes PNGs under **`--run-dir/steps/`**.

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --i 0 --j 1 --motion-mode known_pose
```

#### Full sequence + fused landmarks

Runs all consecutive pairs and fusion summaries:

```bash
dfm-export-steps path/to/dataset --run-dir runs/demo --sequence --fuse-merge-px 4
```

| Option | Meaning |
|--------|---------|
| **`dataset_root`** | Folder with `calibration.json`, `motion.json`, and images |
| **`--run-dir`** | Output directory (default: `runs/export`) |
| **`--sequence`** | Enable multi-pair export + fusion + `summary/` artefacts |
| **`--fuse-merge-px`** | Pixel radius for merging landmarks on shared frames when `--sequence` (default `4`) |
| **`--i`**, **`--j`** | Frame indices for **single-pair** mode only (defaults `0`, `1`) |
| **`--motion-mode`** | `known_pose` or `estimate_essential` |

Outputs:

- **Single pair:** `runs/<name>/steps/01_raw_input.png` … `11_depth_error.png` (fixed step order).
- **Sequence:** `runs/<name>/pairs/000_001/steps/…`, …, plus `summary/trajectory_topdown_full_sequence.png`, `fused_landmarks_topdown.png`, `fused_estimated_depth_ref000.png`.

### Python API (automation / notebooks)

```python
from pathlib import Path
from data.dataset import load_dataset
from viz.step_runner import export_all_stages, export_sequence_consecutive_pairs

ds = load_dataset(Path("path/to/dataset"))

export_all_stages(ds, "runs/single", i=0, j=1, motion_mode="known_pose")

export_sequence_consecutive_pairs(
    ds,
    "runs/sequence",
    motion_mode="known_pose",
    fuse_merge_px=4.0,
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

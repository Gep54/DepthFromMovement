"""Parse ROS ``tf_static`` echo dumps and compose static transforms."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import numpy as np

# Pipeline / race rosbag convention (odom parent + RGB optical frame).
RACE_PIPELINE_WORLD_FRAME = "uav1/gps_baro_origin"
RACE_PIPELINE_CAMERA_FRAME = "uav1/rgb"
RACE_FIXED_ORIGIN_FRAME = "uav1/fixed_origin"
RACE_LOCAL_ORIGIN_FRAME = "uav1/local_origin"
RACE_FCU_FRAME = "uav1/fcu"

TfEdge = tuple[str, str]
TfEdgeMap = dict[TfEdge, np.ndarray]


def quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3×3 rotation (ROS / tf2 convention)."""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_from_translation_quat(
    tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    """4×4 ``parent_T_child``: ``X_parent = T @ X_child`` (ROS ``TransformStamped``)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix(qx, qy, qz, qw)
    T[:3, 3] = (float(tx), float(ty), float(tz))
    return T


def compose_se3(T_left: np.ndarray, T_right: np.ndarray) -> np.ndarray:
    return np.asarray(T_left, dtype=np.float64) @ np.asarray(T_right, dtype=np.float64)


def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def load_tf_static_echo(path: Path) -> TfEdgeMap:
    """
    Load ``ros2 topic echo /tf_static``-style YAML (multiple ``---`` documents).

    Each transform defines ``parent_T_child`` with ``header.frame_id`` parent and
    ``child_frame_id`` child.
    """
    text = path.read_text(encoding="utf-8")
    edges: TfEdgeMap = {}
    for block in text.split("---"):
        if "child_frame_id" not in block:
            continue
        parents = re.findall(r"^\s*frame_id:\s*(\S+)\s*$", block, re.MULTILINE)
        children = re.findall(r"^\s*child_frame_id:\s*(\S+)\s*$", block, re.MULTILINE)
        trans_blocks = re.findall(
            r"translation:\s*\n\s*x:\s*([^\n]+)\n\s*y:\s*([^\n]+)\n\s*z:\s*([^\n]+)"
            r"\s*\n\s*rotation:\s*\n\s*x:\s*([^\n]+)\n\s*y:\s*([^\n]+)\n\s*z:\s*([^\n]+)\n\s*w:\s*([^\n]+)",
            block,
        )
        if not (len(parents) == len(children) == len(trans_blocks)):
            continue
        for parent, child, nums in zip(parents, children, trans_blocks, strict=True):
            tx, ty, tz, qx, qy, qz, qw = (float(x) for x in nums)
            edges[(parent, child)] = transform_from_translation_quat(tx, ty, tz, qx, qy, qz, qw)
    if not edges:
        raise ValueError(f"no TF edges parsed from {path}")
    return edges


def with_race_odom_world_aliases(edges: TfEdgeMap) -> TfEdgeMap:
    """
  Extend static edges for the race bag: odometry world is ``gps_baro_origin``; at mission
  start it coincides with ``local_origin`` (not published on ``/tf_static``).
    """
    out = dict(edges)
    if (RACE_PIPELINE_WORLD_FRAME, RACE_LOCAL_ORIGIN_FRAME) not in out:
        out[(RACE_PIPELINE_WORLD_FRAME, RACE_LOCAL_ORIGIN_FRAME)] = np.eye(4, dtype=np.float64)
    if (RACE_LOCAL_ORIGIN_FRAME, RACE_PIPELINE_WORLD_FRAME) not in out:
        out[(RACE_LOCAL_ORIGIN_FRAME, RACE_PIPELINE_WORLD_FRAME)] = np.eye(4, dtype=np.float64)
    return out


def lookup_transform(edges: TfEdgeMap, target_frame: str, source_frame: str) -> np.ndarray:
    """
    Return ``target_T_source`` with ``X_target = target_T_source @ X_source``.

    BFS from *source* frame, accumulating ``current_T_source``.
    """
    if target_frame == source_frame:
        return np.eye(4, dtype=np.float64)

    forward: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (parent, child), T_pc in edges.items():
        # ``T_pc``: ``X_parent = T_pc @ X_child``
        forward.setdefault(parent, []).append((child, T_pc))
        forward.setdefault(child, []).append((parent, invert_se3(T_pc)))

    queue: deque[tuple[str, np.ndarray]] = deque([(source_frame, np.eye(4, dtype=np.float64))])
    visited = {source_frame}
    while queue:
        frame, T_frame_from_source = queue.popleft()
        if frame == target_frame:
            return T_frame_from_source
        for nxt, T_nxt_from_frame in forward.get(frame, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            # ``X_nxt = T_nxt_from_frame @ X_frame``
            T_nxt_from_source = compose_se3(T_nxt_from_frame, T_frame_from_source)
            queue.append((nxt, T_nxt_from_source))

    raise ValueError(f"no static TF path from {source_frame!r} to {target_frame!r}")


# Backward-compatible alias
lookup_transform_fixed_bfs = lookup_transform


def default_race_tf_static_path() -> Path:
    """Repo-root ``echo_tf_static`` from ``ros2 topic echo /tf_static``."""
    return Path(__file__).resolve().parent.parent / "echo_tf_static"


def resolve_tf_static_path(dataset_root: Path) -> Path | None:
    for name in ("tf_static.echo", "tf_static.yaml", "echo_tf_static"):
        p = dataset_root / name
        if p.is_file():
            return p
    repo = default_race_tf_static_path()
    return repo if repo.is_file() else None


def reframe_world_T_sensor_poses(
    poses: list[np.ndarray],
    edges: TfEdgeMap,
    *,
    src_world: str,
    dst_world: str,
    src_sensor: str,
    dst_sensor: str,
) -> list[np.ndarray]:
    """
    Convert absolute ``world_T_sensor`` poses between world and sensor frames.

    Input poses are ``src_world_T_src_sensor``; output is ``dst_world_T_dst_sensor``.
    """
    world_dst_T_src = lookup_transform(edges, dst_world, src_world)
    sensor_src_T_dst = lookup_transform(edges, src_sensor, dst_sensor)
    out: list[np.ndarray] = []
    for W in poses:
        T = np.asarray(W, dtype=np.float64)
        # dst_W_dst_sensor = world_dst_T_src @ src_W_src_sensor @ sensor_src_T_dst
        out.append(compose_se3(compose_se3(world_dst_T_src, T), sensor_src_T_dst))
    return out

"""Load incremental_vo_node parameters from configuration.env with CLI precedence."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

__all__ = [
    "KNOWN_PARAMETERS",
    "apply_config_to_argv",
    "coerce_ros_param_value",
    "extract_cli_param_overrides",
    "load_env_config",
    "resolve_config_path",
    "strip_config_file_flag",
]

KNOWN_PARAMETERS = frozenset(
    {
        "image_topic",
        "odom_main_topic",
        "subscribe_odom_gt",
        "odom_gt_topic",
        "keyframe_distance_m",
        "keyframe_buffer_start_fraction",
        "output_root",
        "max_image_buffer",
        "camera_info_topic",
        "camera_info_qos_durability",
        "require_camera_info",
        "camera_fx",
        "camera_fy",
        "camera_cx",
        "camera_cy",
        "pair_lookback",
        "publish_sparse_map",
        "sparse_map_topic",
        "sparse_map_publish_period_s",
        "sparse_map_frame_id",
        "sparse_map_max_range_baseline_factor",
        "publish_camera_pose_debug",
        "camera_pose_debug_topic",
        "camera_pose_debug_frame_id",
        "camera_pose_debug_child_frame_id",
        "publish_camera_pose_tf",
        "publish_keyframe_markers",
        "keyframe_marker_topic",
        "keyframe_marker_length_m",
        "keyframe_marker_frame_id",
        "apply_tf_to_camera_pose",
        "save_run_on_shutdown",
        "export_offline_dataset",
        "offline_dataset_root",
        "offline_dataset_image_prefix",
        "offline_dataset_pose_source",
        "base_frame",
        "camera_frame",
        "tf_lookup_period_s",
        "tf_use_latest_time",
        "log_image_hz",
        "feature_method",
        "feature_n_features",
        "descriptor_merge_beta",
        "descriptor_max_match_distance",
        "descriptor_ratio_second_best",
        "fusion_method",
        "fusion_position_blend_weight",
        "provided_pose_topic",
        "eval_world_T_camera0",
    }
)

_CONFIG_FILE_FLAG = "--config-file"
_DEFAULT_CONFIG_NAME = "configuration.env"
_ENV_CONFIG_VAR = "INCREMENTAL_VO_CONFIG"


def load_env_config(path: Path) -> dict[str, str]:
    """Parse dotenv-style ``KEY=value`` lines (``#`` comments, optional ``export`` prefix)."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            _warn(f"{path}:{lineno}: skipping line without '=': {raw!r}")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            _warn(f"{path}:{lineno}: empty key")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in KNOWN_PARAMETERS:
            _warn(f"{path}:{lineno}: unknown parameter {key!r} (ignored)")
            continue
        out[key] = value
    return out


def coerce_ros_param_value(raw: str) -> str:
    """Format a string value for ROS ``-p name:=value`` injection."""
    s = raw.strip()
    if not s:
        return '""'
    lower = s.lower()
    if lower in ("true", "false"):
        return lower
    if lower in ("1", "0") and re.fullmatch(r"[01]", s):
        return "true" if s == "1" else "false"
    if s.startswith("[") and s.endswith("]"):
        return s
    if "," in s and re.fullmatch(r"[\d\s.,eE+-]+", s):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return "[" + ",".join(parts) + "]"
    if re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", s):
        return s
    if " " in s or ":=" in s:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _parse_p_entry(flag: str, argv: list[str], i: int) -> tuple[str, str, int] | None:
    """Return ``(name, value, next_index)`` or ``None`` if not a parameter flag."""
    if flag not in ("-p", "--param"):
        return None
    if i + 1 >= len(argv):
        return None
    token = argv[i + 1]
    if ":=" in token:
        name, _, value = token.partition(":=")
        return name, value, i + 2
    if i + 2 >= len(argv):
        return None
    return token, argv[i + 2], i + 3


def extract_cli_param_overrides(argv: list[str]) -> dict[str, str]:
    """Collect ``-p`` / ``--param`` overrides from ``--ros-args`` sections (last wins)."""
    out: dict[str, str] = {}
    i = 0
    in_ros_args = False
    while i < len(argv):
        tok = argv[i]
        if tok == "--ros-args":
            in_ros_args = True
            i += 1
            continue
        if in_ros_args and tok.startswith("--") and tok not in ("-p", "--param"):
            if tok in ("--",):
                pass
            elif not tok.startswith("-p") and tok != "--param":
                in_ros_args = False
        if in_ros_args:
            parsed = _parse_p_entry(tok, argv, i)
            if parsed is not None:
                name, value, i = parsed
                out[name] = value
                continue
        i += 1
    return out


def _strip_known_p_from_ros_args(argv: list[str]) -> list[str]:
    """Remove ``-p`` entries for :data:`KNOWN_PARAMETERS` inside ``--ros-args`` blocks."""
    out: list[str] = []
    i = 0
    in_ros_args = False
    while i < len(argv):
        tok = argv[i]
        if tok == "--ros-args":
            in_ros_args = True
            out.append(tok)
            i += 1
            continue
        if in_ros_args and tok.startswith("--") and tok not in ("-p", "--param"):
            if tok not in ("--",):
                in_ros_args = False
        if in_ros_args:
            parsed = _parse_p_entry(tok, argv, i)
            if parsed is not None:
                name, _, next_i = parsed
                if name in KNOWN_PARAMETERS:
                    i = next_i
                    continue
        out.append(tok)
        i += 1
    return out


def strip_config_file_flag(argv: list[str]) -> tuple[list[str], Path | None]:
    """Remove ``--config-file PATH`` from *argv*; return updated argv and path if given."""
    out: list[str] = []
    config_path: Path | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == _CONFIG_FILE_FLAG:
            if i + 1 >= len(argv):
                raise ValueError(f"{_CONFIG_FILE_FLAG} requires a path argument")
            config_path = Path(argv[i + 1]).expanduser()
            i += 2
            continue
        if tok.startswith(f"{_CONFIG_FILE_FLAG}="):
            config_path = Path(tok.split("=", 1)[1]).expanduser()
            i += 1
            continue
        out.append(tok)
        i += 1
    return out, config_path


def resolve_config_path(argv: list[str], explicit: Path | None) -> Path | None:
    """Resolve config file: explicit flag, env var, then ``./configuration.env``."""
    if explicit is not None:
        return explicit
    env = os.environ.get(_ENV_CONFIG_VAR, "").strip()
    if env:
        return Path(env).expanduser()
    default = Path.cwd() / _DEFAULT_CONFIG_NAME
    if default.is_file():
        return default
    return None


def _inject_params_into_argv(argv: list[str], params: dict[str, str]) -> list[str]:
    """Append ``-p name:=value`` pairs inside the last ``--ros-args`` block, or create one."""
    if not params:
        return argv
    entries: list[str] = []
    for name, value in params.items():
        entries.extend(["-p", f"{name}:={coerce_ros_param_value(value)}"])

    if "--ros-args" in argv:
        out = list(argv)
        idx = len(out)
        for i, tok in enumerate(out):
            if tok == "--ros-args":
                idx = i + 1
                while idx < len(out) and not (out[idx].startswith("--") and out[idx] not in ("-p", "--param")):
                    if out[idx] == "--":
                        break
                    idx += 1
        out[idx:idx] = entries
        return out

    return [*argv, "--ros-args", *entries]


def apply_config_to_argv(argv: list[str] | None = None) -> list[str]:
    """Merge configuration.env into *argv* (CLI ``-p`` overrides file; code defaults unchanged)."""
    base = list(sys.argv[1:] if argv is None else argv)
    stripped, explicit_config = strip_config_file_flag(base)
    config_path = resolve_config_path(stripped, explicit_config)

    file_params: dict[str, str] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        file_params = load_env_config(config_path)

    cli_params = extract_cli_param_overrides(stripped)
    file_known = {k: v for k, v in file_params.items() if k in KNOWN_PARAMETERS}
    cli_known = {k: v for k, v in cli_params.items() if k in KNOWN_PARAMETERS}
    cli_unknown = {k: v for k, v in cli_params.items() if k not in KNOWN_PARAMETERS}

    merged_known = {
        k: v for k, v in {**file_known, **cli_known}.items() if v.strip() != ""
    }

    rebuilt = _strip_known_p_from_ros_args(stripped)
    if merged_known:
        rebuilt = _inject_params_into_argv(rebuilt, merged_known)
    if cli_unknown:
        rebuilt = _inject_params_into_argv(rebuilt, cli_unknown)

    return rebuilt


def _warn(message: str) -> None:
    print(f"[incremental_vo_ros2] warning: {message}", file=sys.stderr)

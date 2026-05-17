"""Unit tests for incremental_vo_ros2.param_config (no ROS runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "ros2_ws" / "src" / "incremental_vo_ros2"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from incremental_vo_ros2.param_config import (  # noqa: E402
    apply_config_to_argv,
    coerce_ros_param_value,
    extract_cli_param_overrides,
    load_env_config,
    strip_config_file_flag,
)


def test_load_env_config_basic(tmp_path: Path) -> None:
    cfg = tmp_path / "configuration.env"
    cfg.write_text(
        "# comment\n"
        "export keyframe_distance_m=0.3\n"
        'image_topic="/my/topic"\n'
        "unknown_param=1\n"
        "\n",
        encoding="utf-8",
    )
    data = load_env_config(cfg)
    assert data["keyframe_distance_m"] == "0.3"
    assert data["image_topic"] == "/my/topic"
    assert "unknown_param" not in data


def test_coerce_ros_param_value() -> None:
    assert coerce_ros_param_value("true") == "true"
    assert coerce_ros_param_value("FALSE") == "false"
    assert coerce_ros_param_value("1") == "true"
    assert coerce_ros_param_value("0") == "false"
    assert coerce_ros_param_value("0.5") == "0.5"
    assert coerce_ros_param_value("0,1,2") == "[0,1,2]"
    assert coerce_ros_param_value("[1,2,3]") == "[1,2,3]"
    assert coerce_ros_param_value("hello world") == '"hello world"'


def test_extract_cli_param_overrides() -> None:
    argv = ["--ros-args", "-p", "keyframe_distance_m:=0.1", "-p", "use_sim_time:=true"]
    got = extract_cli_param_overrides(argv)
    assert got["keyframe_distance_m"] == "0.1"
    assert got["use_sim_time"] == "true"


def test_extract_cli_param_overrides_colon_form() -> None:
    argv = ["--ros-args", "-p", "pair_lookback:=10"]
    assert extract_cli_param_overrides(argv)["pair_lookback"] == "10"


def test_strip_config_file_flag() -> None:
    stripped, path = strip_config_file_flag(["--config-file", "/tmp/c.env", "--ros-args"])
    assert path == Path("/tmp/c.env")
    assert stripped == ["--ros-args"]


def test_apply_config_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "configuration.env"
    cfg.write_text("keyframe_distance_m=0.99\npair_lookback=5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INCREMENTAL_VO_CONFIG", raising=False)

    argv = apply_config_to_argv(["--config-file", str(cfg)])
    assert "-p" in argv
    merged = extract_cli_param_overrides(argv)
    assert merged["keyframe_distance_m"] == "0.99"
    assert merged["pair_lookback"] == "5"

    argv2 = apply_config_to_argv(
        ["--config-file", str(cfg), "--ros-args", "-p", "keyframe_distance_m:=0.11"]
    )
    merged2 = extract_cli_param_overrides(argv2)
    assert merged2["keyframe_distance_m"] == "0.11"
    assert merged2["pair_lookback"] == "5"


def test_apply_config_preserves_use_sim_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "configuration.env"
    cfg.write_text("keyframe_distance_m=0.5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INCREMENTAL_VO_CONFIG", raising=False)

    argv = apply_config_to_argv(
        [
            "--config-file",
            str(cfg),
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "keyframe_distance_m:=0.2",
        ]
    )
    merged = extract_cli_param_overrides(argv)
    assert merged["use_sim_time"] == "true"
    assert merged["keyframe_distance_m"] == "0.2"


def test_apply_config_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "configuration.env"
    cfg.write_text("fusion_method=odom_only\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INCREMENTAL_VO_CONFIG", raising=False)

    argv = apply_config_to_argv([])
    merged = extract_cli_param_overrides(argv)
    assert merged["fusion_method"] == "odom_only"


def test_apply_config_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "custom.env"
    cfg.write_text("output_root=/data\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INCREMENTAL_VO_CONFIG", str(cfg))

    argv = apply_config_to_argv([])
    merged = extract_cli_param_overrides(argv)
    assert merged["output_root"] == "/data"


def test_apply_config_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INCREMENTAL_VO_CONFIG", raising=False)
    with pytest.raises(FileNotFoundError):
        apply_config_to_argv(["--config-file", str(tmp_path / "missing.env")])


def test_strip_config_file_not_in_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "nope.env"
    cfg.write_text("keyframe_distance_m=0.1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INCREMENTAL_VO_CONFIG", raising=False)
    argv = apply_config_to_argv(["--config-file", str(cfg)])
    assert "--config-file" not in argv

"""Fill empty config fields from environment variables and unambiguous camera discovery."""
from __future__ import annotations

import dataclasses
import os
import types
import typing
from pathlib import Path
from typing import Any, Mapping

import yaml

from robot_infra.devices import Camera, Robot, RobotConfig

_TRUE_STRINGS = {"1", "true", "yes"}


def _scalar_type(annotation: Any) -> type | None:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        annotation = members[0] if len(members) == 1 else None
    return annotation if annotation in (str, int, float, bool) else None


def _cast(value: str, target: type) -> Any:
    if target is bool:
        return value.strip().lower() in _TRUE_STRINGS
    return target(value)


def fill_from_env(config: RobotConfig, env: Mapping[str, str] | None = None) -> RobotConfig:
    env = os.environ if env is None else env
    hints = typing.get_type_hints(type(config))
    updates: dict[str, Any] = {}
    for config_field in dataclasses.fields(config):
        if config_field.name == "arm_backend":
            continue
        if getattr(config, config_field.name) not in (None, ""):
            continue
        target = _scalar_type(hints.get(config_field.name))
        raw = env.get(config_field.name.upper())
        if target is not None and raw is not None:
            updates[config_field.name] = _cast(raw, target)
    updates["cameras"] = [_with_discovered_serial(camera) for camera in config.cameras]
    return dataclasses.replace(config, **updates)


def _with_discovered_serial(camera):
    if camera.serial:
        return camera
    try:
        serials = Camera.backend(camera.backend).discover()
    except ValueError:
        return camera
    return dataclasses.replace(camera, serial=next(iter(serials))) if len(serials) == 1 else camera


def load_config(path: str | Path) -> RobotConfig:
    """Parse a robot yaml into its config without building parts, so an unregistered
    backend is reported by the doctor instead of failing at load time."""
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, Mapping) and "robot" in data:
        data = data["robot"]
    if "type" not in data:
        raise ValueError(f"robot config has no 'type'; registered: {sorted(Robot.types())}")
    return Robot.types()[str(data["type"]).lower()].CONFIG.from_dict(data)


def load_config_with_env(path: str | Path) -> RobotConfig:
    """``load_config`` plus ``fill_from_env``: the one config view shared by daemon, doctor and discovery."""
    return fill_from_env(load_config(path))

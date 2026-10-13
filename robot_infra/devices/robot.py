"""Robot composition: a typed config plus a PartGroup of arm, gripper and cameras."""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import yaml

from robot_infra.devices.arm import Arm
from robot_infra.devices.base import PartGroup
from robot_infra.devices.camera import Camera, CameraConfig
from robot_infra.devices.gripper import Gripper, GripperMode

_SAFETY_BOX_LENGTH = 6


def _check_known_keys(config_cls: type, data: Mapping[str, Any], known: set[str]) -> None:
    for key in data:
        if key not in known:
            raise ValueError(f"unknown key '{key}' for {config_cls.__name__}; known: {sorted(known)}")


@dataclass
class GripperConfig:
    backend: str = "fake"
    max_width_m: float = 0.08
    mode: str = "binary"
    binary_threshold: float = 0.5
    shares_arm_session: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GripperConfig":
        _check_known_keys(cls, data, {f.name for f in dataclasses.fields(cls)})
        return cls(**data)


@dataclass
class RobotConfig:
    """Base config. Subclasses set the ClassVars and add robot-specific fields."""

    TYPE: ClassVar[str] = ""
    NUM_JOINTS: ClassVar[int] = 0
    name: str = "robot"
    arm_backend: str = "fake"
    gripper: GripperConfig | None = field(default_factory=GripperConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    home_joints: list[float] | None = None
    safety_box: list[float] | None = None
    claim: bool = True

    def __post_init__(self) -> None:
        if self.home_joints is not None and len(self.home_joints) != self.NUM_JOINTS:
            raise ValueError(
                f"home_joints must have {self.NUM_JOINTS} entries, got {len(self.home_joints)}"
            )
        if self.safety_box is not None and len(self.safety_box) != _SAFETY_BOX_LENGTH:
            raise ValueError(
                "safety_box must be [xmin, ymin, zmin, xmax, ymax, zmax], "
                f"got {len(self.safety_box)} entries"
            )
        if self.gripper is not None:
            modes = [m.value for m in GripperMode]
            if self.gripper.mode not in modes:
                raise ValueError(f"gripper mode '{self.gripper.mode}' invalid; choose from {modes}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotConfig":
        values = {key: value for key, value in data.items() if key != "type"}
        _check_known_keys(cls, values, {f.name for f in dataclasses.fields(cls)})
        if "gripper" in values and values["gripper"] is not None:
            values["gripper"] = GripperConfig.from_dict(values["gripper"])
        if "cameras" in values:
            values["cameras"] = [CameraConfig(**camera) for camera in values["cameras"]]
        return cls(**values)

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "type": self.TYPE}


class Robot(PartGroup, ABC):
    ROBOT_TYPE: ClassVar[str] = ""
    CONFIG: ClassVar[type[RobotConfig]] = RobotConfig
    _TYPES: ClassVar[dict[str, type["Robot"]]] = {}

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config.name)
        self.config = config
        self.build()

    @abstractmethod
    def build(self) -> None:
        """Add child parts; must not touch hardware."""

    @classmethod
    def register_type(cls, robot_cls: type["Robot"]) -> type["Robot"]:
        key = robot_cls.ROBOT_TYPE.lower()
        if not key:
            raise ValueError(f"{robot_cls.__name__} must set ROBOT_TYPE")
        if key in cls._TYPES:
            raise ValueError(f"robot type '{key}' is already registered")
        if robot_cls.CONFIG.TYPE != robot_cls.ROBOT_TYPE:
            raise ValueError(
                f"{robot_cls.__name__}.CONFIG.TYPE '{robot_cls.CONFIG.TYPE}' "
                f"!= ROBOT_TYPE '{robot_cls.ROBOT_TYPE}'"
            )
        cls._TYPES[key] = robot_cls
        return robot_cls

    @classmethod
    def types(cls) -> dict[str, type["Robot"]]:
        return dict(cls._TYPES)

    @classmethod
    def _type_class(cls, robot_type: str) -> type["Robot"]:
        try:
            return cls._TYPES[robot_type.lower()]
        except KeyError:
            raise ValueError(
                f"unknown robot type '{robot_type}'; registered: {sorted(cls._TYPES)}"
            ) from None

    @classmethod
    def from_config(cls, config: RobotConfig) -> "Robot":
        return cls._type_class(config.TYPE)(config)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Robot":
        if "type" not in data:
            raise ValueError(
                f"robot config has no 'type'; add a 'type' key, e.g. type: franka "
                f"(registered: {sorted(cls._TYPES)})"
            )
        robot_cls = cls._type_class(data["type"])
        return robot_cls(robot_cls.CONFIG.from_dict(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Robot":
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data, Mapping) and "robot" in data:
            data = data["robot"]
        return cls.from_dict(data)

    @property
    def arm(self) -> Arm:
        return self.part("arm")

    @property
    def gripper(self) -> Gripper | None:
        return self.children.get("gripper")

    @property
    def cameras(self) -> dict[str, Camera]:
        group = self.children.get("cameras")
        return dict(group.children) if group is not None else {}

    def describe(self) -> dict:
        return {**super().describe(), "type": self.ROBOT_TYPE}

    def describe_text(self) -> str:
        lines = [f"{self.name} ({self.ROBOT_TYPE})"]
        for name, child in self.children.items():
            lines.append(f"  {name}: {self._part_summary(child)}")
            if isinstance(child, PartGroup):
                lines.extend(
                    f"    {sub_name}: {self._part_summary(sub)}"
                    for sub_name, sub in child.children.items()
                )
        return "\n".join(lines)

    @staticmethod
    def _part_summary(part) -> str:
        return f"{type(part).__name__} [sdk={part.SDK}, connected={part.is_connected}]"

    def _build_standard(self) -> None:
        cfg = self.config
        arm = Arm.backend(cfg.arm_backend).from_config(cfg)
        self.add("arm", arm)
        if cfg.gripper:
            owner = arm if cfg.gripper.shares_arm_session else None
            gripper_cls = Gripper.backend(cfg.gripper.backend)
            self.add("gripper", gripper_cls.from_config(cfg, owner=owner))
        if cfg.cameras:
            group = PartGroup("cameras")
            for camera_cfg in cfg.cameras:
                group.add(camera_cfg.name, Camera.backend(camera_cfg.backend)(camera_cfg))
            self.add("cameras", group)

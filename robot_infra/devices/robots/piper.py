"""AgileX Piper: 6-DoF arm whose gripper shares the arm's CAN session."""
from __future__ import annotations

from dataclasses import dataclass, field

from robot_infra.devices.robot import GripperConfig, Robot, RobotConfig


@dataclass
class PiperConfig(RobotConfig):
    TYPE = "piper"
    NUM_JOINTS = 6
    name: str = "piper"
    arm_backend: str = "pyagxarm"
    can_interface: str = "can0"
    bitrate: int = 1_000_000
    firmware_profile: str | None = None  # default|v183|v188|v189; None probes the arm
    gripper: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(
            backend="pyagxarm", max_width_m=0.07, mode="continuous", shares_arm_session=True
        )
    )
    home_joints: list[float] | None = field(default_factory=lambda: [0.0] * 6)


@Robot.register_type
class PiperRobot(Robot):
    ROBOT_TYPE = "piper"
    CONFIG = PiperConfig

    def build(self) -> None:
        self._build_standard()

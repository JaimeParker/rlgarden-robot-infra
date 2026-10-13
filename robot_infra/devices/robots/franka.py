"""Franka Panda: 7-DoF arm with a binary gripper."""
from __future__ import annotations

from dataclasses import dataclass, field

from robot_infra.devices.robot import GripperConfig, Robot, RobotConfig

FRANKA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
_REALTIME_MODES = ("ignore", "enforce")


@dataclass
class FrankaConfig(RobotConfig):
    TYPE = "franka"
    NUM_JOINTS = 7
    name: str = "franka"
    arm_backend: str = "franky"
    robot_ip: str = "172.16.0.2"
    bridge_url: str = "http://127.0.0.1:5000"
    gripper: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(backend="franky", max_width_m=0.08, mode="binary")
    )
    home_joints: list[float] | None = field(default_factory=lambda: list(FRANKA_HOME_JOINTS))
    relative_dynamics_factor: float = 0.2
    realtime: str = "ignore"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.realtime not in _REALTIME_MODES:
            raise ValueError(f"realtime '{self.realtime}' invalid; choose from {list(_REALTIME_MODES)}")


@Robot.register_type
class FrankaRobot(Robot):
    ROBOT_TYPE = "franka"
    CONFIG = FrankaConfig

    def build(self) -> None:
        self._build_standard()

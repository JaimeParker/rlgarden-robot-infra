"""Franka Hand backend on franky: a libfranka gripper session independent of arm control."""
from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_infra.devices.backends.arms.franky import franky_connect_error, require_franky
from robot_infra.devices.base import Connection
from robot_infra.devices.errors import DeviceFault
from robot_infra.devices.gripper import Gripper, GripperMode

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig

# Wide grasp tolerances make "close on nothing" a successful grasp instead of a failed command.
_GRASP_EPSILON_M = 1.0


@Gripper.register("franky")
class FrankyGripper(Gripper):
    SDK = "franky"

    def __init__(
        self,
        name: str = "gripper",
        *,
        robot_ip: str,
        max_width_m: float = 0.08,
        mode: GripperMode = GripperMode.BINARY,
        binary_threshold: float = 0.5,
        speed: float = 0.05,
        force: float = 20.0,
        owner: Connection | None = None,
    ) -> None:
        super().__init__(
            name,
            max_width_m=max_width_m,
            mode=mode,
            binary_threshold=binary_threshold,
            owner=owner,
        )
        self.robot_ip = robot_ip
        self.speed = speed
        self.force = force

    @classmethod
    def from_config(
        cls, config: "RobotConfig", *, name: str = "gripper", owner: Connection | None = None
    ) -> "FrankyGripper":
        if owner is not None:
            raise ValueError(
                "franky gripper owns its own session; set gripper.shares_arm_session: false"
            )
        gripper = config.gripper
        return cls(
            name,
            robot_ip=config.robot_ip,
            max_width_m=gripper.max_width_m,
            mode=GripperMode(gripper.mode),
            binary_threshold=gripper.binary_threshold,
            owner=owner,
        )

    @classmethod
    def require_sdk(cls) -> ModuleType:
        return require_franky(cls, cls.__name__)

    def _open(self) -> Any:
        self._forget_binary_command()
        franky = self.require_sdk()
        try:
            return franky.Gripper(self.robot_ip)
        except Exception as error:
            raise franky_connect_error(franky, error, self.robot_ip, self.name) from error

    def _release(self, handle: Any) -> None:
        pass

    def open(self) -> None:
        self._require_connected()
        self._handle.open(self.speed)

    def close(self) -> None:
        self._require_connected()
        franky = self.require_sdk()
        try:
            self._handle.grasp(
                0.0,
                self.speed,
                self.force,
                epsilon_inner=_GRASP_EPSILON_M,
                epsilon_outer=_GRASP_EPSILON_M,
            )
        except franky.GripperException:
            pass  # nothing between the fingers: the hand still ends up closed
        except franky.CommandException as error:
            raise DeviceFault(
                f"Franka Hand grasp command failed: {error}",
                hint="check the hand in Desk (homing, errors) and that no other process holds the gripper",
                device=self.name,
            ) from error

    def move(self, width_norm: float) -> None:
        self._require_connected()
        self._handle.move(float(np.clip(width_norm, 0.0, 1.0)) * self._max_width_m, self.speed)

    @property
    def width(self) -> float:
        self._require_connected()
        return float(np.clip(self._handle.width / self._max_width_m, 0.0, 1.0))

    @property
    def is_grasped(self) -> bool | None:
        self._require_connected()
        return bool(self._handle.is_grasped)

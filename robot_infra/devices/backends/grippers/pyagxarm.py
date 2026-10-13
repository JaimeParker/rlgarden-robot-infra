"""AgileX gripper on a Piper, driven through the arm's pyAgxArm effector."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from robot_infra.devices.backends.arms.pyagxarm import AgxArm
from robot_infra.devices.errors import DeviceFault
from robot_infra.devices.gripper import Gripper, GripperMode

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig


@Gripper.register("pyagxarm")
class AgxGripper(Gripper):
    """Borrows the arm's CAN session (owner=arm); the arm must be built with with_gripper=True."""

    SDK = "pyAgxArm"

    def __init__(
        self,
        name: str = "gripper",
        *,
        max_width_m: float = 0.07,
        mode: GripperMode = GripperMode.CONTINUOUS,
        binary_threshold: float = 0.5,
        force_n: float = 1.0,
        owner: AgxArm | None = None,
    ) -> None:
        super().__init__(
            name,
            max_width_m=max_width_m,
            mode=mode,
            binary_threshold=binary_threshold,
            owner=owner,
        )
        self.force_n = force_n

    @classmethod
    def from_config(
        cls, config: "RobotConfig", *, name: str = "gripper", owner: Any = None
    ) -> "AgxGripper":
        if owner is None:
            raise ValueError("pyagxarm gripper requires shares_arm_session: true")
        if not isinstance(owner, AgxArm):
            raise ValueError(
                f"pyagxarm gripper must borrow a pyagxarm arm, got {type(owner).__name__}"
            )
        gripper = config.gripper
        return cls(
            name,
            max_width_m=gripper.max_width_m,
            mode=GripperMode(gripper.mode),
            binary_threshold=gripper.binary_threshold,
            owner=owner,
        )

    def _open(self) -> Any:
        raise RuntimeError(
            "AgxGripper has no session of its own; construct it with owner=<AgxArm>"
        )

    def _release(self, handle: Any) -> None:
        pass

    def _opened(self) -> None:
        super()._opened()
        if not self.owner.with_gripper or self.owner.effector is None:
            raise DeviceFault(
                f"arm '{self.owner.name}' was built without a gripper effector",
                hint="set gripper.shares_arm_session: true and rebuild",
                device=self.name,
            )

    def open(self) -> None:
        self.move(1.0)

    def close(self) -> None:
        self.move(0.0)

    def move(self, width_norm: float) -> None:
        self._require_connected()
        width_m = float(np.clip(width_norm, 0.0, 1.0)) * self._max_width_m
        self.owner.effector.move_gripper_m(width_m, self.force_n)

    @property
    def width(self) -> float:
        self._require_connected()
        status = self.owner.effector.get_gripper_status()
        if status is None:
            raise DeviceFault(
                f"gripper '{self.name}' reports no feedback",
                hint="if the stroke was never configured on the teach pendant the feedback "
                "stays 0 — configure it (0.07 or 0.1 m)",
                device=self.name,
                recoverable=True,
            )
        if status.msg.mode != "width":
            raise DeviceFault(
                f"gripper '{self.name}' reports in '{status.msg.mode}' mode, expected 'width'",
                hint="configure the gripper in width mode",
                device=self.name,
            )
        return float(np.clip(status.msg.value / self._max_width_m, 0.0, 1.0))

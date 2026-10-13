"""AgileX Piper gripper: shares the arm's ``piper_sdk`` CAN session (``owner`` is the arm)."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from robot_infra.devices.base import Connection
from robot_infra.devices.errors import DeviceFault
from robot_infra.devices.gripper import Gripper, GripperMode

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig

_GRIPPER_ENABLE = 0x01
_MILLI_MM_PER_M = 1e6


@Gripper.register("piper_sdk")
class PiperSdkGripper(Gripper):
    """Borrows the arm's CAN session. ``GripperCtrl`` must carry the enable code on every
    call, otherwise the vendor firmware disables the gripper."""

    SDK = "piper_sdk"
    STUCK_GRACE_S = 1.0

    def __init__(
        self,
        name: str = "gripper",
        *,
        max_width_m: float = 0.07,
        mode: GripperMode = GripperMode.CONTINUOUS,
        binary_threshold: float = 0.5,
        effort: int = 1000,
        owner: Connection | None = None,
    ) -> None:
        super().__init__(
            name,
            max_width_m=max_width_m,
            mode=mode,
            binary_threshold=binary_threshold,
            owner=owner,
        )
        self._effort = int(effort)
        self._nonzero_command_at: float | None = None

    @classmethod
    def from_config(
        cls, config: "RobotConfig", *, name: str = "gripper", owner: Connection | None = None
    ) -> "PiperSdkGripper":
        if owner is None:
            raise ValueError("piper_sdk gripper requires shares_arm_session: true")
        gripper = config.gripper
        return cls(
            name,
            max_width_m=gripper.max_width_m,
            mode=GripperMode(gripper.mode),
            binary_threshold=gripper.binary_threshold,
            owner=owner,
        )

    def _open(self) -> Any:
        raise RuntimeError("piper_sdk gripper borrows the arm's session and cannot open its own")

    def _release(self, handle: Any) -> None:
        pass

    def open(self) -> None:
        self.move(1.0)

    def close(self) -> None:
        self.move(0.0)

    def move(self, width_norm: float) -> None:
        self._require_connected()
        width_norm = min(max(float(width_norm), 0.0), 1.0)
        stroke = int(round(width_norm * self._max_width_m * _MILLI_MM_PER_M))
        self.owner._handle.GripperCtrl(stroke, self._effort, _GRIPPER_ENABLE, 0)
        self._nonzero_command_at = time.monotonic() if stroke else None

    @property
    def width(self) -> float:
        self._require_connected()
        angle = self.owner._handle.GetArmGripperMsgs().gripper_state.grippers_angle
        if angle == 0 and self._nonzero_command_at is not None:
            if time.monotonic() - self._nonzero_command_at >= self.STUCK_GRACE_S:
                raise DeviceFault(
                    "gripper feedback stays 0 after a non-zero command",
                    hint="gripper feedback stays 0: configure the gripper stroke (0.07 or 0.1 m) "
                    "on the teach pendant",
                    device=self.name,
                    recoverable=True,
                )
        full_stroke = self._max_width_m * _MILLI_MM_PER_M
        return min(max(angle / full_stroke, 0.0), 1.0)

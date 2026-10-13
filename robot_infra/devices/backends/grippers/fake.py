"""Hardware-free gripper that tracks a normalized width."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_infra.devices.base import Connection
from robot_infra.devices.gripper import Gripper, GripperMode

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig


@Gripper.register("fake")
class FakeGripper(Gripper):
    def __init__(
        self,
        name: str = "gripper",
        *,
        max_width_m: float = 0.08,
        mode: GripperMode = GripperMode.BINARY,
        binary_threshold: float = 0.5,
        owner: Connection | None = None,
    ) -> None:
        super().__init__(
            name,
            max_width_m=max_width_m,
            mode=mode,
            binary_threshold=binary_threshold,
            owner=owner,
        )
        self._width = 1.0
        self.call_log: list[str] = []

    @classmethod
    def from_config(
        cls, config: "RobotConfig", *, name: str = "gripper", owner: Connection | None = None
    ) -> "FakeGripper":
        gripper = config.gripper
        return cls(
            name,
            max_width_m=gripper.max_width_m,
            mode=GripperMode(gripper.mode),
            binary_threshold=gripper.binary_threshold,
            owner=owner,
        )

    def open(self) -> None:
        self.call_log.append("open")
        self._width = 1.0

    def close(self) -> None:
        self.call_log.append("close")
        self._width = 0.0

    def move(self, width_norm: float) -> None:
        self.call_log.append("move")
        self._width = float(width_norm)

    @property
    def width(self) -> float:
        return self._width

    def _open(self) -> Any:
        return {"opened": True}

    def _release(self, handle: Any) -> None:
        pass

    def _opened(self) -> None:
        self.call_log.append("opened")

    def _closing(self) -> None:
        self.call_log.append("closing")

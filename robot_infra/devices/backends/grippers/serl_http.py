"""Legacy Franka gripper: binary open/close through the ROS1 bridge."""
from __future__ import annotations

from typing import TYPE_CHECKING

from robot_infra.devices.backends.arms.serl_http import _BridgeHandle
from robot_infra.devices.base import Connection
from robot_infra.devices.gripper import Gripper, GripperMode

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig


@Gripper.register("serl_http")
class SerlHttpGripper(Gripper):
    """Binary gripper behind the serl_franka_controllers Flask bridge."""

    SDK = None

    def __init__(
        self,
        name: str = "gripper",
        *,
        bridge_url: str,
        timeout_s: float = 5.0,
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
        self._bridge_url = bridge_url
        self._timeout_s = timeout_s

    @classmethod
    def from_config(
        cls, config: "RobotConfig", *, name: str = "gripper", owner: Connection | None = None
    ) -> "SerlHttpGripper":
        if owner is not None:
            raise ValueError(
                "serl_http gripper owns its own session; set gripper.shares_arm_session: false"
            )
        gripper = config.gripper
        return cls(
            name,
            bridge_url=config.bridge_url,
            max_width_m=gripper.max_width_m,
            mode=GripperMode(gripper.mode),
            binary_threshold=gripper.binary_threshold,
            owner=owner,
        )

    def _open(self) -> _BridgeHandle:
        self._forget_binary_command()
        handle = _BridgeHandle(self._bridge_url, self._timeout_s, self.name)
        handle.post("/getstate")
        return handle

    def _release(self, handle: _BridgeHandle) -> None:
        pass

    def open(self) -> None:
        self._require_connected()
        self._handle.post("/open_gripper")

    def close(self) -> None:
        self._require_connected()
        self._handle.post("/close_gripper")

    def move(self, width_norm: float) -> None:
        raise NotImplementedError("serl_http gripper is binary (open/close only)")

    @property
    def width(self) -> float:
        self._require_connected()
        return float(self._handle.post("/getstate")["gripper_pos"])

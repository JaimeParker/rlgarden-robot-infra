"""Deterministic first-order arm for hardware-free tests."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from robot_infra.devices.arm import Arm, ControlMode
from robot_infra.devices.base import Connection
from robot_infra.devices.errors import DeviceFault
from robot_infra.devices.safety import ArmSafety

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig

_FAULTS = (None, "no_feedback", "fault", "drop_commands")


@Arm.register("fake")
class FakeArm(Arm):
    """Deterministic first-order arm for tests. Each read_state() moves the state a fraction `alpha` toward the current target."""

    def __init__(
        self,
        name: str = "arm",
        *,
        num_joints: int = 7,
        alpha: float = 0.5,
        owner: Connection | None = None,
        safety: ArmSafety | None = None,
    ) -> None:
        super().__init__(name, num_joints=num_joints, owner=owner, safety=safety)
        self._alpha = alpha
        self._joint_pos = np.zeros(num_joints)
        self._joint_vel = np.zeros(num_joints)
        self._tcp_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self._joint_target: np.ndarray | None = None
        self._pose_target: np.ndarray | None = None
        self._injected_fault: str | None = None
        self.call_log: list[str] = []

    @classmethod
    def from_config(cls, config: "RobotConfig", *, name: str = "arm") -> "FakeArm":
        return cls(name, num_joints=config.NUM_JOINTS, safety=ArmSafety.from_config(config))

    def inject(self, fault: str | None) -> None:
        if fault not in _FAULTS:
            raise ValueError(f"unknown fault '{fault}'; known: {_FAULTS}")
        self._injected_fault = fault

    def read_state(self) -> dict[str, np.ndarray]:
        if self._joint_target is not None:
            delta = self._alpha * (self._joint_target - self._joint_pos)
            self._joint_pos = self._joint_pos + delta
            self._joint_vel = delta
        if self._pose_target is not None:
            self._tcp_pose[:3] += self._alpha * (self._pose_target[:3] - self._tcp_pose[:3])
            self._tcp_pose[3:] = self._pose_target[3:]
        if self._injected_fault != "no_feedback":
            self._note_feedback()
        return self._remember_state({
            "tcp_pose": self._tcp_pose.copy(),
            "tcp_vel": np.zeros(6),
            "joint_pos": self._joint_pos.copy(),
            "joint_vel": self._joint_vel.copy(),
            "tcp_force": np.zeros(3),
            "tcp_torque": np.zeros(3),
        })

    def _servo_cartesian(self, pose: np.ndarray) -> None:
        self.call_log.append("servo_cartesian")
        self._raise_if_faulted()
        pose = self._checked(pose, (7,))
        if self._injected_fault != "drop_commands":
            self._pose_target = pose

    def _servo_joints(self, q: np.ndarray) -> None:
        self.call_log.append("servo_joints")
        self._raise_if_faulted()
        q = self._checked(q, (self.num_joints,))
        if self._injected_fault != "drop_commands":
            self._joint_target = q

    def _move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self.call_log.append("move_to_pose")
        self._raise_if_faulted()
        self._pose_target = self._checked(pose, (7,))
        if blocking:
            self._tcp_pose = self._pose_target.copy()

    def _move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self.call_log.append("move_to_joints")
        self._raise_if_faulted()
        self._joint_target = self._checked(q, (self.num_joints,))
        if blocking:
            self._joint_pos = self._joint_target.copy()
            self._joint_vel = np.zeros(self.num_joints)

    def stop(self) -> None:
        self.call_log.append("stop")
        self._joint_target = None
        self._pose_target = None

    def recover(self) -> None:
        self.call_log.append("recover")
        if self._injected_fault == "fault":
            self._injected_fault = None
        self._clear_errors()

    def _enter_control_mode(self, mode: ControlMode) -> None:
        self.call_log.append(f"enter:{mode.value}")

    def _open(self) -> Any:
        return {"opened": True}

    def _release(self, handle: Any) -> None:
        pass

    def _raise_if_faulted(self) -> None:
        if self._injected_fault != "fault":
            return
        message = f"fake arm '{self.name}' fault"
        if message not in self._errors:
            self._note_error(message)
        raise DeviceFault(
            message, hint="call recover() to clear the fault", device=self.name, recoverable=True
        )

    @staticmethod
    def _checked(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"expected shape {shape}, got {value.shape}")
        return value.copy()

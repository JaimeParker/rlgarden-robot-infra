"""Arm contract: state read, Cartesian/joint servoing, control modes, health."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from robot_infra.devices.base import Connection, ControllablePart, Features, Health
from robot_infra.devices.errors import DeviceFault
from robot_infra.devices.registry import BackendRegistry

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig


class ControlMode(Enum):
    IDLE = "idle"
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


ARM_STATE_FIELDS = ("tcp_pose", "tcp_vel", "joint_pos", "joint_vel", "tcp_force", "tcp_torque")


def arm_state_shape(field: str, num_joints: int) -> tuple[int, ...]:
    shapes = {
        "tcp_pose": (7,),
        "tcp_vel": (6,),
        "joint_pos": (num_joints,),
        "joint_vel": (num_joints,),
        "tcp_force": (3,),
        "tcp_torque": (3,),
    }
    try:
        return shapes[field]
    except KeyError:
        raise ValueError(f"unknown arm state field '{field}'; known: {list(ARM_STATE_FIELDS)}") from None


@dataclass
class ArmHealth(Health):
    first_feedback_received: bool = False
    errors: list[str] = field(default_factory=list)
    robot_mode: str = ""
    control_mode: str = ControlMode.IDLE.value


class Arm(ControllablePart, BackendRegistry, ABC):
    _BACKENDS: ClassVar[dict[str, type]] = {}
    SUPPORTED_STATE_FIELDS: ClassVar[tuple[str, ...]] = ARM_STATE_FIELDS

    def __init__(self, name: str, *, num_joints: int, owner: Connection | None = None) -> None:
        super().__init__(name, owner=owner)
        self._num_joints = num_joints
        self._control_mode = ControlMode.IDLE
        self._last_feedback_time: float | None = None
        self._errors: list[str] = []
        self._robot_mode = ""

    @classmethod
    def from_config(cls, config: "RobotConfig", *, name: str = "arm") -> "Arm":
        raise NotImplementedError(f"{cls.__name__} does not implement from_config")

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @property
    def observation_features(self) -> Features:
        return {
            f: {"shape": arm_state_shape(f, self._num_joints), "dtype": "float32"}
            for f in self.SUPPORTED_STATE_FIELDS
        }

    def get_observation(self) -> dict[str, np.ndarray]:
        self._require_connected()
        state = self.read_state()
        observation = {}
        for state_field in self.SUPPORTED_STATE_FIELDS:
            if state_field not in state:
                raise KeyError(
                    f"state field '{state_field}' missing from {type(self).__name__}.read_state()"
                )
            observation[state_field] = np.asarray(state[state_field], dtype=np.float32)
        return observation

    @property
    def action_features(self) -> Features:
        if self._control_mode is ControlMode.CARTESIAN_SERVO:
            return {"tcp_pose": {"shape": (7,), "dtype": "float32"}}
        if self._control_mode is ControlMode.JOINT_SERVO:
            return {"joint_pos": {"shape": (self._num_joints,), "dtype": "float32"}}
        return {}

    def send_action(self, action: dict[str, np.ndarray]) -> None:
        self._require_connected()
        if self._control_mode is ControlMode.IDLE:
            raise DeviceFault(
                f"arm '{self.name}' is idle",
                hint="call set_control_mode(ControlMode.CARTESIAN_SERVO) or "
                "set_control_mode(ControlMode.JOINT_SERVO) before sending actions",
                device=self.name,
            )
        if self._control_mode is ControlMode.CARTESIAN_SERVO:
            pose = self._command_array(action, "tcp_pose", (7,))
            self.servo_cartesian(pose)
        else:
            q = self._command_array(action, "joint_pos", (self._num_joints,))
            self.servo_joints(q)

    def _command_array(
        self, action: dict[str, np.ndarray], key: str, shape: tuple[int, ...]
    ) -> np.ndarray:
        if key not in action:
            raise ValueError(
                f"arm '{self.name}' in {self._control_mode.value} mode expects action key '{key}'"
            )
        value = np.asarray(action[key], dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"action '{key}' must have shape {shape}, got {value.shape}")
        return value

    def set_control_mode(self, mode: ControlMode) -> None:
        if mode == self._control_mode:
            return
        self._require_connected()
        self.stop()
        self._enter_control_mode(mode)
        self._control_mode = mode

    @abstractmethod
    def read_state(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def servo_cartesian(self, pose: np.ndarray) -> None: ...

    @abstractmethod
    def servo_joints(self, q: np.ndarray) -> None: ...

    @abstractmethod
    def move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None: ...

    @abstractmethod
    def move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def recover(self) -> None: ...

    @abstractmethod
    def _enter_control_mode(self, mode: ControlMode) -> None: ...

    def health(self) -> ArmHealth:
        age = (
            None
            if self._last_feedback_time is None
            else time.monotonic() - self._last_feedback_time
        )
        return ArmHealth(
            ok=self.is_connected and not self._errors,
            last_feedback_age_s=age,
            first_feedback_received=self._last_feedback_time is not None,
            errors=list(self._errors),
            robot_mode=self._robot_mode,
            control_mode=self._control_mode.value,
        )

    def _note_feedback(self) -> None:
        self._last_feedback_time = time.monotonic()

    def _note_error(self, message: str) -> None:
        self._errors.append(message)

    def _clear_errors(self) -> None:
        self._errors.clear()

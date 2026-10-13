"""Arm contract: state read, Cartesian/joint servoing, control modes, health."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from robot_infra.devices.base import Connection, ControllablePart, Features, Health
from robot_infra.devices.errors import DeviceFault, SafetyViolation
from robot_infra.devices.registry import BackendRegistry
from robot_infra.devices.safety import ArmSafety

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig
    from robot_infra.doctor.checks import CheckResult


class ControlMode(Enum):
    IDLE = "idle"
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


QUAT_NORM_TOLERANCE = 1e-3
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
    safety_clips: int = 0
    safety_rejects: int = 0


class Arm(ControllablePart, BackendRegistry, ABC):
    _BACKENDS: ClassVar[dict[str, type]] = {}
    SUPPORTED_STATE_FIELDS: ClassVar[tuple[str, ...]] = ARM_STATE_FIELDS
    # A cached read_state() older than this is re-read before limiting a servo step.
    MAX_DELTA_STATE_AGE_S: ClassVar[float] = 0.5

    def __init__(
        self,
        name: str,
        *,
        num_joints: int,
        owner: Connection | None = None,
        safety: ArmSafety | None = None,
    ) -> None:
        super().__init__(name, owner=owner)
        if safety is not None and safety.joint_limits is not None and len(safety.joint_limits) != num_joints:
            raise ValueError(
                f"joint_limits has {len(safety.joint_limits)} rows, arm '{name}' has {num_joints} joints"
            )
        self._num_joints = num_joints
        self._control_mode = ControlMode.IDLE
        self._last_feedback_time: float | None = None
        self._errors: list[str] = []
        self._robot_mode = ""
        self._safety = safety
        self._safety_clips = 0
        self._safety_rejects = 0
        self._last_state: dict[str, np.ndarray] | None = None
        self._last_state_time: float | None = None

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
    def safety_limits(self) -> dict | None:
        return None if self._safety is None else self._safety.to_dict()

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
        if key == "tcp_pose":
            return self._sanitize_pose(action[key])
        return self._sanitize_joints(action[key])

    # -- command templates: sanitize -> safety guard -> backend ------------------------

    def servo_cartesian(self, pose: np.ndarray) -> None:
        self._require_connected()
        pose = self._sanitize_pose(pose)
        if self._safety is not None:
            stepped = False
            if self._safety.max_delta_xyz_m is not None:
                current = self._current_state_field("tcp_pose")
                if current is not None:
                    pose, stepped = self._safety.limit_step_pose(pose, current)
            pose, boxed = self._safety.clip_pose(pose)
            if stepped or boxed:
                self._safety_clips += 1
        self._servo_cartesian(pose)

    def servo_joints(self, q: np.ndarray) -> None:
        self._require_connected()
        q = self._sanitize_joints(q)
        if self._safety is not None:
            stepped = False
            if self._safety.max_delta_joint_rad is not None:
                current = self._current_state_field("joint_pos")
                if current is not None:
                    q, stepped = self._safety.limit_step_joints(q, current)
            q, limited = self._safety.clip_joints(q)
            if stepped or limited:
                self._safety_clips += 1
        self._servo_joints(q)

    def move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self._require_connected()
        pose = self._sanitize_pose(pose)
        if self._safety is not None:
            try:
                self._safety.require_pose_inside(pose, device=self.name)
            except SafetyViolation:
                self._safety_rejects += 1
                raise
        try:
            self._move_to_pose(pose, blocking=blocking)
        finally:
            self._forget_state()

    def move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self._require_connected()
        q = self._sanitize_joints(q)
        if self._safety is not None:
            try:
                self._safety.require_joints_inside(q, device=self.name)
            except SafetyViolation:
                self._safety_rejects += 1
                raise
        try:
            self._move_to_joints(q, blocking=blocking)
        finally:
            self._forget_state()

    def _sanitize_pose(self, pose: np.ndarray) -> np.ndarray:
        value = self._sanitize("tcp_pose", pose, (7,))
        quat_norm = float(np.linalg.norm(value[3:]))
        if abs(quat_norm - 1.0) > QUAT_NORM_TOLERANCE:
            raise ValueError(
                f"'tcp_pose' quaternion norm is {quat_norm:.6f}, expected 1 "
                f"(tolerance {QUAT_NORM_TOLERANCE:g})"
            )
        return np.concatenate([value[:3], value[3:] / quat_norm])

    def _sanitize_joints(self, q: np.ndarray) -> np.ndarray:
        return self._sanitize("joint_pos", q, (self._num_joints,))

    @staticmethod
    def _sanitize(key: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"'{key}' must have shape {shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"'{key}' contains non-finite values: {value.tolist()}")
        return value

    def _current_state_field(self, state_field: str) -> np.ndarray | None:
        """Measured ``state_field`` for step limiting: the cached state while it is fresh, else a
        new read_state(). None when the backend does not report the field."""
        if state_field not in self.SUPPORTED_STATE_FIELDS:
            return None
        state = self._last_state
        if (
            state is None
            or self._last_state_time is None
            or time.monotonic() - self._last_state_time > self.MAX_DELTA_STATE_AGE_S
        ):
            state = self.read_state()
        return np.asarray(state[state_field], dtype=np.float64)

    def _remember_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Backends return their read_state() result through this so servo step limits can reuse it."""
        self._last_state = state
        self._last_state_time = time.monotonic()
        return state

    def _forget_state(self) -> None:
        self._last_state = None
        self._last_state_time = None

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
    def _servo_cartesian(self, pose: np.ndarray) -> None: ...

    @abstractmethod
    def _servo_joints(self, q: np.ndarray) -> None: ...

    @abstractmethod
    def _move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None: ...

    @abstractmethod
    def _move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None: ...

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
            safety_clips=self._safety_clips,
            safety_rejects=self._safety_rejects,
        )

    def preflight(self) -> list["CheckResult"]:
        """Backend-specific mode checks run by the doctor after first feedback (FCI active, RT kernel, follower mode ...). Default: none."""
        return []

    def _note_feedback(self) -> None:
        self._last_feedback_time = time.monotonic()

    def _note_error(self, message: str) -> None:
        self._errors.append(message)

    def _clear_errors(self) -> None:
        self._errors.clear()

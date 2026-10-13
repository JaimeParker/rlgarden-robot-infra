"""Gripper contract: normalized width in [0, 1], binary or continuous commands."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from robot_infra.devices.base import Connection, ControllablePart, Features
from robot_infra.devices.registry import BackendRegistry

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig


class GripperMode(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"


class Gripper(ControllablePart, BackendRegistry, ABC):
    _BACKENDS: ClassVar[dict[str, type]] = {}
    closed_action_value: ClassVar[float] = 0.0

    def __init__(
        self,
        name: str,
        *,
        max_width_m: float,
        mode: GripperMode = GripperMode.BINARY,
        binary_threshold: float = 0.5,
        owner: Connection | None = None,
    ) -> None:
        super().__init__(name, owner=owner)
        self._max_width_m = max_width_m
        self._mode = mode
        self._binary_threshold = binary_threshold

    @classmethod
    def from_config(
        cls, config: "RobotConfig", *, name: str = "gripper", owner: Connection | None = None
    ) -> "Gripper":
        raise NotImplementedError(f"{cls.__name__} does not implement from_config")

    @property
    def max_width_m(self) -> float:
        return self._max_width_m

    @property
    def mode(self) -> GripperMode:
        return self._mode

    @property
    def observation_features(self) -> Features:
        return {"gripper_pos": {"shape": (1,), "dtype": "float32"}}

    def get_observation(self) -> dict[str, np.ndarray]:
        self._require_connected()
        return {"gripper_pos": np.asarray([self.width], dtype=np.float32)}

    @property
    def action_features(self) -> Features:
        return {"gripper": {"shape": (1,), "dtype": "float32"}}

    def send_action(self, action: dict[str, np.ndarray]) -> None:
        self._require_connected()
        target = float(np.asarray(action["gripper"]).reshape(-1)[0])
        if self._mode is GripperMode.BINARY:
            if target > self._binary_threshold:
                self.open()
            else:
                self.close()
        else:
            self.move(float(np.clip(target, 0.0, 1.0)))

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def move(self, width_norm: float) -> None: ...

    @property
    @abstractmethod
    def width(self) -> float: ...

    @property
    def is_grasped(self) -> bool | None:
        return None

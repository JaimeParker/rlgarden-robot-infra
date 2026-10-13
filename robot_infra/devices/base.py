"""Device abstractions: lifecycle-managed ``Connection``, observable ``Part``,
``ControllablePart``, and the ``PartGroup`` tree."""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar, TypedDict, Union, final

import numpy as np

from robot_infra.devices.errors import DeviceNotConnected, SdkMissing

FeatureSpec = TypedDict("FeatureSpec", {"shape": tuple[int, ...], "dtype": str})
Features = dict[str, Union[FeatureSpec, "Features"]]


@dataclass
class Health:
    ok: bool
    detail: str = ""
    last_feedback_age_s: float | None = None


class Connection(ABC):
    SDK: ClassVar[str | None] = None

    def __init__(self, name: str, *, owner: "Connection | None" = None) -> None:
        self.name = name
        self.owner = owner
        self._handle: Any = None
        self._borrowing = False

    @classmethod
    def require_sdk(cls) -> ModuleType:
        if cls.SDK is None:
            raise RuntimeError(f"{cls.__name__} declares no vendor SDK")
        try:
            return importlib.import_module(cls.SDK)
        except ImportError:
            raise SdkMissing(
                f"{cls.__name__} needs the '{cls.SDK}' package",
                hint=f"pip install {cls.SDK}",
                device=cls.__name__,
            ) from None

    @classmethod
    def discover(cls) -> set[str]:
        return set()

    @final
    def connect(self) -> None:
        if self.is_connected:
            return
        if self.owner is not None:
            self.owner.connect()
            self._opened()
            self._borrowing = True
        else:
            self._handle = self._open()

    @final
    def disconnect(self) -> None:
        if self.owner is not None:
            if self._borrowing:
                self._closing()
                self._borrowing = False
            return
        handle, self._handle = self._handle, None
        if handle is not None:
            self._release(handle)

    @property
    def is_connected(self) -> bool:
        if self.owner is not None:
            return self.owner.is_connected and self._borrowing
        return self._handle is not None

    @abstractmethod
    def _open(self) -> Any: ...

    @abstractmethod
    def _release(self, handle: Any) -> None: ...

    def _opened(self) -> None:
        pass

    def _closing(self) -> None:
        pass

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnected(
                f"'{self.name}' is not connected",
                hint=f"call connect() on '{self.name}' first",
                device=self.name,
            )

    def health(self) -> Health:
        return Health(ok=self.is_connected)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "sdk": self.SDK,
            "owner": self.owner.name if self.owner is not None else None,
            "connected": self.is_connected,
        }


class Part(Connection, ABC):
    @property
    @abstractmethod
    def observation_features(self) -> Features: ...

    @abstractmethod
    def get_observation(self) -> dict[str, np.ndarray]: ...


class ControllablePart(Part, ABC):
    @property
    @abstractmethod
    def action_features(self) -> Features: ...

    @abstractmethod
    def send_action(self, action: dict[str, np.ndarray]) -> None: ...


class PartGroup(ControllablePart):
    """Named part tree. Children connect in insertion order and disconnect in reverse.
    A child borrowing from a sibling owner connects that owner on demand, so add owners
    before borrowers to keep the reverse-order release correct."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.children: dict[str, Part] = {}

    def add(self, name: str, part: Part) -> Part:
        if name in self.children:
            raise ValueError(f"part '{name}' already exists in group '{self.name}'")
        self.children[name] = part
        return part

    def part(self, name: str) -> Part:
        try:
            return self.children[name]
        except KeyError:
            raise KeyError(
                f"no part '{name}' in group '{self.name}'; available: {list(self.children)}"
            ) from None

    @property
    def observation_features(self) -> Features:
        return {name: child.observation_features for name, child in self.children.items()}

    def get_observation(self) -> dict[str, np.ndarray]:
        return {name: child.get_observation() for name, child in self.children.items()}

    @property
    def action_features(self) -> Features:
        return {
            name: child.action_features
            for name, child in self.children.items()
            if isinstance(child, ControllablePart)
        }

    def send_action(self, action: dict[str, np.ndarray]) -> None:
        controllable = [
            name for name, child in self.children.items() if isinstance(child, ControllablePart)
        ]
        for key in action:
            if key not in controllable:
                raise ValueError(
                    f"unknown part '{key}' in group '{self.name}'; controllable: {controllable}"
                )
        for key, sub_action in action.items():
            self.children[key].send_action(sub_action)

    def _open(self) -> list[Part]:
        try:
            for child in self.children.values():
                child.connect()
        except BaseException:
            try:
                self._disconnect_children_reversed()
            except Exception:
                pass
            raise
        return list(self.children.values())

    def _release(self, handle: list[Part]) -> None:
        self._disconnect_children_reversed()

    def _disconnect_children_reversed(self) -> None:
        last_error: BaseException | None = None
        for child in reversed(list(self.children.values())):
            try:
                child.disconnect()
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise last_error

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "children": {name: child.describe() for name, child in self.children.items()},
        }

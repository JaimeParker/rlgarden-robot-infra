"""Transport abstraction: a server exposes a ``RobotService``; a client calls its methods
by name. Implementations differ only in how ``codec`` bytes move."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

from robot_infra.devices.errors import DeviceFault

if TYPE_CHECKING:
    from robot_infra.server.service import RobotService

READ_ONLY_METHODS = frozenset(
    {
        "robot.describe",
        "robot.health",
        "robot.get_observation",
        "arm.read_state",
        "arm.control_mode",
        "arm.health",
        "arm.preflight",
        "gripper.width",
    }
)


def is_read_only(method: str) -> bool:
    """Read-only methods are safe to retry after a reconnect; commands are not."""
    if method in READ_ONLY_METHODS:
        return True
    parts = method.split(".")
    return len(parts) == 3 and parts[0] == "camera" and parts[2] == "read"


class TransportError(DeviceFault):
    kind = "transport"


class Transport(ABC):
    """Server side."""

    @abstractmethod
    def serve(self, service: "RobotService", bind: str, *, token: str | None = None) -> None:
        """Serve until ``stop()`` is called (blocking)."""

    @abstractmethod
    def start(self, service: "RobotService", bind: str, *, token: str | None = None) -> None:
        """Serve in the background and return immediately."""

    @abstractmethod
    def stop(self, grace_s: float = 1.0) -> None: ...


class TransportClient(ABC):
    _last_rtt_ms: float | None = None

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    @abstractmethod
    def call(self, method: str, **kwargs: Any) -> Any:
        """Raise the remote ``DeviceError`` subclass re-typed, or ``TransportError``."""

    @abstractmethod
    def subscribe(self, method: str, rate_hz: float) -> Iterator[Any]:
        """Yield ``call(method)`` results at ``rate_hz`` until the client is closed."""

    @property
    def last_rtt_ms(self) -> float | None:
        return self._last_rtt_ms

    def __enter__(self) -> "TransportClient":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

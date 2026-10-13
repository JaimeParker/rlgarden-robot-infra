"""In-process transport for tests and single-machine use; optionally mimics the wire by
round-tripping kwargs and results through the codec."""
from __future__ import annotations

import threading
import time
from typing import Any, Iterator

from robot_infra.server.codec import pack, unpack
from robot_infra.server.service import RobotService
from robot_infra.server.transport.base import Transport, TransportClient, TransportError


class LocalTransport(Transport):
    def __init__(self) -> None:
        self.service: RobotService | None = None

    def serve(self, service: RobotService, bind: str = "local", *, token: str | None = None) -> None:
        self.service = service

    def start(self, service: RobotService, bind: str = "local", *, token: str | None = None) -> None:
        self.service = service

    def stop(self, grace_s: float = 1.0) -> None:
        pass


class LocalClient(TransportClient):
    def __init__(self, service: RobotService, *, roundtrip_codec: bool = True) -> None:
        self._service = service
        self._roundtrip_codec = roundtrip_codec
        self._closed = threading.Event()
        self._closed.set()

    def open(self) -> None:
        self._closed.clear()

    def close(self) -> None:
        self._closed.set()

    @property
    def is_open(self) -> bool:
        return not self._closed.is_set()

    def call(self, method: str, **kwargs: Any) -> Any:
        if not self.is_open:
            raise TransportError(
                f"local client is closed; cannot call '{method}'",
                hint="call open() on the client first",
                device="local",
            )
        start = time.monotonic()
        try:
            if self._roundtrip_codec:
                kwargs = unpack(pack(kwargs))
            result = self._service.dispatch(method, kwargs)
            if self._roundtrip_codec:
                result = unpack(pack(result))
            return result
        finally:
            self._last_rtt_ms = (time.monotonic() - start) * 1000.0

    def subscribe(self, method: str, rate_hz: float) -> Iterator[Any]:
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        period = 1.0 / rate_hz
        deadline = time.monotonic()
        while self.is_open:
            yield self.call(method)
            deadline = max(deadline + period, time.monotonic())
            if self._closed.wait(max(0.0, deadline - time.monotonic())):
                return

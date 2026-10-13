"""Camera contract: a background capture thread feeds read()/latest(); backends only
implement device open/close and a blocking single-frame capture."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, final

import numpy as np

from robot_infra.devices.base import Features, Health, Part
from robot_infra.devices.errors import DeviceFault
from robot_infra.devices.registry import BackendRegistry

_THREAD_JOIN_TIMEOUT_S = 2.0


@dataclass
class CameraConfig:
    name: str
    serial: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = False
    backend: str = "realsense"


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray | None
    timestamp: float


class Camera(Part, BackendRegistry, ABC):
    _BACKENDS: ClassVar[dict[str, type]] = {}

    def __init__(
        self,
        config: CameraConfig,
        *,
        read_timeout_s: float = 2.0,
        max_consecutive_failures: int = 10,
    ) -> None:
        super().__init__(config.name, owner=None)
        self._config = config
        self._read_timeout_s = read_timeout_s
        self._max_consecutive_failures = max_consecutive_failures
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._last_capture_error = ""
        self._caller_state = threading.local()
        # Serializes open/release/reopen: a read() timeout may reopen from another thread.
        self._lifecycle_lock = threading.RLock()
        self._capture_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._live_device: Any = None

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def observation_features(self) -> Features:
        c = self._config
        features: Features = {"color": {"shape": (c.height, c.width, 3), "dtype": "uint8"}}
        if c.enable_depth:
            features["depth"] = {"shape": (c.height, c.width), "dtype": "float32"}
        return features

    def get_observation(self) -> dict[str, np.ndarray]:
        frame = self.read()
        observation = {"color": frame.color}
        if self._config.enable_depth:
            observation["depth"] = frame.depth
        return observation

    @final
    def _open(self) -> Any:
        with self._lifecycle_lock:
            handle = self._open_device()
            self._live_device = handle
            self._stop_event = threading.Event()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(handle, self._stop_event),
                name=f"camera-{self.name}",
                daemon=True,
            )
            self._capture_thread.start()
            return handle

    @final
    def _release(self, handle: Any) -> None:
        # Closes whatever device is live: a concurrent reopen() may have replaced ``handle``.
        with self._lifecycle_lock:
            self._close_live_device()
            self._handle = None

    def _close_live_device(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        self._capture_thread = None
        self._stop_event = None
        device, self._live_device = self._live_device, None
        if device is not None:
            self._close_device(device)

    @abstractmethod
    def _open_device(self) -> Any: ...

    @abstractmethod
    def _close_device(self, handle: Any) -> None: ...

    @abstractmethod
    def _capture_frame(self, handle: Any) -> Frame: ...

    def _capture_loop(self, handle: Any, stop_event: threading.Event) -> None:
        failures = 0
        while not stop_event.is_set():
            try:
                frame = self._capture_frame(handle)
            except Exception as error:
                failures += 1
                self._last_capture_error = str(error)
                if failures >= self._max_consecutive_failures:
                    break
                continue
            if stop_event.is_set():
                break
            failures = 0
            with self._condition:
                self._latest = frame
                self._seq += 1
                self._condition.notify_all()

    def read(self, timeout_s: float | None = None) -> Frame:
        self._require_connected()
        session = self._handle
        timeout = self._read_timeout_s if timeout_s is None else timeout_s
        frame = self._wait_for_new_frame(timeout)
        if frame is None:
            # Two readers timing out together reopen once: the second sees a newer session.
            self._reopen_session(session)
            frame = self._wait_for_new_frame(timeout)
        if frame is None:
            raise DeviceFault(
                f"camera '{self.name}' produced no frame within {timeout:.1f}s",
                hint="check the USB cable/bandwidth and run `python -m robot_infra.doctor`",
                device=self.name,
                recoverable=True,
            )
        return frame

    def _wait_for_new_frame(self, timeout_s: float) -> Frame | None:
        last_returned = getattr(self._caller_state, "last_seq", 0)
        with self._condition:
            if not self._condition.wait_for(lambda: self._seq > last_returned, timeout=timeout_s):
                return None
            self._caller_state.last_seq = self._seq
            return self._latest

    def latest(self) -> Frame | None:
        with self._condition:
            return self._latest

    def is_ready(self, timeout_s: float = 2.0) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: self._latest is not None, timeout=timeout_s)

    def reopen(self) -> None:
        self._reopen_session(self._handle)

    def _reopen_session(self, session: Any) -> None:
        """Restart the device if ``session`` is still the current one. ``_handle`` stays set
        while reopening, so a concurrent disconnect() clears it and the new device is closed."""
        with self._lifecycle_lock:
            if session is None or self._handle is not session:
                return
            self._close_live_device()
            with self._condition:
                self._latest = None
            self._last_capture_error = ""
            try:
                handle = self._open()
            except BaseException:
                if self._handle is session:
                    self._handle = None
                raise
            if self._handle is not session:
                self._close_live_device()
                return
            self._handle = handle

    def health(self) -> Health:
        frame = self.latest()
        age = None if frame is None else time.monotonic() - frame.timestamp
        fresh = age is not None and age <= 2.0 / self._config.fps
        return Health(
            ok=self.is_connected and fresh,
            detail=self._last_capture_error,
            last_feedback_age_s=age,
        )

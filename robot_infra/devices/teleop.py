"""Teleop contract: a background poll thread feeds read()/latest(); backends only implement
device open/close and a single poll. Edge events latch in the thread until read() consumes them,
so a short button press between two client reads is never lost.

Single consumer: ``read()`` clears the latched events, so two clients reading the same teleop
split the events between them. ``seq`` increases monotonically (also across reconnects) so a
client can detect dropped or repeated samples."""
from __future__ import annotations

import dataclasses
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, final

import numpy as np

from robot_infra.devices.base import Features, Health, Part
from robot_infra.devices.errors import DeviceFault, DeviceNotConnected
from robot_infra.devices.registry import BackendRegistry

ACTION_SPACES = ("cartesian_delta", "joint_abs", "joint_delta")
EVENT_KEYS = ("episode_end", "success", "failure", "rerecord")
_CARTESIAN_DELTA_DIM = 6
_THREAD_JOIN_TIMEOUT_S = 2.0

# Module-level clock so tests can drive hold_s without sleeping.
_now = time.monotonic


@dataclass
class TeleopConfig:
    name: str = "teleop"
    backend: str = "fake"
    hold_s: float = 0.0  # keep intervening this long after the last activity; 0 = no hysteresis
    options: dict = field(default_factory=dict)  # backend-private

    def __post_init__(self) -> None:
        if self.hold_s < 0:
            raise ValueError(f"hold_s must be >= 0, got {self.hold_s}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeleopConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        for key in data:
            if key not in known:
                raise ValueError(f"unknown key '{key}' for {cls.__name__}; known: {sorted(known)}")
        return cls(**data)


@dataclass
class TeleopSample:
    action: np.ndarray  # float32; cartesian_delta -> (6,) twist in [-1, 1]; joint_* -> (N,)
    action_space: str
    gripper: float | None  # normalized width target in [0, 1]; None = no gripper channel
    buttons: dict[str, bool]  # current level
    events: dict[str, bool]  # latched edges, cleared by read(); always holds EVENT_KEYS
    intervening: bool
    timestamp: float  # time.monotonic() of the poll that produced the sample
    seq: int

    def to_dict(self) -> dict:
        """Wire form: msgpack-codec friendly (ndarray, bool, dict, float, int, str, None)."""
        return {
            "action": self.action,
            "action_space": self.action_space,
            "gripper": self.gripper,
            "buttons": dict(self.buttons),
            "events": dict(self.events),
            "intervening": self.intervening,
            "timestamp": self.timestamp,
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeleopSample":
        gripper = data["gripper"]
        return cls(
            action=np.asarray(data["action"], dtype=np.float32),
            action_space=str(data["action_space"]),
            gripper=None if gripper is None else float(gripper),
            buttons={str(k): bool(v) for k, v in data["buttons"].items()},
            events={str(k): bool(v) for k, v in data["events"].items()},
            intervening=bool(data["intervening"]),
            timestamp=float(data["timestamp"]),
            seq=int(data["seq"]),
        )


@dataclass
class RawReading:
    """One poll of a backend. ``active`` is the backend's judgement that a human is operating;
    the base class turns it into ``intervening`` with the ``hold_s`` hysteresis."""

    action: np.ndarray
    gripper: float | None = None
    buttons: dict[str, bool] = field(default_factory=dict)
    events_now: dict[str, bool] = field(default_factory=dict)
    active: bool = False


@dataclass
class TeleopHealth(Health):
    thread_alive: bool = False
    error_count: int = 0
    consecutive_failures: int = 0


def teleop_observation_features(action_dim: int, has_gripper: bool) -> Features:
    features: Features = {
        "action": {"shape": (action_dim,), "dtype": "float32"},
        "intervening": {"shape": (1,), "dtype": "bool"},
    }
    if has_gripper:
        features["gripper"] = {"shape": (1,), "dtype": "float32"}
    return features


def teleop_observation(
    sample: TeleopSample | None, action_dim: int, has_gripper: bool
) -> dict[str, np.ndarray]:
    """ndarray form of a sample (zeros before the first one); never consumes events."""
    if sample is None:
        action = np.zeros(action_dim, dtype=np.float32)
        intervening, gripper = False, 0.0
    else:
        action = np.asarray(sample.action, dtype=np.float32).copy()
        intervening = sample.intervening
        gripper = 0.0 if sample.gripper is None else sample.gripper
    observation = {"action": action, "intervening": np.array([intervening], dtype=bool)}
    if has_gripper:
        observation["gripper"] = np.array([gripper], dtype=np.float32)
    return observation


class Teleop(Part, BackendRegistry, ABC):
    """Operator input device. Subclasses declare ``ACTION_SPACE``/``HAS_GRIPPER`` (and
    ``NUM_JOINTS`` for joint spaces) and implement ``_open_device``/``_close_device``/
    ``_poll_device``. ``_poll_device`` may block until the device has data, or return None
    when there is nothing new."""

    _BACKENDS: ClassVar[dict[str, type]] = {}
    ACTION_SPACE: ClassVar[str] = "cartesian_delta"
    HAS_GRIPPER: ClassVar[bool] = False
    NUM_JOINTS: ClassVar[int | None] = None
    # read() raises a recoverable DeviceFault once this many polls in a row failed.
    MAX_CONSECUTIVE_FAILURES: ClassVar[int] = 5
    FAILURE_BACKOFF_S: ClassVar[float] = 0.01
    IDLE_POLL_INTERVAL_S: ClassVar[float] = 0.001

    def __init__(
        self,
        config: TeleopConfig,
        *,
        owner: Part | None = None,
        read_timeout_s: float = 1.0,
    ) -> None:
        super().__init__(config.name, owner=owner)
        if self.ACTION_SPACE not in ACTION_SPACES:
            raise ValueError(
                f"teleop action space '{self.ACTION_SPACE}' invalid; choose from {list(ACTION_SPACES)}"
            )
        self._config = config
        self._read_timeout_s = read_timeout_s
        self._condition = threading.Condition()
        self._latest: TeleopSample | None = None  # intervening holds the raw ``active``
        self._seq = 0
        self._consumed_seq = 0
        self._closing = False
        self._pending_events: dict[str, bool] = {}
        self._last_active_time: float | None = None
        self._error_count = 0
        self._consecutive_failures = 0
        self._last_error = ""
        self._poll_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._live_device: Any = None

    @property
    def config(self) -> TeleopConfig:
        return self._config

    @property
    def action_space(self) -> str:
        return self.ACTION_SPACE

    @property
    def has_gripper(self) -> bool:
        return self.HAS_GRIPPER

    @property
    def action_dim(self) -> int:
        if self.ACTION_SPACE == "cartesian_delta":
            return _CARTESIAN_DELTA_DIM
        if self.NUM_JOINTS is None:
            raise ValueError(f"{type(self).__name__} uses '{self.ACTION_SPACE}' but declares no NUM_JOINTS")
        return self.NUM_JOINTS

    @property
    def observation_features(self) -> Features:
        return teleop_observation_features(self.action_dim, self.HAS_GRIPPER)

    def get_observation(self) -> dict[str, np.ndarray]:
        """Snapshot of ``latest()``: no blocking and no event consumption, so merging it into
        ``robot.get_observation()`` is harmless."""
        self._require_connected()
        return teleop_observation(self.latest(), self.action_dim, self.HAS_GRIPPER)

    def describe(self) -> dict:
        return {
            **super().describe(),
            "action_space": self.ACTION_SPACE,
            "has_gripper": self.HAS_GRIPPER,
            "action_dim": self.action_dim,
        }

    # -- lifecycle -----------------------------------------------------------

    @final
    def _open(self) -> Any:
        handle = self._open_device()
        with self._condition:
            self._latest = None
            self._closing = False
            self._pending_events = {}
            self._last_active_time = None
            self._consecutive_failures = 0
            self._last_error = ""
        self._live_device = handle
        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(handle, self._stop_event),
            name=f"teleop-{self.name}",
            daemon=True,
        )
        self._poll_thread.start()
        return handle

    @final
    def _release(self, handle: Any) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()  # wake blocked read()s; they raise DeviceNotConnected
        if self._stop_event is not None:
            self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        self._poll_thread = None
        self._stop_event = None
        device, self._live_device = self._live_device, None
        if device is not None:
            self._close_device(device)

    @abstractmethod
    def _open_device(self) -> Any: ...

    @abstractmethod
    def _close_device(self, handle: Any) -> None: ...

    @abstractmethod
    def _poll_device(self, handle: Any) -> RawReading | None: ...

    def _reset_device(self, handle: Any) -> None:
        """Clear backend-side sticky state on reset(); default no-op."""

    # -- poll thread ---------------------------------------------------------

    def _poll_loop(self, handle: Any, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                reading = self._poll_device(handle)
                if reading is not None and not stop_event.is_set():
                    self._publish(reading)
            except Exception as error:
                if stop_event.is_set():
                    break  # closing the device under a lingering poll is not a failure of the next session
                with self._condition:
                    self._error_count += 1
                    self._consecutive_failures += 1
                    self._last_error = str(error)
                    self._condition.notify_all()
                stop_event.wait(self.FAILURE_BACKOFF_S)
                continue
            if reading is None:
                stop_event.wait(self.IDLE_POLL_INTERVAL_S)

    def _publish(self, reading: RawReading) -> None:
        action = np.asarray(reading.action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"teleop '{self.name}' produced action shape {action.shape}, expected ({self.action_dim},)"
            )
        if reading.gripper is not None and not self.HAS_GRIPPER:
            raise ValueError(f"teleop '{self.name}' declares no gripper but produced one")
        gripper = None if reading.gripper is None else float(reading.gripper)
        with self._condition:
            now = _now()
            if reading.active:
                self._last_active_time = now
            for key, fired in reading.events_now.items():
                if fired:
                    self._pending_events[str(key)] = True
            self._seq += 1
            self._latest = TeleopSample(
                action=action.copy(),
                action_space=self.ACTION_SPACE,
                gripper=gripper,
                buttons={str(k): bool(v) for k, v in reading.buttons.items()},
                events={},
                intervening=bool(reading.active),
                timestamp=now,
                seq=self._seq,
            )
            self._consecutive_failures = 0
            self._condition.notify_all()

    # -- consumer side -------------------------------------------------------

    def _failing(self) -> bool:
        return self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES

    def _view(self, *, consume_events: bool) -> TeleopSample | None:
        """Caller holds ``_condition``."""
        latest = self._latest
        if latest is None:
            return None
        events = {key: False for key in EVENT_KEYS}
        events.update(self._pending_events)
        if consume_events:
            self._pending_events = {}
        held = (
            self._config.hold_s > 0
            and self._last_active_time is not None
            and _now() - self._last_active_time <= self._config.hold_s
        )
        return dataclasses.replace(
            latest,
            action=latest.action.copy(),
            buttons=dict(latest.buttons),
            events=events,
            intervening=latest.intervening or held,
        )

    def read(self, timeout_s: float | None = None) -> TeleopSample:
        """Block for a sample newer than the last read, then consume the latched events."""
        self._require_connected()
        timeout = self._read_timeout_s if timeout_s is None else timeout_s
        with self._condition:
            self._condition.wait_for(
                lambda: self._seq > self._consumed_seq or self._failing() or self._closing, timeout=timeout
            )
            if self._closing:
                raise DeviceNotConnected(
                    f"'{self.name}' was disconnected while reading",
                    hint=f"call connect() on '{self.name}' first",
                    device=self.name,
                )
            if self._failing():
                raise DeviceFault(
                    f"teleop '{self.name}' failed {self._consecutive_failures} polls in a row: "
                    f"{self._last_error}",
                    hint="check the device cable and permissions, then reconnect the teleop "
                    "(disconnect()/connect() the robot); run `python -m robot_infra.doctor`",
                    device=self.name,
                    recoverable=True,
                )
            if self._seq <= self._consumed_seq:
                raise DeviceFault(
                    f"teleop '{self.name}' produced no sample within {timeout:.2f}s",
                    hint="check that the device is plugged in and its poll thread is alive "
                    "(robot.health); run `python -m robot_infra.doctor`",
                    device=self.name,
                    recoverable=True,
                )
            self._consumed_seq = self._seq
            return self._view(consume_events=True)

    def latest(self) -> TeleopSample | None:
        """Most recent sample without blocking and without consuming events."""
        with self._condition:
            return self._view(consume_events=False)

    def reset(self) -> None:
        """Drop latched events and the hold_s hysteresis, then reset the backend."""
        self._require_connected()
        with self._condition:
            self._pending_events = {}
            self._last_active_time = None
        self._reset_device(self._live_device)

    def health(self) -> TeleopHealth:
        with self._condition:
            latest = self._latest
            failures = self._consecutive_failures
            errors = self._error_count
            detail = self._last_error
        thread = self._poll_thread
        alive = thread is not None and thread.is_alive()
        age = None if latest is None else _now() - latest.timestamp
        return TeleopHealth(
            ok=self.is_connected and alive and failures == 0,
            detail=detail,
            last_feedback_age_s=age,
            thread_alive=alive,
            error_count=errors,
            consecutive_failures=failures,
        )

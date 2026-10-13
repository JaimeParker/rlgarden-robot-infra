"""Hardware-free teleop driven by a script that advances one step per read(), plus in-process
``push()`` injection for unit tests. A scripted sample is produced only while a read() is
waiting, so the n-th read always returns the n-th scripted sample."""
from __future__ import annotations

import collections
import threading
from typing import Any, Mapping

import numpy as np

from robot_infra.devices.teleop import ACTION_SPACES, RawReading, Teleop, TeleopConfig, TeleopSample

_OPTION_KEYS = {"action_space", "has_gripper", "num_joints", "script", "on_end", "fail_after"}
_STEP_KEYS = {"action", "gripper", "buttons", "events", "active", "hold"}
_ON_END = ("hold_last", "idle", "loop")
_OPEN_GRIPPER = 1.0


@Teleop.register("fake")
class FakeTeleop(Teleop):
    """Options: ``action_space`` (default cartesian_delta), ``has_gripper`` (default True),
    ``num_joints`` (joint spaces), ``script`` (list of ``{action, gripper, buttons, events,
    active, hold}``; ``hold`` = number of reads the step lasts, its events fire on the first),
    ``on_end`` (hold_last | idle | loop), ``fail_after`` (every poll after the n-th sample fails
    until reconnect). An omitted step ``gripper`` keeps the previous target."""

    ACTION_SPACE = "cartesian_delta"
    HAS_GRIPPER = True

    def __init__(self, config: TeleopConfig, **kwargs: Any) -> None:
        options = dict(config.options)
        for key in options:
            if key not in _OPTION_KEYS:
                raise ValueError(f"unknown fake teleop option '{key}'; known: {sorted(_OPTION_KEYS)}")
        action_space = options.get("action_space", type(self).ACTION_SPACE)
        if action_space not in ACTION_SPACES:
            raise ValueError(f"action_space '{action_space}' invalid; choose from {list(ACTION_SPACES)}")
        if action_space != "cartesian_delta" and options.get("num_joints") is None:
            raise ValueError(f"fake teleop with action_space '{action_space}' needs options.num_joints")
        # Instance attributes shadow the ClassVars: the fake's shape is configured per instance.
        self.ACTION_SPACE = action_space
        self.HAS_GRIPPER = bool(options.get("has_gripper", type(self).HAS_GRIPPER))
        self.NUM_JOINTS = options.get("num_joints")
        super().__init__(config, **kwargs)
        self._on_end = options.get("on_end", "hold_last")
        if self._on_end not in _ON_END:
            raise ValueError(f"on_end '{self._on_end}' invalid; choose from {list(_ON_END)}")
        self._fail_after = options.get("fail_after")
        self._script = [
            self._validated_step(index, step) for index, step in enumerate(options.get("script", []))
        ]
        self._script_lock = threading.Lock()
        self._pushed: collections.deque[RawReading] = collections.deque()
        self._read_pending = False  # a read() is waiting for the next scripted sample
        self._reset_script()

    @classmethod
    def discover(cls) -> set[str]:
        return {"fake"}

    def _validated_step(self, index: int, step: Mapping[str, Any]) -> dict:
        for key in step:
            if key not in _STEP_KEYS:
                raise ValueError(
                    f"unknown key '{key}' in fake teleop script[{index}]; known: {sorted(_STEP_KEYS)}"
                )
        step = dict(step)
        action = np.asarray(step.get("action", np.zeros(self.action_dim)), dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"script[{index}].action has shape {action.shape}, expected ({self.action_dim},)")
        step["action"] = action
        if step.get("gripper") is not None and not self.HAS_GRIPPER:
            raise ValueError(f"script[{index}] sets gripper but has_gripper is false")
        hold = step.get("hold", 1)
        if isinstance(hold, bool) or not isinstance(hold, int) or hold < 1:
            raise ValueError(f"script[{index}].hold must be a positive int, got {hold!r}")
        step["hold"] = hold
        return step

    def _reset_script(self) -> None:
        self._step_index = 0
        self._step_reads = 0
        self._produced = 0
        self._gripper = _OPEN_GRIPPER if self.HAS_GRIPPER else None
        self._pushed.clear()

    def push(self, sample: Mapping[str, Any]) -> None:
        """Publish one sample now (same keys as a script step, without ``hold``)."""
        self._require_connected()
        step = self._validated_step(-1, sample)
        with self._script_lock:
            self._publish(self._reading(step, with_events=True))

    def read(self, timeout_s: float | None = None) -> TeleopSample:
        with self._script_lock:
            self._read_pending = True
        try:
            return super().read(timeout_s)
        finally:
            with self._script_lock:
                self._read_pending = False

    def _open_device(self) -> Any:
        with self._script_lock:
            self._reset_script()
        return {"opened": True}

    def _close_device(self, handle: Any) -> None:
        pass

    def _poll_device(self, handle: Any) -> RawReading | None:
        if self._fail_after is not None and self._produced >= self._fail_after:
            raise RuntimeError(f"fake teleop failure after {self._fail_after} samples")
        with self._script_lock:
            if not self._read_pending:
                return None
            self._read_pending = False  # one sample per read()
            self._produced += 1
            return self._next_scripted()

    def _reading(self, step: Mapping[str, Any], *, with_events: bool) -> RawReading:
        if step.get("gripper") is not None:
            self._gripper = float(step["gripper"])
        return RawReading(
            action=np.asarray(step["action"], dtype=np.float32).copy(),
            gripper=self._gripper,
            buttons=dict(step.get("buttons", {})),
            events_now=dict(step.get("events", {})) if with_events else {},
            active=bool(step.get("active", False)),
        )

    def _idle_reading(self) -> RawReading:
        return RawReading(action=np.zeros(self.action_dim, dtype=np.float32), gripper=self._gripper)

    def _next_scripted(self) -> RawReading:
        if not self._script:
            return self._idle_reading()
        if self._step_index >= len(self._script):
            if self._on_end == "idle":
                return self._idle_reading()
            if self._on_end == "hold_last":
                return self._reading(self._script[-1], with_events=False)
            self._step_index, self._step_reads = 0, 0
        step = self._script[self._step_index]
        reading = self._reading(step, with_events=self._step_reads == 0)
        self._step_reads += 1
        if self._step_reads >= step["hold"]:
            self._step_index += 1
            self._step_reads = 0
        return reading

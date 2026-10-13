"""Transport-neutral robot service: a string method table over a ``Robot``, a device
claim held while connected, and a watchdog that idles a servoing arm whose client went quiet."""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from robot_infra.devices.arm import ControlMode
from robot_infra.devices.base import PartGroup
from robot_infra.devices.claims import DeviceClaim, claim_key
from robot_infra.devices.errors import DeviceFault, DeviceNotFound, SdkMissing
from robot_infra.devices.gripper import Gripper
from robot_infra.devices.robot import Robot
from robot_infra.server.codec import encode_jpeg

logger = logging.getLogger(__name__)

CAMERA_READ_METHOD = "camera.<name>.read"
TELEOP_READ_METHOD = "teleop.<name>.read"
TELEOP_RESET_METHOD = "teleop.<name>.reset"
TELEOP_READ_MAX_TIMEOUT_S = 2.0  # a client cannot park a gRPC worker thread longer than this
METHODS = (
    "robot.describe",
    "robot.connect",
    "robot.disconnect",
    "robot.health",
    "robot.get_observation",
    "robot.send_action",
    "robot.step",
    "arm.read_state",
    "arm.servo_cartesian",
    "arm.servo_joints",
    "arm.move_to_pose",
    "arm.move_to_joints",
    "arm.stop",
    "arm.recover",
    "arm.set_control_mode",
    "arm.control_mode",
    "arm.health",
    "arm.preflight",
    "gripper.open",
    "gripper.close",
    "gripper.move",
    "gripper.width",
    CAMERA_READ_METHOD,
    TELEOP_READ_METHOD,
    TELEOP_RESET_METHOD,
)
_WATCHDOG_JOIN_TIMEOUT_S = 2.0


def _require_jpeg_backend() -> None:
    try:
        import PIL  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SdkMissing(
            "--jpeg-quality needs Pillow or OpenCV",
            hint='pip install "rlgarden-robot-infra[jpeg]"',
            device="codec",
        ) from None


class RobotService:
    def __init__(
        self,
        robot: Robot,
        *,
        watchdog_s: float = 0.5,
        claim: bool | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        if watchdog_s <= 0:
            raise ValueError(f"watchdog_s must be positive, got {watchdog_s}")
        if jpeg_quality is not None:
            if not 1 <= jpeg_quality <= 100:
                raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
            _require_jpeg_backend()
        self.jpeg_quality = jpeg_quality
        self.robot = robot
        self.watchdog_s = watchdog_s
        self._use_claim = robot.config.claim if claim is None else claim
        self._claim: DeviceClaim | None = None
        # Serializes robot commands with the watchdog's check-and-trip.
        self._lock = threading.RLock()
        self._last_action_time: float | None = None
        self._servo_entered_time: float | None = None
        self._watchdog_trips = 0
        self._watchdog_tripped_at: float | None = None
        self._watchdog_idle_s = 0.0
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._methods: dict[str, Callable[..., Any]] = {
            "robot.describe": self._describe,
            "robot.connect": self._connect,
            "robot.disconnect": self._disconnect,
            "robot.health": self._health,
            "robot.get_observation": self._get_observation,
            "robot.send_action": self._send_action,
            "robot.step": self._step,
            "arm.read_state": self._arm_read_state,
            "arm.servo_cartesian": self._arm_servo_cartesian,
            "arm.servo_joints": self._arm_servo_joints,
            "arm.move_to_pose": self._arm_move_to_pose,
            "arm.move_to_joints": self._arm_move_to_joints,
            "arm.stop": self._arm_stop,
            "arm.recover": self._arm_recover,
            "arm.set_control_mode": self._arm_set_control_mode,
            "arm.control_mode": self._arm_control_mode,
            "arm.health": self._arm_health,
            "arm.preflight": self._arm_preflight,
            "gripper.open": self._gripper_open,
            "gripper.close": self._gripper_close,
            "gripper.move": self._gripper_move,
            "gripper.width": self._gripper_width,
        }
        for name in robot.cameras:
            self._methods[f"camera.{name}.read"] = self._camera_reader(name)
        if robot.teleop is not None:
            self._methods[f"teleop.{robot.teleop.name}.read"] = self._teleop_read
            self._methods[f"teleop.{robot.teleop.name}.reset"] = self._teleop_reset

    @property
    def methods(self) -> list[str]:
        return list(self._methods)

    @property
    def watchdog_trips(self) -> int:
        return self._watchdog_trips

    @property
    def watchdog_running(self) -> bool:
        return self._watchdog_thread is not None and self._watchdog_thread.is_alive()

    @property
    def last_action_age_s(self) -> float | None:
        if self._last_action_time is None:
            return None
        return time.monotonic() - self._last_action_time

    def dispatch(self, method: str, kwargs: dict | None = None) -> Any:
        try:
            handler = self._methods[method]
        except KeyError:
            raise ValueError(f"unknown method '{method}'; available: {self.methods}") from None
        return handler(**(kwargs or {}))

    # -- robot.* -------------------------------------------------------------

    def _describe(self) -> dict:
        robot = self.robot
        arm = robot.arm
        gripper = robot.gripper
        gripper_config = robot.config.gripper
        return {
            "type": robot.ROBOT_TYPE,
            "describe": robot.describe(),
            "arm": {
                "num_joints": arm.num_joints,
                "supported_state_fields": list(arm.SUPPORTED_STATE_FIELDS),
                "control_mode": arm.control_mode.value,
                "safety": arm.safety_limits,
                "home_joints": robot.config.home_joints,
            },
            "gripper": None
            if gripper is None
            else {
                "max_width_m": gripper.max_width_m,
                "mode": gripper.mode.value,
                "binary_threshold": gripper_config.binary_threshold if gripper_config else None,
            },
            "cameras": {name: dataclasses.asdict(camera.config) for name, camera in robot.cameras.items()},
            # Daemon-global, kept out of the per-camera dicts: older clients unpack those as CameraConfig.
            "image_encoding": self._image_encoding(),
            "teleop": self._describe_teleop(),
        }

    def _describe_teleop(self) -> dict | None:
        teleop = self.robot.teleop
        if teleop is None:
            return None
        return {
            "name": teleop.name,
            "backend": teleop.config.backend,
            "action_space": teleop.ACTION_SPACE,
            "has_gripper": teleop.HAS_GRIPPER,
            "hold_s": teleop.config.hold_s,
        }

    def _connect(self) -> None:
        with self._lock:
            self._acquire_claim()
            try:
                self.robot.connect()
            except BaseException:
                self._release_claim()
                raise
            self.start_watchdog()

    def _disconnect(self) -> None:
        with self._lock:
            try:
                self.robot.disconnect()
            finally:
                self._release_claim()

    def _health(self) -> dict:
        parts: dict[str, Any] = {}
        for name, child in self.robot.children.items():
            if isinstance(child, PartGroup):
                parts[name] = {sub: dataclasses.asdict(part.health()) for sub, part in child.children.items()}
            else:
                parts[name] = dataclasses.asdict(child.health())
        return {
            "robot": dataclasses.asdict(self.robot.health()),
            **parts,
            "connected": self.robot.is_connected,
            "watchdog_trips": self._watchdog_trips,
            "last_action_age_s": self.last_action_age_s,
        }

    def _get_observation(self) -> dict:
        return self._wire_observation(self.robot.get_observation())

    def _send_action(self, action: dict) -> None:
        with self._lock:
            self._raise_if_watchdog_tripped()
            self.robot.send_action(action)
            self._mark_action()

    def _step(self, action: dict) -> dict:
        self._send_action(action)
        return self._wire_observation(self.robot.get_observation())

    # -- arm.* ---------------------------------------------------------------

    def _arm_read_state(self) -> dict:
        return self.robot.arm.read_state()

    def _arm_servo_cartesian(self, pose: Any) -> None:
        with self._lock:
            self._raise_if_watchdog_tripped()
            self.robot.arm.servo_cartesian(np.asarray(pose, dtype=np.float64))
            self._mark_action()

    def _arm_servo_joints(self, q: Any) -> None:
        with self._lock:
            self._raise_if_watchdog_tripped()
            self.robot.arm.servo_joints(np.asarray(q, dtype=np.float64))
            self._mark_action()

    def _arm_move_to_pose(self, pose: Any, blocking: bool = True) -> None:
        with self._lock:
            self.robot.arm.move_to_pose(np.asarray(pose, dtype=np.float64), blocking=blocking)

    def _arm_move_to_joints(self, q: Any, blocking: bool = True) -> None:
        with self._lock:
            self.robot.arm.move_to_joints(np.asarray(q, dtype=np.float64), blocking=blocking)

    def _arm_stop(self) -> None:
        with self._lock:
            self.robot.arm.stop()

    def _arm_recover(self) -> None:
        with self._lock:
            self.robot.arm.recover()

    def _arm_set_control_mode(self, mode: str) -> None:
        try:
            control_mode = ControlMode(mode)
        except ValueError:
            raise ValueError(
                f"unknown control mode '{mode}'; choose from {[m.value for m in ControlMode]}"
            ) from None
        with self._lock:
            self.robot.arm.set_control_mode(control_mode)
            self._watchdog_tripped_at = None
            if control_mode is not ControlMode.IDLE:
                # A client that enters servo mode and never sends an action still trips the watchdog.
                self._servo_entered_time = time.monotonic()

    def _arm_control_mode(self) -> str:
        return self.robot.arm.control_mode.value

    def _arm_health(self) -> dict:
        return dataclasses.asdict(self.robot.arm.health())

    def _arm_preflight(self) -> list[dict]:
        results = []
        for result in self.robot.arm.preflight():
            entry = {
                field.name: getattr(result, field.name)
                for field in dataclasses.fields(result)
                if field.name != "autofix"
            }
            entry["has_autofix"] = result.autofix is not None
            results.append(entry)
        return results

    # -- gripper.* -----------------------------------------------------------

    def _require_gripper(self) -> Gripper:
        gripper = self.robot.gripper
        if gripper is None:
            raise DeviceNotFound(
                f"robot '{self.robot.name}' has no gripper",
                hint="this robot has no gripper",
                device="gripper",
            )
        return gripper

    def _gripper_open(self) -> None:
        with self._lock:
            self._require_gripper().open()

    def _gripper_close(self) -> None:
        with self._lock:
            self._require_gripper().close()

    def _gripper_move(self, width: float) -> None:
        with self._lock:
            self._require_gripper().move(float(width))

    def _gripper_width(self) -> float:
        return float(self._require_gripper().width)

    # -- camera.<name>.read --------------------------------------------------

    def _camera_reader(self, name: str) -> Callable[..., dict]:
        def read(timeout_s: float | None = None) -> dict:
            frame = self.robot.cameras[name].read(timeout_s)
            return {"color": self._wire_color(frame.color), "depth": frame.depth, "timestamp": frame.timestamp}

        return read

    def _image_encoding(self) -> dict | None:
        return None if self.jpeg_quality is None else {"jpeg": self.jpeg_quality}

    def _wire_color(self, color: np.ndarray) -> Any:
        return color if self.jpeg_quality is None else encode_jpeg(color, self.jpeg_quality)

    def _wire_observation(self, observation: dict) -> dict:
        """JPEG-encode each camera's color (depth and other keys untouched); runs outside ``_lock``."""
        if self.jpeg_quality is None or "cameras" not in observation:
            return observation
        cameras = {
            name: {**fields, "color": self._wire_color(fields["color"])}
            for name, fields in observation["cameras"].items()
        }
        return {**observation, "cameras": cameras}

    # -- teleop.<name>.* -----------------------------------------------------
    # Not under ``_lock`` and not ``_mark_action``: reading the operator never blocks arm
    # commands and never keeps the watchdog quiet.

    def _teleop_read(self, timeout_s: float | None = None) -> dict:
        timeout_s = TELEOP_READ_MAX_TIMEOUT_S if timeout_s is None else timeout_s
        timeout_s = min(max(timeout_s, 0.0), TELEOP_READ_MAX_TIMEOUT_S)
        return self.robot.teleop.read(timeout_s).to_dict()

    def _teleop_reset(self) -> None:
        self.robot.teleop.reset()

    # -- watchdog ------------------------------------------------------------

    def start_watchdog(self) -> None:
        if self.watchdog_running:
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name=f"watchdog-{self.robot.name}", daemon=True
        )
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread, self._watchdog_thread = self._watchdog_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_WATCHDOG_JOIN_TIMEOUT_S)

    def _watchdog_loop(self) -> None:
        period = self.watchdog_s / 4
        while not self._watchdog_stop.wait(period):
            try:
                self._watchdog_check()
            except Exception:
                logger.exception("watchdog check on robot '%s' failed", self.robot.name)

    def _watchdog_check(self) -> None:
        with self._lock:
            arm = self.robot.arm
            if not arm.is_connected or arm.control_mode is ControlMode.IDLE:
                return
            reference = max(
                (t for t in (self._last_action_time, self._servo_entered_time) if t is not None),
                default=None,
            )
            if reference is None or time.monotonic() - reference <= self.watchdog_s:
                return
            self._watchdog_trips += 1
            self._watchdog_idle_s = time.monotonic() - reference
            logger.warning(
                "watchdog: no action for %.3fs in %s mode; stopping arm '%s' and idling it",
                self._watchdog_idle_s,
                arm.control_mode.value,
                arm.name,
            )
            arm.stop()
            arm.set_control_mode(ControlMode.IDLE)
            self._watchdog_tripped_at = time.monotonic()

    def _raise_if_watchdog_tripped(self) -> None:
        """Replace the generic 'arm is idle' fault with the real cause until the client re-enters a mode."""
        if self._watchdog_tripped_at is None:
            return
        raise DeviceFault(
            f"the daemon watchdog stopped the arm after {self._watchdog_idle_s:.2f}s without actions "
            f"(limit {self.watchdog_s}s)",
            hint="the client stalled or the network dropped; call set_control_mode(...) again to resume "
            "servoing, and keep the action rate above 1/watchdog_s",
            recoverable=True,
            device=self.robot.arm.name,
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self.stop_watchdog()
        with self._lock:
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            finally:
                self._release_claim()

    def _mark_action(self) -> None:
        self._last_action_time = time.monotonic()

    def _acquire_claim(self) -> None:
        if not self._use_claim or self._claim is not None:
            return
        claim = DeviceClaim(claim_key(self.robot))
        claim.acquire()
        self._claim = claim

    def _release_claim(self) -> None:
        claim, self._claim = self._claim, None
        if claim is not None:
            claim.release()


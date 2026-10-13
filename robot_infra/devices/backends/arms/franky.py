"""Franka arm backend on franky (https://github.com/TimSchneider42/franky), imported lazily.

Servo modes stream targets into franky's torque-level impedance trackers; ``move_to_*`` runs
Ruckig-planned position motions. libfranka refuses to switch control-signal type while a motion
runs, so every mode change joins the running motion and recovers before starting the next one.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from robot_infra.devices.arm import Arm, ControlMode
from robot_infra.devices.errors import (
    DeviceFault,
    DeviceNotFound,
    SdkMissing,
    VersionMismatch,
)
from robot_infra.devices.safety import ArmSafety

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig
    from robot_infra.doctor.checks import CheckResult

REALTIME_KERNEL_FLAG_PATH = "/sys/kernel/realtime"
FRANKY_DISTRIBUTION = "franky-control"
FRANKA_UNREACHABLE_HINT = (
    "check: ping <ip>; FCI activated in Desk; cable on the Control (not Arm) port; "
    "'UDP receive: Timeout' means a firewall"
)
FRANKY_VERSION_HINT = (
    "install the franky wheel matching your robot server version: pip install franky-control "
    "--extra-index-url https://timschneider42.github.io/franky/whl/by-robot-server-version/<N>/ "
    "(see franky README; libfranka/docs/compatibility_matrix.rst maps firmware to server version)"
)
_BLOCKING_ROBOT_MODES = ("Reflex", "UserStopped", "Guiding")


def require_franky(device_cls: type, device: str) -> ModuleType:
    """Import franky; the PyPI distribution is ``franky-control``, not ``franky``."""
    try:
        return importlib.import_module("franky")
    except ImportError:
        raise SdkMissing(
            f"{device_cls.__name__} needs the 'franky' package",
            hint=f"pip install {FRANKY_DISTRIBUTION} (or: pip install 'rlgarden-robot-infra[franky]')",
            device=device,
        ) from None


def franky_connect_error(franky: ModuleType, error: Exception, robot_ip: str, device: str) -> Exception:
    """Map a franky/libfranka connect failure to a typed device error."""
    if isinstance(error, franky.IncompatibleVersionException):
        return VersionMismatch(
            f"libfranka/robot firmware mismatch: {error}", hint=FRANKY_VERSION_HINT, device=device
        )
    if isinstance(error, franky.RealtimeException):
        return DeviceFault(
            f"libfranka realtime setup failed for {robot_ip}: {error}",
            hint="install a PREEMPT_RT kernel (and the realtime group limits) or set realtime: ignore",
            device=device,
        )
    return DeviceNotFound(
        f"cannot reach Franka at {robot_ip}: {error}", hint=FRANKA_UNREACHABLE_HINT, device=device
    )


def franky_version() -> str:
    """franky version; wheels built against a non-default libfranka carry a ``+libfranka.X`` label."""
    try:
        version = importlib.metadata.version(FRANKY_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    _, _, local = version.partition("+")
    libfranka = local.split("libfranka.", 1)[1] if "libfranka." in local else "bundled default"
    return f"{version} (libfranka: {libfranka})"


@dataclass
class _FrankyHandle:
    robot: Any
    tracker: Any = None


@Arm.register("franky")
class FrankyArm(Arm):
    SDK = "franky"
    SUPPORTED_STATE_FIELDS = ("tcp_pose", "tcp_vel", "joint_pos", "joint_vel", "tcp_force", "tcp_torque")

    def __init__(
        self,
        name: str = "arm",
        *,
        robot_ip: str,
        relative_dynamics_factor: float = 0.2,
        realtime: str = "ignore",
        home_joints: list[float] | None = None,
        safety: ArmSafety | None = None,
    ) -> None:
        super().__init__(name, num_joints=7, safety=safety)
        if realtime not in ("ignore", "enforce"):
            raise ValueError(f"realtime '{realtime}' invalid; choose from ['ignore', 'enforce']")
        self.robot_ip = robot_ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.realtime = realtime
        self.home_joints = None if home_joints is None else list(home_joints)

    @classmethod
    def from_config(cls, config: "RobotConfig", *, name: str = "arm") -> "FrankyArm":
        return cls(
            name,
            robot_ip=config.robot_ip,
            relative_dynamics_factor=config.relative_dynamics_factor,
            realtime=config.realtime,
            home_joints=config.home_joints,
            safety=ArmSafety.from_config(config),
        )

    @classmethod
    def require_sdk(cls) -> ModuleType:
        return require_franky(cls, cls.__name__)

    # --- lifecycle ---

    def _open(self) -> _FrankyHandle:
        franky = self.require_sdk()
        realtime_config = (
            franky.RealtimeConfig.Enforce if self.realtime == "enforce" else franky.RealtimeConfig.Ignore
        )
        try:
            robot = franky.Robot(
                self.robot_ip,
                relative_dynamics_factor=self.relative_dynamics_factor,
                realtime_config=realtime_config,
            )
        except Exception as error:
            raise franky_connect_error(franky, error, self.robot_ip, self.name) from error
        handle = _FrankyHandle(robot=robot)
        self._read_state_from(robot)
        return handle

    def _release(self, handle: _FrankyHandle) -> None:
        try:
            self._stop_motion(handle)
        finally:
            handle.robot = None

    # --- state ---

    def read_state(self) -> dict[str, np.ndarray]:
        self._require_connected()
        return self._remember_state(self._read_state_from(self._handle.robot))

    def _read_state_from(self, robot: Any) -> dict[str, np.ndarray]:
        franky = self.require_sdk()
        state = robot.state
        ee_pose = state.O_T_EE
        dq = np.asarray(state.dq, dtype=np.float64)
        # O_dP_EE_c is the commanded twist (zero under torque control), so measure it: J(q) dq.
        jacobian = np.asarray(robot.model.zero_jacobian(franky.Frame.EndEffector, state), dtype=np.float64)
        wrench = np.asarray(state.O_F_ext_hat_K, dtype=np.float64)
        self._robot_mode = getattr(state.robot_mode, "name", str(state.robot_mode))
        self._note_feedback()
        return {
            "tcp_pose": np.concatenate(
                [np.asarray(ee_pose.translation, dtype=np.float64), np.asarray(ee_pose.quaternion, dtype=np.float64)]
            ),
            "tcp_vel": jacobian @ dq,
            "joint_pos": np.asarray(state.q, dtype=np.float64),
            "joint_vel": dq,
            "tcp_force": wrench[:3],
            "tcp_torque": wrench[3:],
        }

    # --- commands ---

    def _servo_cartesian(self, pose: np.ndarray) -> None:
        tracker = self._live_tracker(ControlMode.CARTESIAN_SERVO)
        tracker.set_target(self._affine(pose))

    def _servo_joints(self, q: np.ndarray) -> None:
        tracker = self._live_tracker(ControlMode.JOINT_SERVO)
        tracker.set_target(self._checked(q, (self.num_joints,)))

    def _move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self._require_idle("move_to_pose")
        franky = self.require_sdk()
        motion = franky.CartesianMotion(self._affine(pose))
        with self._control_exceptions_as_faults("move_to_pose"):
            self._handle.robot.move(motion, asynchronous=not blocking)

    def _move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self._require_idle("move_to_joints")
        franky = self.require_sdk()
        target = self._checked(q, (self.num_joints,))
        with self._control_exceptions_as_faults("move_to_joints"):
            self._handle.robot.move(franky.JointMotion(target), asynchronous=not blocking)

    def stop(self) -> None:
        """Bring any running tracker or position motion to rest. Power and brakes are untouched."""
        self._require_connected()
        self._stop_motion(self._handle)

    def recover(self) -> None:
        self._require_connected()
        if not self._handle.robot.recover_from_errors():
            raise DeviceFault(
                f"arm '{self.name}' did not recover from its errors: {self._errors}",
                hint="release the external activation device / clear errors in Desk, then retry",
                device=self.name,
                recoverable=True,
            )
        self._clear_errors()

    def _enter_control_mode(self, mode: ControlMode) -> None:
        handle = self._handle
        self._stop_motion(handle)
        self._settle_and_recover(handle.robot)
        if mode is not ControlMode.IDLE:
            handle.tracker = self._new_tracker(handle.robot, mode)

    # --- health / doctor ---

    def health(self):
        if self.is_connected and getattr(self._handle.robot, "has_errors", False):
            message = f"franka errors: {self._handle.robot.state.current_errors}"
            if message not in self._errors:
                self._note_error(message)
        return super().health()

    def preflight(self) -> list["CheckResult"]:
        from robot_infra.doctor.checks import CheckResult

        return [
            self._realtime_kernel_check(CheckResult),
            self._robot_mode_check(CheckResult),
            CheckResult(5, "mode:libfranka_version", True, f"franky {franky_version()}"),
        ]

    def _realtime_kernel_check(self, check_result: type) -> "CheckResult":
        has_rt_kernel = _read_realtime_flag() == "1"
        if has_rt_kernel:
            return check_result(5, "mode:realtime_kernel", True, f"PREEMPT_RT kernel; realtime: {self.realtime}")
        if self.realtime == "ignore":
            return check_result(
                5,
                "mode:realtime_kernel",
                True,
                "no PREEMPT_RT kernel; running with RealtimeConfig.Ignore (control jitter possible)",
            )
        return check_result(
            5,
            "mode:realtime_kernel",
            False,
            "realtime: enforce but the kernel is not PREEMPT_RT",
            hint="install a PREEMPT_RT kernel or set realtime: ignore",
        )

    def _robot_mode_check(self, check_result: type) -> "CheckResult":
        if self.is_connected:
            self.read_state()  # re-read so a doctor rerun after the recover autofix sees the new mode
        if self._robot_mode in _BLOCKING_ROBOT_MODES:
            return check_result(
                5,
                "mode:robot_mode",
                False,
                f"robot mode is {self._robot_mode}",
                hint="release the external activation device / clear errors in Desk, then run with --fix (recover)",
                autofix=self.recover,
            )
        return check_result(5, "mode:robot_mode", True, f"robot mode is {self._robot_mode or 'unknown'}")

    # --- helpers ---

    def _new_tracker(self, robot: Any, mode: ControlMode) -> Any:
        """Constructing a franky tracker seeds the current pose and starts the torque motion."""
        franky = self.require_sdk()
        tracker_cls = (
            franky.CartesianImpedanceTracker
            if mode is ControlMode.CARTESIAN_SERVO
            else franky.JointImpedanceTracker
        )
        with self._control_exceptions_as_faults(f"starting the {mode.value} tracker"):
            return tracker_cls(robot)

    @contextlib.contextmanager
    def _control_exceptions_as_faults(self, operation: str) -> Iterator[None]:
        """Record a libfranka ControlException (e.g. a reflex) and raise it as a recoverable fault."""
        franky = self.require_sdk()
        try:
            yield
        except franky.ControlException as error:
            if _is_preemption(error):
                raise
            message = f"franka {operation} failed: {error}"
            self._note_error(message)
            raise DeviceFault(
                message,
                hint="call recover(); check the workspace / Desk",
                device=self.name,
                recoverable=True,
            ) from error

    def _live_tracker(self, mode: ControlMode) -> Any:
        """Return the running tracker for ``mode``, restarting it after a plain stop() and
        surfacing a reflex that killed it (franky would otherwise drop targets silently)."""
        self._require_connected()
        if self._control_mode is not mode:
            raise DeviceFault(
                f"arm '{self.name}' is in {self._control_mode.value} mode, not {mode.value}",
                hint=f"call set_control_mode(ControlMode.{mode.name}) first",
                device=self.name,
            )
        handle = self._handle
        if handle.tracker is not None and not handle.robot.is_in_control:
            handle.tracker = None
            message = f"franka impedance tracker stopped: {self._motion_error(handle.robot)}"
            self._note_error(message)
            raise DeviceFault(
                message,
                hint="call recover(); the next servo command restarts the tracker",
                device=self.name,
                recoverable=True,
            )
        if handle.tracker is None:
            if self._errors:
                raise DeviceFault(
                    f"arm '{self.name}' has unrecovered errors: {self._errors}",
                    hint="call recover() before servoing",
                    device=self.name,
                    recoverable=True,
                )
            self._enter_control_mode(mode)
        return handle.tracker

    def _stop_motion(self, handle: _FrankyHandle) -> None:
        tracker, handle.tracker = handle.tracker, None
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as error:
                self._note_error(f"stopping the impedance tracker failed: {error}")
        elif handle.robot is not None and handle.robot.is_in_control:
            handle.robot.stop()
            try:
                handle.robot.join_motion()
            except Exception as error:
                if not _is_preemption(error):
                    self._note_error(f"stopping the franka motion failed: {error}")

    def _settle_and_recover(self, robot: Any) -> None:
        """Join the previous motion and recover so the next one can start. A real motion error
        stays recorded (health() reports it, servoing refuses to restart) until recover()."""
        try:
            robot.join_motion()
        except Exception as error:
            if not _is_preemption(error):
                self._note_error(f"previous franka motion ended with: {error}")
        robot.recover_from_errors()

    def _require_idle(self, command: str) -> None:
        self._require_connected()
        if self._control_mode is not ControlMode.IDLE:
            raise DeviceFault(
                f"{command} needs the arm idle, but it is in {self._control_mode.value} mode",
                hint="call set_control_mode(ControlMode.IDLE) first",
                device=self.name,
            )

    def _affine(self, pose: np.ndarray) -> Any:
        franky = self.require_sdk()
        pose = self._checked(pose, (7,))
        return franky.Affine(pose[:3], pose[3:])

    @staticmethod
    def _motion_error(robot: Any) -> str:
        try:
            robot.join_motion()
        except Exception as error:
            return str(error)
        return "control ended without an error"

    @staticmethod
    def _checked(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"expected shape {shape}, got {value.shape}")
        return value


def _is_preemption(error: Exception) -> bool:
    """libfranka reports a motion ended by Robot.stop() as a 'Move command preempted!' ControlException."""
    return "command preempted!" in str(error)


def _read_realtime_flag() -> str:
    try:
        return Path(REALTIME_KERNEL_FLAG_PATH).read_text().strip()
    except OSError:
        return ""

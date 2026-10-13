"""AgileX Piper arm on a CAN bus via ``pyAgxArm``.

Releasing the arm only closes the CAN session. ``disable()`` and ``reset()``
are never called: both cut motor power, and a raised Piper falls.
"""
from __future__ import annotations

import contextlib
import importlib
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from robot_infra.devices.arm import Arm, ArmHealth, ControlMode
from robot_infra.devices.backends.arms._rotations import quat_xyzw_to_rpy, rpy_to_quat_xyzw
from robot_infra.devices.errors import DeviceFault, DeviceNotFound, SdkMissing
from robot_infra.devices.safety import ArmSafety

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig
    from robot_infra.doctor.checks import CheckResult

def require_pyagxarm(device_cls: type) -> ModuleType:
    """Import pyAgxArm; it is installed from GitHub, not PyPI."""
    try:
        return importlib.import_module("pyAgxArm")
    except ImportError:
        raise SdkMissing(
            f"{device_cls.__name__} needs the 'pyAgxArm' package",
            hint="pip install 'rlgarden-robot-infra[pyagxarm]'  (pyAgxArm is installed from GitHub)",
            device=device_cls.__name__,
        ) from None


PIPER_MODEL = "piper"
PIPER_NUM_JOINTS = 6
FIRMWARE_PROFILES = ("default", "v183", "v188", "v189")
_PROBE_PROFILE = "default"
_ARM_STATUS_NORMAL = 0
_MOTION_REACHED = 0
_ERROR_FLAGS = tuple(f"joint_{i}_angle_limit" for i in range(1, PIPER_NUM_JOINTS + 1)) + tuple(
    f"communication_status_joint_{i}" for i in range(1, PIPER_NUM_JOINTS + 1)
)
# Parser slots filled by the per-frame CAN feedback (joints 1-2/3-4/5-6, end pose xy/zrx/ryrz).
# The SDK's get_joint_angles/get_flange_pose already return after any one of them, with the
# missing fields left at 0.
_FEEDBACK_FRAMES = ("joint_12", "joint_34", "joint_56", "end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
_RECOVER_HINT = "call recover(); if it persists, release the e-stop or power-cycle"
_FIRMWARE_HINT = "set firmware_profile explicitly (default|v183|v188|v189) in the robot config"
_FOLLOWER_HINT = (
    "no CAN feedback: the arm may be in master/teach mode; switch it to follower (slave) "
    "mode and power-cycle"
)


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


@Arm.register("pyagxarm")
class AgxArm(Arm):
    """Piper 6-DoF arm. Orientation feedback/commands are the SDK's extrinsic-XYZ rpy
    (``R = Rz @ Ry @ Rx``), converted to/from ``[x, y, z, w]`` quaternions. ``tcp_pose``
    is the flange pose: no TCP offset is configured on the SDK."""

    SDK = "pyAgxArm"

    @classmethod
    def require_sdk(cls) -> ModuleType:
        return require_pyagxarm(cls)
    SUPPORTED_STATE_FIELDS = ("tcp_pose", "joint_pos", "joint_vel")
    ENABLE_TIMEOUT_S = 5.0
    FEEDBACK_TIMEOUT_S = 2.0
    FEEDBACK_STALE_S = 0.5
    MOTION_TIMEOUT_S = 15.0
    MOTION_SETTLE_S = 0.5
    POLL_S = 0.01

    def __init__(
        self,
        name: str = "arm",
        *,
        can_interface: str = "can0",
        bitrate: int = 1_000_000,
        firmware_profile: str | None = None,
        home_joints: list[float] | None = None,
        with_gripper: bool = False,
        safety: ArmSafety | None = None,
    ) -> None:
        if firmware_profile is not None and firmware_profile not in FIRMWARE_PROFILES:
            raise ValueError(
                f"unknown Piper firmware_profile '{firmware_profile}'; "
                f"choose from {list(FIRMWARE_PROFILES)} or None to auto-detect"
            )
        super().__init__(name, num_joints=PIPER_NUM_JOINTS, safety=safety)
        self.can_interface = can_interface
        self.bitrate = bitrate
        self.firmware_profile = firmware_profile
        self.home_joints = None if home_joints is None else list(home_joints)
        self.with_gripper = with_gripper
        self._resolved_profile: str | None = None
        self._reported_firmware: str | None = None
        self._effector: Any = None
        self._last_joint_stamp: Any = None
        self._joint_stamp_changed_at = 0.0
        self._last_status_signature: tuple | None = None

    @classmethod
    def from_config(cls, config: "RobotConfig", *, name: str = "arm") -> "AgxArm":
        gripper = config.gripper
        return cls(
            name,
            can_interface=config.can_interface,
            bitrate=config.bitrate,
            firmware_profile=config.firmware_profile,
            home_joints=config.home_joints,
            with_gripper=gripper is not None and gripper.shares_arm_session,
            safety=ArmSafety.from_config(config),
        )

    @property
    def effector(self) -> Any:
        """The SDK gripper driver from ``init_effector``; None unless built with_gripper."""
        return self._effector

    @property
    def resolved_firmware_profile(self) -> str | None:
        return self._resolved_profile

    # ------------------------------------------------------------------ lifecycle

    def _can_hint(self) -> str:
        iface, bitrate = self.can_interface, self.bitrate
        return (
            f"plug in the USB-CAN adapter and run: bash can_activate.sh {iface} {bitrate}  "
            f"(piper_sdk) or: sudo ip link set {iface} up type can bitrate {bitrate}"
        )

    def _check_can_interface(self) -> None:
        from robot_infra.discovery import enumerate as can_discovery

        iface = self.can_interface
        info = can_discovery.list_can_interfaces().get(iface)
        if info is None:
            raise DeviceNotFound(
                f"CAN interface '{iface}' not found", hint=self._can_hint(), device=self.name
            )
        if info.operstate != "up":
            raise DeviceFault(
                f"CAN interface '{iface}' is {info.operstate}, not up",
                hint=self._can_hint(),
                device=self.name,
                recoverable=False,
            )
        if info.bitrate is not None and info.bitrate != self.bitrate:
            raise DeviceFault(
                f"CAN interface '{iface}' runs at {info.bitrate} bit/s, expected {self.bitrate}",
                hint=self._can_hint(),
                device=self.name,
                recoverable=False,
            )

    def _sdk_config(self, sdk: Any, profile: str) -> dict:
        # The vendor spells the keyword "firmeware_version".
        return sdk.create_agx_arm_config(
            robot=PIPER_MODEL,
            firmeware_version=profile,
            channel=self.can_interface,
            interface="socketcan",
            bitrate=self.bitrate,
        )

    def _create_driver(self, sdk: Any, profile: str) -> Any:
        try:
            return sdk.AgxArmFactory.create_arm(self._sdk_config(sdk, profile))
        except Exception as error:
            raise DeviceFault(
                f"pyAgxArm could not build a Piper driver for profile '{profile}': {error}",
                hint=_FIRMWARE_HINT,
                device=self.name,
            ) from error

    def _detect_firmware_profile(self, sdk: Any) -> str:
        probe = self._create_driver(sdk, _PROBE_PROFILE)
        try:
            probe.connect()
            reported = probe.get_firmware()
        except Exception as error:
            raise DeviceFault(
                f"firmware probe on '{self.can_interface}' failed: {error}",
                hint=_FIRMWARE_HINT,
                device=self.name,
            ) from error
        finally:
            probe.disconnect()
        version = (reported or {}).get("software_version")
        if not version:
            raise DeviceFault(
                f"Piper on '{self.can_interface}' did not report a firmware version",
                hint=_FIRMWARE_HINT,
                device=self.name,
            )
        try:
            profile = sdk.resolve_firmware_profile(PIPER_MODEL, version)
        except (TypeError, ValueError) as error:
            raise DeviceFault(
                f"cannot map Piper firmware '{version}' to an SDK profile: {error}",
                hint=_FIRMWARE_HINT,
                device=self.name,
            ) from error
        self._reported_firmware = version
        return profile

    def _open(self) -> Any:
        self._check_can_interface()
        sdk = self.require_sdk()
        profile = self.firmware_profile or self._detect_firmware_profile(sdk)
        driver = self._create_driver(sdk, profile)
        try:
            if self.with_gripper:
                # Once per driver, before connect() starts the read thread.
                self._effector = driver.init_effector(driver.OPTIONS.EFFECTOR.AGX_GRIPPER)
            try:
                driver.connect()
            except Exception as error:
                raise DeviceFault(
                    f"could not open CAN bus '{self.can_interface}': {error}",
                    hint=self._can_hint(),
                    device=self.name,
                ) from error
            self._wait_until(
                driver.enable,
                self.ENABLE_TIMEOUT_S,
                DeviceFault(
                    f"Piper on '{self.can_interface}' did not enable within "
                    f"{self.ENABLE_TIMEOUT_S:g}s",
                    hint=f"check that the arm is powered, that '{self.can_interface}' is up at "
                    f"{self.bitrate} bit/s, and that its emergency stop is released",
                    device=self.name,
                    recoverable=True,
                ),
            )
            self._wait_until(
                lambda: self._missing_feedback_frames(driver) == [],
                self.FEEDBACK_TIMEOUT_S,
                DeviceFault(
                    f"Piper on '{self.can_interface}' enabled but sent no joint feedback within "
                    f"{self.FEEDBACK_TIMEOUT_S:g}s",
                    hint=_FOLLOWER_HINT,
                    device=self.name,
                ),
            )
        except BaseException:
            self._effector = None
            with contextlib.suppress(Exception):
                driver.disconnect()
            raise
        self._resolved_profile = profile
        return driver

    def _release(self, handle: Any) -> None:
        # disconnect() only stops the read thread and closes the bus; the motors keep
        # holding. Never disable()/reset() here: both cut power and drop a raised arm.
        self._effector = None
        self._last_joint_stamp = None
        handle.disconnect()

    def _wait_until(self, condition: Callable[[], bool], timeout_s: float, error: DeviceFault) -> None:
        deadline = time.monotonic() + timeout_s
        while not condition():
            if time.monotonic() >= deadline:
                raise error
            time.sleep(self.POLL_S)

    # ------------------------------------------------------------------ state

    def _no_feedback(self, what: str) -> DeviceFault:
        return DeviceFault(
            f"Piper '{self.name}': no feedback yet ({what})",
            hint=_FOLLOWER_HINT,
            device=self.name,
            recoverable=True,
        )

    @staticmethod
    def _missing_feedback_frames(driver: Any) -> list[str]:
        parser = getattr(driver, "_parser", None)
        return [frame for frame in _FEEDBACK_FRAMES if getattr(parser, frame, None) is None]

    def _fresh_joint_angles(self, driver: Any) -> Any:
        """Joint feedback once every frame has arrived; notes feedback when the joint timestamp
        advances and faults when it has not advanced for FEEDBACK_STALE_S."""
        joints = driver.get_joint_angles()
        missing = self._missing_feedback_frames(driver)
        if joints is None or missing:
            raise self._no_feedback(f"joint angles; missing frames {missing}")
        now = time.monotonic()
        if joints.timestamp != self._last_joint_stamp:
            self._last_joint_stamp = joints.timestamp
            self._joint_stamp_changed_at = now
            self._note_feedback()
        elif now - self._joint_stamp_changed_at > self.FEEDBACK_STALE_S:
            raise DeviceFault(
                f"Piper '{self.name}' feedback is stale "
                f"(>{self.FEEDBACK_STALE_S:g}s without a new joint frame)",
                hint=_FOLLOWER_HINT,
                device=self.name,
                recoverable=True,
            )
        return joints

    def read_state(self) -> dict[str, np.ndarray]:
        """joint_vel comes from the per-joint motor states; it is zeros until every joint
        has reported one (the SDK has no other velocity source)."""
        self._require_connected()
        driver = self._handle
        joints = driver.get_joint_angles()
        if joints is None:
            raise self._no_feedback("joint angles")
        flange = driver.get_flange_pose()
        if flange is None:
            raise self._no_feedback("flange pose")
        motor_states = [driver.get_motor_states(i) for i in range(1, PIPER_NUM_JOINTS + 1)]
        if any(state is None for state in motor_states):
            joint_vel = np.zeros(PIPER_NUM_JOINTS)
        else:
            joint_vel = np.array([state.msg.velocity for state in motor_states], dtype=np.float64)
        if joints.timestamp != self._last_joint_stamp:
            self._last_joint_stamp = joints.timestamp
            self._joint_stamp_changed_at = time.monotonic()
            self._note_feedback()
        pose6 = np.asarray(flange.msg, dtype=np.float64)
        return self._remember_state({
            "tcp_pose": np.concatenate([pose6[:3], rpy_to_quat_xyzw(pose6[3:])]),
            "joint_pos": np.asarray(joints.msg, dtype=np.float64),
            "joint_vel": joint_vel,
        })

    # ------------------------------------------------------------------ commands

    def _require_mode(self, mode: ControlMode, command: str) -> None:
        if self._control_mode is not mode:
            raise DeviceFault(
                f"{command} needs control mode '{mode.value}', arm '{self.name}' is in "
                f"'{self._control_mode.value}'",
                hint=f"call set_control_mode(ControlMode.{mode.name}) first",
                device=self.name,
            )

    @staticmethod
    def _pose6(pose: np.ndarray) -> list[float]:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"pose must have shape (7,), got {pose.shape}")
        return [float(v) for v in (*pose[:3], *quat_xyzw_to_rpy(pose[3:]))]

    def _joints(self, q: np.ndarray) -> list[float]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (PIPER_NUM_JOINTS,):
            raise ValueError(f"joint target must have shape ({PIPER_NUM_JOINTS},), got {q.shape}")
        return [float(v) for v in q]

    def _servo_cartesian(self, pose: np.ndarray) -> None:
        self._require_connected()
        self._require_mode(ControlMode.CARTESIAN_SERVO, "servo_cartesian")
        self._require_servo_ready("servo_cartesian")
        self._handle.move_p(self._pose6(pose))

    def _servo_joints(self, q: np.ndarray) -> None:
        self._require_connected()
        self._require_mode(ControlMode.JOINT_SERVO, "servo_joints")
        self._require_servo_ready("servo_joints")
        self._handle.move_j(self._joints(q))

    def _require_servo_ready(self, command: str) -> None:
        """CAN sends fail silently, so feedback is the only fault signal: refuse to stream a
        target into a faulted arm or on stale feedback (the arm would jump once it clears)."""
        driver = self._handle
        status = driver.get_arm_status()
        if status is not None:
            self._note_status_errors(status.msg)
        faulted = status is not None and int(status.msg.arm_status) != _ARM_STATUS_NORMAL
        if faulted or self._errors:
            raise DeviceFault(
                f"{command} refused: Piper '{self.name}' is faulted ({self._errors})",
                hint=_RECOVER_HINT,
                device=self.name,
                recoverable=True,
            )
        self._fresh_joint_angles(driver)

    def _move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self._require_connected()
        self._require_mode(ControlMode.IDLE, "move_to_pose")
        self._move_and_wait(lambda: self._handle.move_p(self._pose6(pose)), blocking)

    def _move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self._require_connected()
        self._require_mode(ControlMode.IDLE, "move_to_joints")
        self._move_and_wait(lambda: self._handle.move_j(self._joints(q)), blocking)

    def _move_and_wait(self, send: Callable[[], None], blocking: bool) -> None:
        """Arrival is ``motion_status == 0`` on a status frame newer than the one seen
        before sending and after a short settle: right after a new target the previous
        motion's 'reached' status is still cached."""
        driver = self._handle
        before = driver.get_arm_status()
        stale_stamp = None if before is None else before.timestamp
        send()
        if not blocking:
            return
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            status = driver.get_arm_status()
            if status is not None and status.timestamp != stale_stamp:
                if int(status.msg.arm_status) != _ARM_STATUS_NORMAL:
                    raise DeviceFault(
                        f"Piper '{self.name}' reports {_enum_name(status.msg.arm_status)} "
                        "while moving",
                        hint=_RECOVER_HINT,
                        device=self.name,
                        recoverable=True,
                    )
                if elapsed >= self.MOTION_SETTLE_S and int(status.msg.motion_status) == _MOTION_REACHED:
                    return
            if elapsed >= self.MOTION_TIMEOUT_S:
                raise DeviceFault(
                    f"Piper '{self.name}' did not reach its target within {self.MOTION_TIMEOUT_S:g}s",
                    hint="check the target is reachable and inside the joint limits",
                    device=self.name,
                    recoverable=True,
                )
            time.sleep(self.POLL_S)

    def stop(self) -> None:
        """pyAgxArm has no hold command (``electronic_emergency_stop`` is a damped descent),
        so stopping re-sends the current joint angles as the ``move_j`` target."""
        self._require_connected()
        joints = self._fresh_joint_angles(self._handle)
        self._handle.move_j([float(v) for v in joints.msg])

    def recover(self) -> None:
        self._require_connected()
        if not self._handle.clear_joint_error():
            raise DeviceFault(
                f"Piper '{self.name}' did not acknowledge clear_joint_error",
                hint="release e-stop / clear errors / power-cycle",
                device=self.name,
            )
        self._last_status_signature = None
        self._clear_errors()

    def _enter_control_mode(self, mode: ControlMode) -> None:
        # move_p/move_j switch the firmware motion mode themselves; nothing to arm here.
        pass

    # ------------------------------------------------------------------ diagnostics

    def _note_status_errors(self, status_msg: Any) -> None:
        signature = (int(status_msg.arm_status), int(status_msg.err_code))
        if signature == self._last_status_signature:
            return
        self._last_status_signature = signature
        messages = []
        if signature[0] != _ARM_STATUS_NORMAL:
            messages.append(f"arm_status={_enum_name(status_msg.arm_status)}")
        messages.extend(
            flag for flag in _ERROR_FLAGS if getattr(status_msg.err_status, flag, False)
        )
        for message in messages:
            if message not in self._errors:
                self._note_error(message)

    def health(self) -> ArmHealth:
        if self.is_connected:
            status = self._handle.get_arm_status()
            if status is not None:
                self._robot_mode = _enum_name(status.msg.ctrl_mode)
                self._note_status_errors(status.msg)
        return super().health()

    def preflight(self) -> list["CheckResult"]:
        from robot_infra.doctor.checks import CheckResult

        self._require_connected()
        driver = self._handle
        status = driver.get_arm_status()
        if status is None:
            status_ok, status_message = False, "no arm status frame received"
        else:
            enabled = all(driver.get_joints_enable_status_list())
            status_ok = int(status.msg.arm_status) == _ARM_STATUS_NORMAL and enabled
            status_message = (
                f"arm_status={_enum_name(status.msg.arm_status)}, "
                f"ctrl_mode={_enum_name(status.msg.ctrl_mode)}, joints enabled={enabled}"
            )
        feedback_ok = driver.get_joint_angles() is not None and driver.is_ok()
        firmware = self._resolved_profile or "unknown"
        if self._reported_firmware:
            firmware_message = f"profile '{firmware}' (arm reports {self._reported_firmware})"
        else:
            firmware_message = f"profile '{firmware}' (set in config)"
        return [
            CheckResult(
                5,
                "mode:piper_status",
                status_ok,
                status_message,
                "" if status_ok else "release e-stop / clear errors / power-cycle",
            ),
            CheckResult(
                5,
                "mode:follower",
                feedback_ok,
                "joint feedback flowing" if feedback_ok else "no joint feedback",
                "" if feedback_ok else _FOLLOWER_HINT,
            ),
            CheckResult(5, "mode:firmware_profile", True, firmware_message),
        ]

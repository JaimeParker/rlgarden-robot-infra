"""AgileX Piper arm over the vendor ``piper_sdk`` (``C_PiperInterface_V2``, SocketCAN).

The SDK speaks integer wire units: joints/orientation in 0.001 deg, positions in
0.001 mm, joint speed in 0.001 rad/s. This module converts to the SI contract
(rad, m, rad/s, quaternion xyzw).

Releasing the session only calls ``DisconnectPort()``. ``DisableArm``/``ResetPiper``
cut motor power and the arm falls, so they are never sent from here.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_infra.devices.arm import Arm, ArmHealth, ControlMode
from robot_infra.devices.backends.arms._rotations import quat_xyzw_to_rpy, rpy_to_quat_xyzw
from robot_infra.devices.errors import DeviceFault, DeviceNotFound
from robot_infra.discovery.enumerate import list_can_interfaces

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig
    from robot_infra.doctor.checks import CheckResult

NUM_JOINTS = 6
_CTRL_MODE_CAN = 0x01
_MOVE_P = 0x00
_MOVE_J = 0x01
_MILLI_DEG_PER_RAD = 1000.0 * 180.0 / np.pi
_MILLI_MM_PER_M = 1e6
_MILLI_PER_UNIT = 1e-3
_ARM_STATUS_NORMAL = 0x00
_ARM_STATUS_NAMES = {
    0x00: "normal",
    0x01: "emergency stop",
    0x02: "no inverse-kinematics solution",
    0x03: "singularity",
    0x04: "target angle over limit",
    0x05: "joint communication error",
    0x06: "joint brake not released",
    0x07: "collision",
    0x08: "teaching overspeed",
    0x09: "joint status error",
    0x0A: "other error",
    0x0E: "main-controller over-temperature",
    0x0F: "release-resistor over-temperature",
}
_MOTION_REACHED = 0x00

NO_FEEDBACK_HINT = (
    "no CAN feedback: the arm may be in master/teach mode; switch to follower (slave) "
    "mode; also check the gripper stroke is configured"
)


def _can_hint(can_interface: str, bitrate: int) -> str:
    return (
        f"bash can_activate.sh {can_interface} {bitrate}  (piper_sdk/scripts) or "
        f"sudo ip link set {can_interface} up type can bitrate {bitrate}"
    )


@Arm.register("piper_sdk")
class PiperSdkArm(Arm):
    """6-DoF Piper follower arm. ``stop()`` re-sends the current joints (the SDK has no hold command)."""

    SDK = "piper_sdk"
    SUPPORTED_STATE_FIELDS = ("tcp_pose", "joint_pos", "joint_vel")

    ENABLE_TIMEOUT_S = 5.0
    FIRST_FEEDBACK_TIMEOUT_S = 2.0
    FEEDBACK_STALE_S = 1.0
    # Joint frames 12/34/56 each bump the shared joint timestamp; the C_PiperInterface_V2
    # singleton keeps the previous session's cached frames, so feedback counts only after
    # this many advances following ConnectPort().
    FIRST_FEEDBACK_MIN_STAMP_ADVANCES = 3
    MOTION_TIMEOUT_S = 15.0
    MOTION_SETTLE_S = 0.5
    POLL_PERIOD_S = 0.02
    DISCONNECT_JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        name: str = "arm",
        *,
        can_interface: str = "can0",
        bitrate: int = 1_000_000,
        home_joints: list[float] | None = None,
        move_speed_percent: int = 30,
    ) -> None:
        super().__init__(name, num_joints=NUM_JOINTS)
        self._can_interface = can_interface
        self._bitrate = bitrate
        self._home_joints = home_joints
        self._move_speed_percent = int(move_speed_percent)
        self._last_joint_stamp: float | None = None
        self._joint_stamp_changed_at = 0.0
        self._reported_status: tuple | None = None

    @classmethod
    def from_config(cls, config: "RobotConfig", *, name: str = "arm") -> "PiperSdkArm":
        return cls(
            name,
            can_interface=config.can_interface,
            bitrate=config.bitrate,
            home_joints=config.home_joints,
        )

    # -- lifecycle -------------------------------------------------------------------

    def _open(self) -> Any:
        self._check_can_interface()
        sdk = self.require_sdk()
        iface = self._can_interface
        # The ctor hard-codes 1 Mbit/s socketcan, so judge_flag=False: the precheck above
        # already reported CAN problems with an actionable hint.
        piper = sdk.C_PiperInterface_V2(iface, judge_flag=False)
        piper.ConnectPort()
        try:
            stamp_at_connect = piper.GetArmJointMsgs().time_stamp
            self._enable(piper)
            self._wait_first_feedback(piper, stamp_at_connect)
        except BaseException:
            self._disconnect_port(piper)
            raise
        self._last_joint_stamp = None
        self._reported_status = None
        return piper

    def _release(self, piper: Any) -> None:
        # Deliberately no DisableArm()/ResetPiper(): both drop motor power and the arm falls.
        self._close_port(piper)

    def _close_port(self, piper: Any) -> None:
        piper.DisconnectPort(thread_timeout=self.DISCONNECT_JOIN_TIMEOUT_S)
        # The vendor DisconnectPort only joins its reader when hasattr(self, 'can_deal_th'),
        # which never matches the name-mangled attribute, so the old reader would outlive the
        # session and a reconnect would keep reading frozen cached frames. Join it here.
        reader = getattr(piper, "_C_PiperInterface_V2__can_deal_th", None)
        if reader is not None and reader.is_alive():
            reader.join(timeout=self.DISCONNECT_JOIN_TIMEOUT_S)

    def _disconnect_port(self, piper: Any) -> None:
        try:
            self._close_port(piper)
        except Exception:
            pass

    def _check_can_interface(self) -> None:
        iface, bitrate = self._can_interface, self._bitrate
        info = list_can_interfaces().get(iface)
        if info is None:
            raise DeviceNotFound(
                f"CAN interface '{iface}' not found",
                hint=_can_hint(iface, bitrate),
                device=self.name,
            )
        if info.operstate != "up":
            raise DeviceFault(
                f"CAN interface '{iface}' is {info.operstate}, expected up",
                hint=_can_hint(iface, bitrate),
                device=self.name,
            )
        if info.bitrate is not None and info.bitrate != bitrate:
            raise DeviceFault(
                f"CAN interface '{iface}' bitrate is {info.bitrate}, expected {bitrate}",
                hint=_can_hint(iface, bitrate),
                device=self.name,
            )

    def _enable(self, piper: Any) -> None:
        deadline = time.monotonic() + self.ENABLE_TIMEOUT_S
        # EnablePiper() sends the enable frame and returns the PREVIOUS enable snapshot,
        # so it has to be called repeatedly (as in the vendor demos).
        while not piper.EnablePiper():
            if time.monotonic() >= deadline:
                iface = self._can_interface
                raise DeviceFault(
                    f"Piper on '{iface}' did not enable within {self.ENABLE_TIMEOUT_S:g}s",
                    hint=f"check that the arm is powered, that '{iface}' is up at "
                    f"{self._bitrate} bit/s, and that its emergency stop is released",
                    device=self.name,
                    recoverable=True,
                )
            time.sleep(self.POLL_PERIOD_S / 2)

    def _wait_first_feedback(self, piper: Any, stamp_at_connect: float) -> None:
        """Wait until every joint frame has arrived in this session: the joint timestamp must
        advance FIRST_FEEDBACK_MIN_STAMP_ADVANCES times after ConnectPort()."""
        deadline = time.monotonic() + self.FIRST_FEEDBACK_TIMEOUT_S
        last_stamp = stamp_at_connect
        advances = 0
        while advances < self.FIRST_FEEDBACK_MIN_STAMP_ADVANCES:
            joint_msgs = piper.GetArmJointMsgs()
            if joint_msgs.time_stamp != last_stamp and self._has_feedback(joint_msgs):
                last_stamp = joint_msgs.time_stamp
                advances += 1
                continue
            if time.monotonic() >= deadline:
                raise self._no_feedback_fault()
            time.sleep(self.POLL_PERIOD_S / 2)

    @staticmethod
    def _has_feedback(joint_msgs: Any) -> bool:
        # All-zero joints with no timestamp is the SDK's never-received default object.
        return joint_msgs.time_stamp != 0 or any(PiperSdkArm._joint_ints(joint_msgs))

    @staticmethod
    def _joint_ints(joint_msgs: Any) -> list[int]:
        state = joint_msgs.joint_state
        return [getattr(state, f"joint_{i}") for i in range(1, NUM_JOINTS + 1)]

    def _no_feedback_fault(self) -> DeviceFault:
        return DeviceFault(
            f"no joint feedback from Piper on '{self._can_interface}'",
            hint=NO_FEEDBACK_HINT,
            device=self.name,
            recoverable=True,
        )

    # -- state -----------------------------------------------------------------------

    def read_state(self) -> dict[str, np.ndarray]:
        self._require_connected()
        piper = self._handle
        joint_msgs = piper.GetArmJointMsgs()
        self._check_fresh(joint_msgs)
        joint_pos = np.array(self._joint_ints(joint_msgs), dtype=np.float64) / _MILLI_DEG_PER_RAD
        pose = piper.GetArmEndPoseMsgs().end_pose
        position = (
            np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / _MILLI_MM_PER_M
        )
        rpy = np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / _MILLI_DEG_PER_RAD
        motors = piper.GetArmHighSpdInfoMsgs()
        joint_vel = (
            np.array(
                [getattr(motors, f"motor_{i}").motor_speed for i in range(1, NUM_JOINTS + 1)],
                dtype=np.float64,
            )
            * _MILLI_PER_UNIT
        )
        return {
            "tcp_pose": np.concatenate([position, rpy_to_quat_xyzw(rpy)]),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def _check_fresh(self, joint_msgs: Any) -> None:
        """Fault when no joint frame ever arrived, or its timestamp stopped advancing.

        The message timestamp comes from the CAN driver, so only its change is used,
        never its absolute value.
        """
        if not self._has_feedback(joint_msgs):
            raise self._no_feedback_fault()
        now = time.monotonic()
        stamp = joint_msgs.time_stamp
        if stamp != self._last_joint_stamp:
            self._last_joint_stamp = stamp
            self._joint_stamp_changed_at = now
            self._note_feedback()
        elif now - self._joint_stamp_changed_at > self.FEEDBACK_STALE_S:
            raise DeviceFault(
                f"Piper feedback on '{self._can_interface}' is stale "
                f"(>{self.FEEDBACK_STALE_S:g}s without a joint frame)",
                hint=NO_FEEDBACK_HINT,
                device=self.name,
                recoverable=True,
            )

    # -- commands --------------------------------------------------------------------

    def servo_cartesian(self, pose: np.ndarray) -> None:
        self._require_connected()
        self._require_mode(ControlMode.CARTESIAN_SERVO, "servo_cartesian")
        pose = self._checked(pose, (7,))
        self._require_servo_ready("servo_cartesian")
        self._send_pose(self._handle, pose)

    def servo_joints(self, q: np.ndarray) -> None:
        self._require_connected()
        self._require_mode(ControlMode.JOINT_SERVO, "servo_joints")
        q = self._checked(q, (NUM_JOINTS,))
        self._require_servo_ready("servo_joints")
        self._send_joints(self._handle, q)

    def _require_servo_ready(self, command: str) -> None:
        """Refuse to stream a target into a faulted arm or on stale feedback: CAN sends never
        report failure, and the arm would jump to the latest target once the fault clears."""
        self._refresh_status()
        if self._errors:
            raise DeviceFault(
                f"{command} refused: Piper on '{self._can_interface}' is faulted ({self._errors})",
                hint="clear the cause shown by health() (e-stop, collision, limits), then retry",
                device=self.name,
                recoverable=True,
            )
        self._check_fresh(self._handle.GetArmJointMsgs())

    def move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self._require_connected()
        self._require_mode(ControlMode.IDLE, "move_to_pose")
        pose = self._checked(pose, (7,))
        self._move(lambda: self._send_pose(self._handle, pose), blocking)

    def move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self._require_connected()
        self._require_mode(ControlMode.IDLE, "move_to_joints")
        q = self._checked(q, (NUM_JOINTS,))
        self._move(lambda: self._send_joints(self._handle, q), blocking)

    def _move(self, send: Any, blocking: bool) -> None:
        stale_stamp = self._handle.GetArmStatus().time_stamp
        send()
        if not blocking:
            return
        start = time.monotonic()
        deadline = start + self.MOTION_TIMEOUT_S
        # The command is re-sent while waiting, as the vendor demos stream it. Right after a new
        # target the cached status still reports the previous target as reached, so arrival
        # needs a status frame newer than the one seen before sending, after a settle period.
        while True:
            time.sleep(self.POLL_PERIOD_S)
            status = self._handle.GetArmStatus()
            if (
                status.time_stamp != stale_stamp
                and time.monotonic() - start >= self.MOTION_SETTLE_S
                and status.arm_status.motion_status == _MOTION_REACHED
            ):
                return
            if time.monotonic() >= deadline:
                raise DeviceFault(
                    f"Piper did not reach its target within {self.MOTION_TIMEOUT_S:g}s",
                    hint="check the target is inside the workspace and joint limits, and that "
                    "the arm status is normal (health())",
                    device=self.name,
                    recoverable=True,
                )
            send()

    def _send_pose(self, piper: Any, pose: np.ndarray) -> None:
        x, y, z = np.rint(pose[:3] * _MILLI_MM_PER_M).astype(int).tolist()
        rx, ry, rz = np.rint(quat_xyzw_to_rpy(pose[3:]) * _MILLI_DEG_PER_RAD).astype(int).tolist()
        piper.MotionCtrl_2(_CTRL_MODE_CAN, _MOVE_P, self._move_speed_percent)
        piper.EndPoseCtrl(x, y, z, rx, ry, rz)

    def _send_joints(self, piper: Any, q: np.ndarray) -> None:
        self._send_joint_ints(piper, np.rint(q * _MILLI_DEG_PER_RAD).astype(int).tolist())

    def _send_joint_ints(self, piper: Any, joint_ints: list[int]) -> None:
        piper.MotionCtrl_2(_CTRL_MODE_CAN, _MOVE_J, self._move_speed_percent)
        piper.JointCtrl(*joint_ints)

    def stop(self) -> None:
        """Hold position by re-sending the measured joints; piper_sdk has no stop/hold command
        (``MotionCtrl_1`` emergency stop and ``ResetPiper`` both power the motors down)."""
        self._require_connected()
        joint_msgs = self._handle.GetArmJointMsgs()
        # Faults instead of commanding missing (zero) or frozen joints as the hold target.
        self._check_fresh(joint_msgs)
        self._send_joint_ints(self._handle, self._joint_ints(joint_msgs))

    def recover(self) -> None:
        """piper_sdk exposes no fault-clear that is safe to send (the reset frame drops the
        arm), so this only re-reads the arm status: errors stay while the arm is still faulted;
        fix the cause shown by health()."""
        self._require_connected()
        self._reported_status = None
        self._clear_errors()
        self._refresh_status()

    def _enter_control_mode(self, mode: ControlMode) -> None:
        # Every command frame is preceded by MotionCtrl_2, which selects the MOVE mode.
        pass

    def _require_mode(self, required: ControlMode, operation: str) -> None:
        if self._control_mode is not required:
            raise DeviceFault(
                f"{operation} needs control mode '{required.value}', arm is in "
                f"'{self._control_mode.value}'",
                hint=f"call set_control_mode(ControlMode.{required.name}) first"
                if required is not ControlMode.IDLE
                else "call set_control_mode(ControlMode.IDLE) first; blocking moves are "
                "not available while servoing",
                device=self.name,
            )

    @staticmethod
    def _checked(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"expected shape {shape}, got {value.shape}")
        return value

    # -- diagnostics -----------------------------------------------------------------

    def health(self) -> ArmHealth:
        if self.is_connected:
            self._refresh_status()
        return super().health()

    def _refresh_status(self) -> None:
        status = self._handle.GetArmStatus().arm_status
        self._robot_mode = f"ctrl_mode={status.ctrl_mode} mode_feed={status.mode_feed}"
        errors = []
        if status.arm_status != _ARM_STATUS_NORMAL:
            name = _ARM_STATUS_NAMES.get(status.arm_status, f"code {status.arm_status:#x}")
            errors.append(f"arm_status: {name}")
        err = status.err_status
        for joint in range(1, NUM_JOINTS + 1):
            if getattr(err, f"joint_{joint}_angle_limit", False):
                errors.append(f"joint {joint} over angle limit")
            if getattr(err, f"communication_status_joint_{joint}", False):
                errors.append(f"joint {joint} communication error")
        # Keyed on the arm status and the joint flags together: a joint flag can change
        # while arm_status stays the same.
        signature = (status.arm_status, tuple(errors))
        if signature == self._reported_status:
            return
        self._reported_status = signature
        self._clear_errors()
        for message in errors:
            self._note_error(message)

    def preflight(self) -> list["CheckResult"]:
        from robot_infra.doctor.checks import CheckResult

        layer = 5
        piper = self._handle
        status = piper.GetArmStatus().arm_status
        status_ok = status.arm_status == _ARM_STATUS_NORMAL
        status_name = _ARM_STATUS_NAMES.get(status.arm_status, f"code {status.arm_status:#x}")
        feedback_ok = self._has_feedback(piper.GetArmJointMsgs())
        gripper = piper.GetArmGripperMsgs()
        gripper_flowing = gripper.time_stamp != 0 or gripper.gripper_state.grippers_angle != 0
        return [
            CheckResult(
                layer,
                "mode:piper_status",
                status_ok,
                f"arm_status: {status_name}",
                "" if status_ok else "release the emergency stop / clear the fault on the "
                "teach pendant, then reconnect",
            ),
            CheckResult(
                layer,
                "mode:follower",
                feedback_ok,
                "joint feedback flowing" if feedback_ok else "no joint feedback",
                "" if feedback_ok else NO_FEEDBACK_HINT,
            ),
            CheckResult(
                layer,
                "mode:gripper_stroke",
                True,
                "gripper feedback flowing"
                if gripper_flowing
                else "no gripper feedback (informational: no gripper fitted, or stroke not configured)",
                "" if gripper_flowing else "if a gripper is fitted, configure its stroke "
                "(0.07 or 0.1 m) on the teach pendant",
                skipped=not gripper_flowing,
            ),
        ]

"""Hardware-free tests for the pyAgxArm Piper backend. A fake ``pyAgxArm`` module
mirrors the SDK names the backend uses and records every driver call."""
from __future__ import annotations

import enum
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from robot_infra.devices import Arm, ControlMode, DeviceFault, DeviceNotFound, Gripper, Robot
from robot_infra.devices.backends.arms._rotations import rpy_to_quat_xyzw
from robot_infra.devices.backends.arms.pyagxarm import AgxArm
from robot_infra.devices.backends.grippers.pyagxarm import AgxGripper
from robot_infra.devices.robots.piper import PiperConfig
from robot_infra.discovery import enumerate as can_discovery
from robot_infra.discovery.enumerate import CanInfo

JOINTS = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
FLANGE_RPY = [0.1, -0.2, 0.3]
FLANGE = [0.25, -0.05, 0.3, *FLANGE_RPY]
POWER_CUTTING_CALLS = {"disable", "reset", "electronic_emergency_stop"}


class ArmStatus(enum.IntEnum):
    NORMAL = 0x00
    EMERGENCY_STOP = 0x01
    NO_SOLUTION = 0x02


class CtrlMode(enum.IntEnum):
    STANDBY = 0x00
    CAN_CTRL = 0x01


def make_status(arm_status=ArmStatus.NORMAL, motion_status=0, err_code=0):
    flags = {f"joint_{i}_angle_limit": bool(err_code & (1 << (i + 7))) for i in range(1, 7)}
    flags.update(
        {f"communication_status_joint_{i}": bool(err_code & (1 << (i - 1))) for i in range(1, 7)}
    )
    return SimpleNamespace(
        ctrl_mode=CtrlMode.CAN_CTRL,
        arm_status=arm_status,
        motion_status=motion_status,
        err_code=err_code,
        err_status=SimpleNamespace(**flags),
    )


class FakeSdk(types.ModuleType):
    """Stand-in for the ``pyAgxArm`` package plus the knobs a test turns."""

    def __init__(self):
        super().__init__("pyAgxArm")
        self.log: list[tuple] = []
        self.firmware = "S-V1.8-8"
        self.enable_after = 1  # enable() returns True from this call on; None never
        self.feedback = True
        self.motor_states = True
        self.missing_frames: set[str] = set()
        self.advance_joint_stamp = True
        self.status_sequence = [make_status()]
        self.advance_status_stamp = True
        self.gripper_status = SimpleNamespace(value=0.035, force=1.0, mode="width")
        self.drivers: list[FakeDriver] = []
        sdk = self

        class AgxArmFactory:
            @staticmethod
            def create_arm(config, **kwargs):
                driver = FakeDriver(sdk, config)
                sdk.drivers.append(driver)
                return driver

        self.AgxArmFactory = AgxArmFactory

    def create_agx_arm_config(self, robot, comm="can", firmeware_version="default", **kwargs):
        assert set(kwargs) == {"channel", "interface", "bitrate"}
        return {"robot": robot, "firmeware_version": firmeware_version, "can": kwargs}

    def resolve_firmware_profile(self, robot, firmware_version):
        assert robot == "piper"
        return {"S-V1.8-2": "default", "S-V1.8-8": "v188", "S-V1.8-9": "v189"}[firmware_version]

    def calls(self, name):
        return [entry for entry in self.log if entry[0] == name]


class FakeEffector:
    def __init__(self, sdk):
        self.sdk = sdk

    def move_gripper_m(self, value=0.0, force=1.0):
        self.sdk.log.append(("move_gripper_m", value, force))

    def get_gripper_status(self):
        status = self.sdk.gripper_status
        return None if status is None else SimpleNamespace(msg=status, timestamp=1.0)


class FakeDriver:
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx_gripper"))

    def __init__(self, sdk, config):
        self.sdk = sdk
        self.profile = config["firmeware_version"]
        self.connected = False
        self.enable_calls = 0
        self.status_index = 0
        self.joint_stamp = 1.0
        sdk.log.append(("create", self.profile, config["can"]["channel"]))

    def init_effector(self, effector):
        assert not self.connected, "init_effector must run before connect()"
        self.sdk.log.append(("init_effector", effector))
        return FakeEffector(self.sdk)

    def connect(self, start_read_thread=True):
        self.sdk.log.append(("connect", self.profile))
        self.connected = True

    def disconnect(self, join_timeout=1.0):
        self.sdk.log.append(("disconnect", self.profile))
        self.connected = False

    def get_firmware(self, timeout=1.0, min_interval=1.0):
        self.sdk.log.append(("get_firmware",))
        return None if self.sdk.firmware is None else {"software_version": self.sdk.firmware}

    def enable(self, joint_index=255):
        self.enable_calls += 1
        if self.enable_calls == 1:
            self.sdk.log.append(("enable",))
        return self.sdk.enable_after is not None and self.enable_calls >= self.sdk.enable_after

    @property
    def _parser(self):
        frames = ("joint_12", "joint_34", "joint_56", "end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
        return SimpleNamespace(
            **{
                frame: (object() if self.sdk.feedback and frame not in self.sdk.missing_frames else None)
                for frame in frames
            }
        )

    def get_joint_angles(self):
        if not self.sdk.feedback:
            return None
        if self.sdk.advance_joint_stamp:
            self.joint_stamp += 0.005
        return SimpleNamespace(msg=list(JOINTS), timestamp=self.joint_stamp, hz=200.0)

    def get_flange_pose(self):
        if not self.sdk.feedback:
            return None
        return SimpleNamespace(msg=list(FLANGE), timestamp=self.joint_stamp, hz=200.0)

    def get_motor_states(self, joint_index):
        if not self.sdk.motor_states:
            return None
        return SimpleNamespace(msg=SimpleNamespace(velocity=0.01 * joint_index), timestamp=1.0)

    def get_arm_status(self):
        sequence = self.sdk.status_sequence
        status = sequence[min(self.status_index, len(sequence) - 1)]
        stamp = float(self.status_index) if self.sdk.advance_status_stamp else 0.0
        self.status_index += 1
        self.sdk.log.append(("get_arm_status",))
        return SimpleNamespace(msg=status, timestamp=stamp)

    def get_joints_enable_status_list(self):
        return [True] * 6

    def is_ok(self):
        return self.sdk.feedback

    def move_p(self, pose):
        assert isinstance(pose, list) and len(pose) == 6
        self.sdk.log.append(("move_p", list(pose)))

    def move_j(self, joints):
        assert isinstance(joints, list) and len(joints) == 6
        self.sdk.log.append(("move_j", list(joints)))

    def clear_joint_error(self, joint_index=255, timeout=1.0):
        self.sdk.log.append(("clear_joint_error",))
        return True

    def disable(self, joint_index=255):
        self.sdk.log.append(("disable",))

    def reset(self):
        self.sdk.log.append(("reset",))

    def electronic_emergency_stop(self):
        self.sdk.log.append(("electronic_emergency_stop",))


def set_can(monkeypatch, interfaces):
    monkeypatch.setattr(can_discovery, "list_can_interfaces", lambda: interfaces)


@pytest.fixture
def sdk(monkeypatch):
    fake = FakeSdk()
    monkeypatch.setitem(sys.modules, "pyAgxArm", fake)
    set_can(monkeypatch, {"can0": CanInfo("can0", "up", 1_000_000)})
    monkeypatch.setattr(AgxArm, "ENABLE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(AgxArm, "FEEDBACK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(AgxArm, "MOTION_TIMEOUT_S", 0.05)
    monkeypatch.setattr(AgxArm, "MOTION_SETTLE_S", 0.0)
    monkeypatch.setattr(AgxArm, "POLL_S", 0.001)
    return fake


@pytest.fixture
def arm(sdk):
    arm = AgxArm(firmware_profile="v189")
    arm.connect()
    yield arm
    arm.disconnect()


def test_registry_resolves_pyagxarm():
    assert Arm.backend("pyagxarm") is AgxArm
    assert Gripper.backend("pyagxarm") is AgxGripper


def test_unknown_firmware_profile_rejected():
    with pytest.raises(ValueError, match="v188"):
        AgxArm(firmware_profile="v999")


@pytest.mark.parametrize(
    "interfaces, error_type, message",
    [
        ({}, DeviceNotFound, "CAN interface 'can0' not found"),
        ({"can0": CanInfo("can0", "down", 1_000_000)}, DeviceFault, "is down"),
        ({"can0": CanInfo("can0", "up", 500_000)}, DeviceFault, "500000 bit/s, expected 1000000"),
    ],
)
def test_can_precheck_failures_carry_bringup_hint(sdk, monkeypatch, interfaces, error_type, message):
    set_can(monkeypatch, interfaces)
    arm = AgxArm(firmware_profile="v189")
    with pytest.raises(error_type, match=message) as info:
        arm.connect()
    assert "bash can_activate.sh can0 1000000" in info.value.hint
    assert "sudo ip link set can0 up type can bitrate 1000000" in info.value.hint
    if error_type is DeviceFault:
        assert info.value.recoverable is False
    assert sdk.log == []
    assert not arm.is_connected


def test_unknown_bitrate_passes_precheck(sdk, monkeypatch):
    set_can(monkeypatch, {"can0": CanInfo("can0", "up", None)})
    arm = AgxArm(firmware_profile="v189")
    arm.connect()
    assert arm.is_connected


def test_connect_probes_firmware_then_inits_effector_before_connect(sdk):
    arm = AgxArm(with_gripper=True)
    arm.connect()
    assert sdk.log == [
        ("create", "default", "can0"),
        ("connect", "default"),
        ("get_firmware",),
        ("disconnect", "default"),
        ("create", "v188", "can0"),
        ("init_effector", "agx_gripper"),
        ("connect", "v188"),
        ("enable",),
    ]
    assert arm.resolved_firmware_profile == "v188"
    assert arm.effector is not None


def test_explicit_firmware_profile_skips_probe(sdk):
    arm = AgxArm(firmware_profile="v183")
    arm.connect()
    assert sdk.calls("get_firmware") == []
    assert sdk.calls("create") == [("create", "v183", "can0")]
    assert sdk.calls("init_effector") == []


def test_firmware_probe_without_version_hints_explicit_profile(sdk):
    sdk.firmware = None
    with pytest.raises(DeviceFault, match="did not report a firmware version") as info:
        AgxArm().connect()
    assert "firmware_profile" in info.value.hint
    assert sdk.log[-1] == ("disconnect", "default")
    assert len(sdk.drivers) == 1


def test_enable_timeout_hint_and_cleanup(sdk):
    sdk.enable_after = None
    arm = AgxArm(firmware_profile="v189", with_gripper=True)
    with pytest.raises(DeviceFault, match="did not enable within") as info:
        arm.connect()
    assert info.value.recoverable is True
    assert "emergency stop is released" in info.value.hint
    assert "'can0' is up at 1000000 bit/s" in info.value.hint
    assert sdk.log[-1] == ("disconnect", "v189")
    assert not POWER_CUTTING_CALLS & {entry[0] for entry in sdk.log}
    assert not arm.is_connected and arm.effector is None


def test_no_feedback_hint_names_follower_mode(sdk):
    sdk.feedback = False
    arm = AgxArm(firmware_profile="v189")
    with pytest.raises(DeviceFault) as info:
        arm.connect()
    assert "follower (slave) mode" in info.value.hint
    assert sdk.log[-1] == ("disconnect", "v189")
    assert not arm.is_connected


def test_read_state_shapes_units_and_quaternion(arm):
    state = arm.read_state()
    assert set(state) == {"tcp_pose", "joint_pos", "joint_vel"}
    assert state["tcp_pose"].shape == (7,)
    assert np.allclose(state["tcp_pose"][:3], FLANGE[:3])
    # extrinsic xyz: q = q_z(yaw) * q_y(pitch) * q_x(roll)
    r, p, y = (v / 2 for v in FLANGE_RPY)
    qx = np.array([np.sin(r), 0, 0, np.cos(r)])
    qy = np.array([0, np.sin(p), 0, np.cos(p)])
    qz = np.array([0, 0, np.sin(y), np.cos(y)])

    def mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    expected = mul(mul(qz, qy), qx)
    quat = state["tcp_pose"][3:]
    assert np.allclose(quat, expected, atol=1e-12) or np.allclose(quat, -expected, atol=1e-12)
    assert np.allclose(state["joint_pos"], JOINTS)
    assert np.allclose(state["joint_vel"], [0.01 * i for i in range(1, 7)])
    assert arm.health().first_feedback_received
    observation = arm.get_observation()
    assert {key: value.shape for key, value in observation.items()} == {
        "tcp_pose": (7,),
        "joint_pos": (6,),
        "joint_vel": (6,),
    }
    assert all(value.dtype == np.float32 for value in observation.values())


def test_joint_vel_zeros_without_motor_states(arm, sdk):
    sdk.motor_states = False
    assert np.array_equal(arm.read_state()["joint_vel"], np.zeros(6))


def test_read_state_without_feedback_is_recoverable(arm, sdk):
    sdk.feedback = False
    with pytest.raises(DeviceFault, match="no feedback yet") as info:
        arm.read_state()
    assert info.value.recoverable is True
    assert "follower" in info.value.hint


def test_set_control_mode_stop_resends_current_joints(arm, sdk):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    assert sdk.calls("move_j") == [("move_j", JOINTS)]


def test_servo_cartesian_round_trips_rpy(arm, sdk):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    pose = np.array([0.2, 0.1, 0.35, *rpy_to_quat_xyzw(np.array(FLANGE_RPY))])
    arm.send_action({"tcp_pose": pose})
    (_, sent), = sdk.calls("move_p")
    assert np.allclose(sent[:3], pose[:3], atol=1e-12)
    assert np.allclose(sent[3:], FLANGE_RPY, atol=1e-9)


def test_servo_joints_sends_move_j(arm, sdk):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.servo_joints(np.array([0.0, 0.5, -0.5, 0.0, 0.2, 0.0]))
    assert sdk.calls("move_j")[-1] == ("move_j", [0.0, 0.5, -0.5, 0.0, 0.2, 0.0])


def test_servo_rejected_outside_its_mode(arm):
    with pytest.raises(DeviceFault, match="cartesian_servo"):
        arm.servo_cartesian(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    with pytest.raises(DeviceFault, match="joint_servo"):
        arm.servo_joints(np.zeros(6))


def test_move_to_joints_blocks_until_motion_reached(arm, sdk):
    sdk.status_sequence = [make_status(motion_status=s) for s in (0, 1, 1, 0)]
    arm.move_to_joints(np.zeros(6))
    assert sdk.calls("move_j") == [("move_j", [0.0] * 6)]
    assert len(sdk.calls("get_arm_status")) == 4


def test_move_ignores_stale_reached_status_and_times_out(arm, sdk):
    sdk.advance_status_stamp = False
    with pytest.raises(DeviceFault, match="did not reach its target") as info:
        arm.move_to_joints(np.zeros(6))
    assert info.value.recoverable is True


def test_move_to_pose_times_out_when_never_reached(arm, sdk):
    sdk.status_sequence = [make_status(motion_status=1)]
    with pytest.raises(DeviceFault, match="did not reach its target"):
        arm.move_to_pose(np.array([0.2, 0.0, 0.3, 0, 0, 0, 1.0]))
    assert len(sdk.calls("move_p")) == 1


def test_move_fails_fast_on_arm_status_error(arm, sdk):
    sdk.status_sequence = [make_status(), make_status(ArmStatus.NO_SOLUTION, motion_status=1)]
    with pytest.raises(DeviceFault, match="NO_SOLUTION"):
        arm.move_to_pose(np.array([0.2, 0.0, 0.3, 0, 0, 0, 1.0]))


def test_non_blocking_move_does_not_poll(arm, sdk):
    arm.move_to_joints(np.zeros(6), blocking=False)
    assert len(sdk.calls("get_arm_status")) == 1


@pytest.mark.parametrize("mode", [ControlMode.CARTESIAN_SERVO, ControlMode.JOINT_SERVO])
def test_move_to_rejected_in_servo_mode(arm, mode):
    arm.set_control_mode(mode)
    with pytest.raises(DeviceFault, match="idle"):
        arm.move_to_joints(np.zeros(6))
    with pytest.raises(DeviceFault, match="idle"):
        arm.move_to_pose(np.array([0, 0, 0, 0, 0, 0, 1.0]))


def test_disconnect_never_cuts_power(sdk):
    arm = AgxArm(firmware_profile="v189")
    arm.connect()
    arm.disconnect()
    assert sdk.log[-1] == ("disconnect", "v189")
    assert not POWER_CUTTING_CALLS & {entry[0] for entry in sdk.log}
    assert not arm.is_connected


def test_health_notes_status_errors_once_and_recover_clears(arm, sdk):
    sdk.status_sequence = [make_status(ArmStatus.EMERGENCY_STOP, err_code=(1 << 0) | (1 << 9))]
    health = arm.health()
    assert not health.ok
    assert health.errors == [
        "arm_status=EMERGENCY_STOP",
        "joint_2_angle_limit",
        "communication_status_joint_1",
    ]
    assert health.robot_mode == "CAN_CTRL"
    assert arm.health().errors == health.errors
    arm.recover()
    assert sdk.calls("clear_joint_error") == [("clear_joint_error",)]
    sdk.status_sequence = [make_status()]
    assert arm.health().ok


def test_preflight_reports_status_follower_and_profile(arm, sdk):
    results = {result.name: result for result in arm.preflight()}
    assert set(results) == {"mode:piper_status", "mode:follower", "mode:firmware_profile"}
    assert all(result.ok and result.layer == 5 for result in results.values())
    assert "v189" in results["mode:firmware_profile"].message

    sdk.status_sequence = [make_status(ArmStatus.EMERGENCY_STOP)]
    sdk.feedback = False
    results = {result.name: result for result in arm.preflight()}
    assert not results["mode:piper_status"].ok
    assert "release e-stop" in results["mode:piper_status"].hint
    assert not results["mode:follower"].ok
    assert "follower (slave) mode" in results["mode:follower"].hint
    assert results["mode:firmware_profile"].ok


def test_preflight_reports_probed_firmware(sdk):
    arm = AgxArm()
    arm.connect()
    message = {r.name: r for r in arm.preflight()}["mode:firmware_profile"].message
    assert "v188" in message and "S-V1.8-8" in message


# ---------------------------------------------------------------- gripper


@pytest.fixture
def robot(sdk):
    robot = Robot.from_config(PiperConfig(firmware_profile="v189"))
    yield robot
    robot.disconnect()


def test_gripper_borrows_arm_session(robot, sdk):
    gripper = robot.gripper
    assert isinstance(robot.arm, AgxArm) and robot.arm.with_gripper
    assert isinstance(gripper, AgxGripper) and gripper.owner is robot.arm
    gripper.connect()
    assert robot.arm.is_connected and gripper.is_connected
    robot.connect()
    assert len(sdk.calls("connect")) == 1
    assert sdk.calls("init_effector") == [("init_effector", "agx_gripper")]


def test_gripper_move_and_width_mapping(robot, sdk):
    robot.connect()
    gripper = robot.gripper
    gripper.move(0.5)
    gripper.open()
    gripper.close()
    gripper.send_action({"gripper": np.array([1.7])})
    assert sdk.calls("move_gripper_m") == [
        ("move_gripper_m", pytest.approx(0.035), 1.0),
        ("move_gripper_m", pytest.approx(0.07), 1.0),
        ("move_gripper_m", 0.0, 1.0),
        ("move_gripper_m", pytest.approx(0.07), 1.0),
    ]
    assert gripper.width == pytest.approx(0.5)
    sdk.gripper_status.value = 0.09
    assert gripper.width == 1.0
    assert robot.get_observation()["gripper"]["gripper_pos"].shape == (1,)


def test_gripper_without_feedback_hints_stroke_config(robot, sdk):
    robot.connect()
    sdk.gripper_status = None
    with pytest.raises(DeviceFault, match="no feedback") as info:
        robot.gripper.width
    assert info.value.recoverable is True
    assert "teach pendant" in info.value.hint and "0.07 or 0.1 m" in info.value.hint


def test_gripper_in_angle_mode_is_a_fault(robot, sdk):
    robot.connect()
    sdk.gripper_status = SimpleNamespace(value=30.0, force=1.0, mode="angle")
    with pytest.raises(DeviceFault, match="angle"):
        robot.gripper.width


def test_gripper_from_config_requires_shared_session():
    with pytest.raises(ValueError, match="shares_arm_session: true"):
        AgxGripper.from_config(PiperConfig(), owner=None)


def test_gripper_on_arm_without_effector_faults(sdk):
    arm = AgxArm(firmware_profile="v189", with_gripper=False)
    gripper = AgxGripper(owner=arm)
    with pytest.raises(DeviceFault) as info:
        gripper.connect()
    assert "shares_arm_session: true" in info.value.hint


def test_gripper_own_open_is_misuse():
    with pytest.raises(RuntimeError, match="owner"):
        AgxGripper()._open()


# ---------------------------------------------------------------- fault gating / feedback


def test_servo_joints_refused_while_arm_status_faulted(arm, sdk):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    sdk.log.clear()
    sdk.status_sequence = [make_status(ArmStatus.EMERGENCY_STOP)]
    with pytest.raises(DeviceFault, match="refused") as info:
        arm.servo_joints(np.zeros(6))
    assert info.value.recoverable is True
    assert "recover()" in info.value.hint
    assert sdk.calls("move_j") == []
    assert not arm.health().ok


def test_servo_stays_refused_until_recover_after_fault_clears(arm, sdk):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    sdk.status_sequence = [make_status(err_code=1 << 0)]
    pose = np.array([0.2, 0.0, 0.3, 0, 0, 0, 1.0])
    with pytest.raises(DeviceFault):
        arm.servo_cartesian(pose)
    sdk.status_sequence = [make_status()]
    with pytest.raises(DeviceFault):
        arm.servo_cartesian(pose)
    assert sdk.calls("move_p") == []
    arm.recover()
    arm.servo_cartesian(pose)
    assert len(sdk.calls("move_p")) == 1


def test_servo_refused_on_stale_feedback(arm, sdk, monkeypatch):
    monkeypatch.setattr(AgxArm, "FEEDBACK_STALE_S", 0.0)
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    sdk.log.clear()
    sdk.advance_joint_stamp = False
    with pytest.raises(DeviceFault, match="stale") as info:
        arm.servo_joints(np.zeros(6))
    assert info.value.recoverable is True
    assert sdk.calls("move_j") == []


def test_connect_waits_for_every_feedback_frame(sdk):
    sdk.missing_frames = {"joint_56"}
    arm = AgxArm(firmware_profile="v189")
    with pytest.raises(DeviceFault, match="no joint feedback"):
        arm.connect()
    assert not arm.is_connected


def test_stop_refuses_partial_feedback(arm, sdk):
    sdk.log.clear()
    sdk.missing_frames = {"joint_34"}
    with pytest.raises(DeviceFault, match="joint_34") as info:
        arm.set_control_mode(ControlMode.JOINT_SERVO)
    assert info.value.recoverable is True
    assert sdk.calls("move_j") == []


def test_stop_refuses_stale_feedback(arm, sdk, monkeypatch):
    arm.stop()
    monkeypatch.setattr(AgxArm, "FEEDBACK_STALE_S", 0.0)
    sdk.advance_joint_stamp = False
    sdk.log.clear()
    with pytest.raises(DeviceFault, match="stale"):
        arm.stop()
    assert sdk.calls("move_j") == []


@pytest.mark.parametrize("backend_cls", [AgxArm, AgxGripper])
def test_missing_sdk_hint_points_to_extra(monkeypatch, backend_cls):
    from robot_infra.devices import SdkMissing

    monkeypatch.setitem(sys.modules, "pyAgxArm", None)
    with pytest.raises(SdkMissing) as info:
        backend_cls.require_sdk()
    assert info.value.hint == "pip install 'rlgarden-robot-infra[pyagxarm]'  (pyAgxArm is installed from GitHub)"

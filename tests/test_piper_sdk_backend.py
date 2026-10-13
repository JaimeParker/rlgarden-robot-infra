"""Hardware-free tests for the piper_sdk Piper arm/gripper using a recording fake SDK."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from robot_infra.devices import Arm, ControlMode, DeviceFault, DeviceNotFound, Gripper, SdkMissing
from robot_infra.devices.backends.arms import piper_sdk as arm_module
from robot_infra.devices.backends.arms._rotations import rpy_to_quat_xyzw
from robot_infra.devices.backends.arms.piper_sdk import PiperSdkArm
from robot_infra.devices.backends.grippers import piper_sdk as gripper_module  # noqa: F401
from robot_infra.devices.backends.grippers.piper_sdk import PiperSdkGripper
from robot_infra.devices.gripper import GripperMode
from robot_infra.devices.robot import GripperConfig
from robot_infra.discovery.enumerate import CanInfo

MILLI_DEG_PER_RAD = 1000 * 180 / np.pi


class FakePiper:
    """Records calls and serves configurable messages (wire units, like the real SDK)."""

    def __init__(self, can_name="can0", **kwargs):
        self.can_name = can_name
        self.ctor_kwargs = kwargs
        self.calls: list[tuple] = []
        self.enable_after = 0
        self.joints = [10_000, -20_000, 30_000, 0, 5_000, 1_000]
        self.joint_stamp = 1.0
        self.advance_joint_stamp = True  # each read sees a new joint frame, like a live arm
        self.status_stamp = 0.0
        self.advance_status_stamp = True
        self.end_pose = [100_000, -50_000, 250_000, 0, 90_000, 45_000]
        self.speeds = [1000, -2000, 0, 0, 0, 500]
        self.arm_status = 0
        self.ctrl_mode = 1
        self.motion_status_sequence = [0]
        self.err_status = SimpleNamespace()
        self.gripper_angle = 0
        self.gripper_stamp = 0.0

    def _log(self, name, *args):
        self.calls.append((name, *args))

    def names(self):
        return [call[0] for call in self.calls]

    def ConnectPort(self, *args, **kwargs):
        self._log("ConnectPort")

    def DisconnectPort(self, *args, **kwargs):
        self._log("DisconnectPort")
        self.disconnect_kwargs = kwargs

    def EnablePiper(self):
        self._log("EnablePiper")
        return self.names().count("EnablePiper") > self.enable_after

    def DisableArm(self, *args, **kwargs):
        self._log("DisableArm")

    def ResetPiper(self):
        self._log("ResetPiper")

    def MotionCtrl_2(self, *args):
        self._log("MotionCtrl_2", *args)

    def JointCtrl(self, *args):
        self._log("JointCtrl", *args)

    def EndPoseCtrl(self, *args):
        self._log("EndPoseCtrl", *args)

    def GripperCtrl(self, *args):
        self._log("GripperCtrl", *args)

    def GetArmJointMsgs(self):
        if self.advance_joint_stamp:
            self.joint_stamp += 0.005
        names = [f"joint_{i}" for i in range(1, 7)]
        return SimpleNamespace(
            time_stamp=self.joint_stamp, joint_state=SimpleNamespace(**dict(zip(names, self.joints)))
        )

    def GetArmEndPoseMsgs(self):
        names = ["X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis"]
        return SimpleNamespace(end_pose=SimpleNamespace(**dict(zip(names, self.end_pose))))

    def GetArmHighSpdInfoMsgs(self):
        return SimpleNamespace(
            **{f"motor_{i + 1}": SimpleNamespace(motor_speed=v) for i, v in enumerate(self.speeds)}
        )

    def GetArmStatus(self):
        sequence = self.motion_status_sequence
        motion = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        if self.advance_status_stamp:
            self.status_stamp += 0.01
        return SimpleNamespace(
            time_stamp=self.status_stamp,
            arm_status=SimpleNamespace(
                ctrl_mode=self.ctrl_mode,
                arm_status=self.arm_status,
                mode_feed=1,
                motion_status=motion,
                err_status=self.err_status,
            )
        )

    def GetArmGripperMsgs(self):
        return SimpleNamespace(
            time_stamp=self.gripper_stamp,
            gripper_state=SimpleNamespace(grippers_angle=self.gripper_angle),
        )


@pytest.fixture
def piper_env(monkeypatch):
    created: list[FakePiper] = []

    def factory(*args, **kwargs):
        piper = FakePiper(*args, **kwargs)
        created.append(piper)
        return piper

    monkeypatch.setitem(sys.modules, "piper_sdk", types.SimpleNamespace(C_PiperInterface_V2=factory))
    monkeypatch.setattr(
        arm_module, "list_can_interfaces", lambda: {"can0": CanInfo("can0", "up", 1_000_000)}
    )
    monkeypatch.setattr(PiperSdkArm, "ENABLE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(PiperSdkArm, "FIRST_FEEDBACK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(PiperSdkArm, "MOTION_TIMEOUT_S", 0.1)
    monkeypatch.setattr(PiperSdkArm, "MOTION_SETTLE_S", 0.0)
    monkeypatch.setattr(PiperSdkArm, "POLL_PERIOD_S", 0.002)
    monkeypatch.setattr(PiperSdkGripper, "STUCK_GRACE_S", 0.0)
    return SimpleNamespace(created=created, monkeypatch=monkeypatch)


@pytest.fixture
def arm(piper_env):
    arm = PiperSdkArm()
    arm.connect()
    yield arm
    arm.disconnect()


def sdk(arm) -> FakePiper:
    return arm._handle


# -- registry / config ---------------------------------------------------------------


def test_registry_resolves_piper_sdk():
    assert Arm.backend("piper_sdk") is PiperSdkArm
    assert Gripper.backend("piper_sdk") is PiperSdkGripper


def test_observation_features_exclude_tcp_vel():
    features = PiperSdkArm().observation_features
    assert set(features) == {"tcp_pose", "joint_pos", "joint_vel"}
    assert features["joint_pos"]["shape"] == (6,)


def test_from_config_uses_piper_config():
    from robot_infra.devices.robots.piper import PiperConfig

    config = PiperConfig(can_interface="can3", bitrate=500_000)
    arm = PiperSdkArm.from_config(config)
    assert (arm._can_interface, arm._bitrate) == ("can3", 500_000)


def test_sdk_missing_is_typed(monkeypatch):
    monkeypatch.setitem(sys.modules, "piper_sdk", None)
    monkeypatch.setattr(
        arm_module, "list_can_interfaces", lambda: {"can0": CanInfo("can0", "up", 1_000_000)}
    )
    with pytest.raises(SdkMissing):
        PiperSdkArm().connect()


# -- CAN precheck --------------------------------------------------------------------


def test_can_missing(piper_env):
    piper_env.monkeypatch.setattr(arm_module, "list_can_interfaces", lambda: {})
    with pytest.raises(DeviceNotFound) as info:
        PiperSdkArm().connect()
    assert "bash can_activate.sh can0 1000000" in info.value.hint
    assert "sudo ip link set can0 up type can bitrate 1000000" in info.value.hint
    assert piper_env.created == []


def test_can_down(piper_env):
    piper_env.monkeypatch.setattr(
        arm_module, "list_can_interfaces", lambda: {"can0": CanInfo("can0", "down", None)}
    )
    with pytest.raises(DeviceFault) as info:
        PiperSdkArm().connect()
    assert "bash can_activate.sh can0 1000000" in info.value.hint


def test_can_bitrate_mismatch(piper_env):
    piper_env.monkeypatch.setattr(
        arm_module, "list_can_interfaces", lambda: {"can0": CanInfo("can0", "up", 500_000)}
    )
    with pytest.raises(DeviceFault, match="500000.*1000000"):
        PiperSdkArm().connect()


# -- connect sequence ----------------------------------------------------------------


def test_connect_sequence(arm):
    assert sdk(arm).names()[:2] == ["ConnectPort", "EnablePiper"]
    assert sdk(arm).can_name == "can0"
    assert arm.is_connected


def test_enable_loop_retries_until_enabled(piper_env):
    piper_env.monkeypatch.setattr(
        arm_module,
        "list_can_interfaces",
        lambda: {"can0": CanInfo("can0", "up", 1_000_000)},
    )
    original = FakePiper.__init__

    def slow_enable(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.enable_after = 3

    piper_env.monkeypatch.setattr(FakePiper, "__init__", slow_enable)
    arm = PiperSdkArm()
    arm.connect()
    assert sdk(arm).names().count("EnablePiper") == 4
    arm.disconnect()


def test_enable_timeout_hint_and_cleanup(piper_env):
    original = FakePiper.__init__

    def never_enable(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.enable_after = 10**9

    piper_env.monkeypatch.setattr(FakePiper, "__init__", never_enable)
    arm = PiperSdkArm()
    with pytest.raises(DeviceFault) as info:
        arm.connect()
    assert info.value.recoverable
    assert "did not enable within" in info.value.message
    assert "emergency stop is released" in info.value.hint
    assert "'can0' is up at 1000000 bit/s" in info.value.hint
    created = piper_env.created[0]
    assert "DisconnectPort" in created.names()
    assert "DisableArm" not in created.names()
    assert not arm.is_connected


def test_all_zero_feedback_hint(piper_env):
    original = FakePiper.__init__

    def silent(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.joints = [0] * 6
        self.joint_stamp = 0.0
        self.advance_joint_stamp = False

    piper_env.monkeypatch.setattr(FakePiper, "__init__", silent)
    with pytest.raises(DeviceFault) as info:
        PiperSdkArm().connect()
    assert "master/teach mode" in info.value.hint
    assert "follower (slave) mode" in info.value.hint
    assert "gripper stroke" in info.value.hint
    assert info.value.recoverable


def test_zero_joints_with_timestamp_count_as_feedback(piper_env):
    original = FakePiper.__init__

    def at_zero(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.joints = [0] * 6
        self.joint_stamp = 5.0

    piper_env.monkeypatch.setattr(FakePiper, "__init__", at_zero)
    arm = PiperSdkArm()
    arm.connect()
    assert arm.is_connected
    arm.disconnect()


# -- state ---------------------------------------------------------------------------


def test_read_state_unit_conversions(arm):
    piper = sdk(arm)
    piper.joints = [57296, -28648, 0, 1000, 0, 180_000]
    piper.end_pose = [123_456, -7_000, 300_000, 0, 0, 90_000]
    piper.speeds = [1500, -250, 0, 0, 0, 1000]
    state = arm.read_state()
    np.testing.assert_allclose(
        state["joint_pos"], np.array(piper.joints) / MILLI_DEG_PER_RAD, atol=1e-12
    )
    assert state["joint_pos"][5] == pytest.approx(np.pi)
    np.testing.assert_allclose(state["tcp_pose"][:3], [0.123456, -0.007, 0.3], atol=1e-12)
    np.testing.assert_allclose(state["joint_vel"], [1.5, -0.25, 0, 0, 0, 1.0], atol=1e-12)


def test_read_state_quaternion_from_rpy(arm):
    sdk(arm).end_pose = [0, 0, 0, 30_000, -45_000, 90_000]
    rpy = np.deg2rad([30.0, -45.0, 90.0])
    np.testing.assert_allclose(arm.read_state()["tcp_pose"][3:], rpy_to_quat_xyzw(rpy), atol=1e-9)


def test_get_observation_marks_feedback(arm):
    assert arm.health().first_feedback_received is False
    observation = arm.get_observation()
    assert set(observation) == {"tcp_pose", "joint_pos", "joint_vel"}
    assert arm.health().first_feedback_received


def test_stale_feedback_faults(arm, monkeypatch):
    monkeypatch.setattr(PiperSdkArm, "FEEDBACK_STALE_S", 0.0)
    sdk(arm).advance_joint_stamp = False
    arm.read_state()
    with pytest.raises(DeviceFault) as info:
        arm.read_state()
    assert "stale" in info.value.message
    assert "master/teach mode" in info.value.hint
    assert info.value.recoverable
    sdk(arm).joint_stamp = 2.0
    arm.read_state()


# -- commands ------------------------------------------------------------------------


def test_servo_joints_exact_integers_and_mode_frame_first(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    sdk(arm).calls.clear()
    q = np.array([0.1, -0.2, 0.3, 0.0, 1.0, np.pi])
    arm.servo_joints(q)
    expected = [int(np.rint(v * MILLI_DEG_PER_RAD)) for v in q]
    assert sdk(arm).calls == [("MotionCtrl_2", 0x01, 0x01, 30), ("JointCtrl", *expected)]
    assert expected[0] == 5730 and expected[5] == 180_000
    assert all(type(v) is int for v in sdk(arm).calls[1][1:])


def test_servo_cartesian_exact_integers_and_mode_frame_first(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    sdk(arm).calls.clear()
    rpy = np.deg2rad([10.0, 20.0, 30.0])
    pose = np.concatenate([[0.1234567, -0.05, 0.25], rpy_to_quat_xyzw(rpy)])
    arm.servo_cartesian(pose)
    assert sdk(arm).calls[0] == ("MotionCtrl_2", 0x01, 0x00, 30)
    assert sdk(arm).calls[1] == ("EndPoseCtrl", 123_457, -50_000, 250_000, 10_000, 20_000, 30_000)
    assert all(type(v) is int for v in sdk(arm).calls[1][1:])


def test_send_action_goes_through_servo(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    sdk(arm).calls.clear()
    arm.send_action({"joint_pos": np.zeros(6)})
    assert sdk(arm).names() == ["MotionCtrl_2", "JointCtrl"]


def test_every_command_frame_preceded_by_motion_ctrl_2(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    for _ in range(3):
        arm.servo_joints(np.zeros(6))
    arm.set_control_mode(ControlMode.IDLE)
    arm.move_to_joints(np.zeros(6))
    names = sdk(arm).names()
    for index, name in enumerate(names):
        if name in ("JointCtrl", "EndPoseCtrl"):
            assert names[index - 1] == "MotionCtrl_2"


def test_servo_requires_matching_mode(arm):
    with pytest.raises(DeviceFault):
        arm.servo_joints(np.zeros(6))
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    with pytest.raises(DeviceFault):
        arm.servo_cartesian(np.array([0, 0, 0, 0, 0, 0, 1.0]))


def test_move_to_rejected_in_servo_mode(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    sdk(arm).calls.clear()
    with pytest.raises(DeviceFault) as info:
        arm.move_to_joints(np.zeros(6))
    assert "ControlMode.IDLE" in info.value.hint
    with pytest.raises(DeviceFault):
        arm.move_to_pose(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    assert sdk(arm).calls == []


def test_move_to_joints_blocks_until_motion_status_zero(arm):
    sdk(arm).motion_status_sequence = [1, 1, 1, 0]
    arm.move_to_joints(np.zeros(6), blocking=True)
    assert sdk(arm).motion_status_sequence == [0]
    assert sdk(arm).names().count("JointCtrl") >= 3


def test_move_to_pose_uses_move_p(arm):
    sdk(arm).calls.clear()
    arm.move_to_pose(np.array([0.1, 0.0, 0.2, 0, 0, 0, 1.0]), blocking=False)
    assert sdk(arm).calls[0] == ("MotionCtrl_2", 0x01, 0x00, 30)
    assert sdk(arm).calls[1] == ("EndPoseCtrl", 100_000, 0, 200_000, 0, 0, 0)


def test_non_blocking_move_does_not_poll(arm):
    sdk(arm).motion_status_sequence = [1]
    arm.move_to_joints(np.zeros(6), blocking=False)
    assert sdk(arm).names().count("JointCtrl") == 1


def test_blocking_move_timeout_faults(arm):
    sdk(arm).motion_status_sequence = [1]
    with pytest.raises(DeviceFault) as info:
        arm.move_to_joints(np.zeros(6), blocking=True)
    assert info.value.recoverable
    assert "did not reach" in info.value.message


def test_wrong_shapes_rejected(arm):
    with pytest.raises(ValueError):
        arm.move_to_joints(np.zeros(7))
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    with pytest.raises(ValueError):
        arm.servo_cartesian(np.zeros(6))


def test_stop_resends_measured_joints(arm):
    sdk(arm).calls.clear()
    arm.stop()
    assert sdk(arm).calls == [
        ("MotionCtrl_2", 0x01, 0x01, 30),
        ("JointCtrl", 10_000, -20_000, 30_000, 0, 5_000, 1_000),
    ]


def test_move_speed_percent_is_configurable(piper_env):
    arm = PiperSdkArm(move_speed_percent=80)
    arm.connect()
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.servo_joints(np.zeros(6))
    assert ("MotionCtrl_2", 0x01, 0x01, 80) in sdk(arm).calls
    arm.disconnect()


# -- disconnect ----------------------------------------------------------------------


def test_disconnect_never_disables_or_resets(piper_env):
    arm = PiperSdkArm()
    arm.connect()
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    piper = sdk(arm)
    arm.disconnect()
    assert piper.names()[-1] == "DisconnectPort"
    assert "DisableArm" not in piper.names()
    assert "ResetPiper" not in piper.names()
    assert not arm.is_connected


def test_recover_sends_nothing_and_clears_when_status_normal(arm):
    arm._note_error("x")
    sdk(arm).calls.clear()
    arm.recover()
    assert sdk(arm).calls == []
    assert arm.health().errors == []


# -- health / preflight --------------------------------------------------------------


def test_health_reports_arm_status_once_per_change(arm):
    piper = sdk(arm)
    assert arm.health().ok
    piper.arm_status = 0x07
    for _ in range(3):
        health = arm.health()
    assert health.errors == ["arm_status: collision"]
    assert not health.ok
    piper.arm_status = 0x00
    assert arm.health().errors == []


def test_health_reports_joint_flags(arm):
    sdk(arm).arm_status = 0x04
    sdk(arm).err_status = SimpleNamespace(joint_2_angle_limit=True, communication_status_joint_5=True)
    errors = arm.health().errors
    assert "arm_status: target angle over limit" in errors
    assert "joint 2 over angle limit" in errors
    assert "joint 5 communication error" in errors


def test_preflight_ok_without_gripper(arm):
    results = {r.name: r for r in arm.preflight()}
    assert set(results) == {"mode:piper_status", "mode:follower", "mode:gripper_stroke"}
    assert all(r.layer == 5 for r in results.values())
    assert results["mode:piper_status"].ok
    assert results["mode:follower"].ok
    assert results["mode:gripper_stroke"].ok and results["mode:gripper_stroke"].skipped


def test_preflight_flags_fault_status(arm):
    sdk(arm).arm_status = 0x01
    results = {r.name: r for r in arm.preflight()}
    assert not results["mode:piper_status"].ok
    assert "emergency stop" in results["mode:piper_status"].message


def test_preflight_gripper_feedback_flowing(arm):
    sdk(arm).gripper_stamp = 3.0
    result = {r.name: r for r in arm.preflight()}["mode:gripper_stroke"]
    assert result.ok and not result.skipped


# -- gripper -------------------------------------------------------------------------


@pytest.fixture
def gripper(arm):
    gripper = PiperSdkGripper(owner=arm)
    gripper.connect()
    yield gripper
    gripper.disconnect()


def test_gripper_requires_owner_in_from_config():
    from robot_infra.devices.robots.piper import PiperConfig

    config = PiperConfig(
        gripper=GripperConfig(backend="piper_sdk", shares_arm_session=False)
    )
    with pytest.raises(ValueError, match="shares_arm_session: true"):
        PiperSdkGripper.from_config(config, owner=None)


def test_gripper_from_config_with_owner(arm):
    from robot_infra.devices.robots.piper import PiperConfig

    config = PiperConfig(
        gripper=GripperConfig(
            backend="piper_sdk", max_width_m=0.1, mode="binary", shares_arm_session=True
        )
    )
    gripper = PiperSdkGripper.from_config(config, owner=arm)
    assert gripper.max_width_m == 0.1
    assert gripper.mode is GripperMode.BINARY


def test_gripper_borrows_owner_session(arm):
    gripper = PiperSdkGripper(owner=arm)
    assert not gripper.is_connected
    gripper.connect()
    assert gripper.is_connected
    assert sdk(arm).names().count("ConnectPort") == 1
    gripper.disconnect()
    assert arm.is_connected
    assert "DisconnectPort" not in sdk(arm).names()


def test_gripper_connects_owner_on_demand(piper_env):
    arm = PiperSdkArm()
    gripper = PiperSdkGripper(owner=arm)
    gripper.connect()
    assert arm.is_connected
    arm.disconnect()
    assert not gripper.is_connected


def test_gripper_open_close_commands(gripper, arm):
    gripper.open()
    gripper.close()
    gripper.move(0.5)
    calls = [c for c in sdk(arm).calls if c[0] == "GripperCtrl"]
    assert calls == [
        ("GripperCtrl", 70_000, 1000, 0x01, 0),
        ("GripperCtrl", 0, 1000, 0x01, 0),
        ("GripperCtrl", 35_000, 1000, 0x01, 0),
    ]
    assert all(type(v) is int for c in calls for v in c[1:])


def test_gripper_move_is_clipped_and_uses_custom_effort(arm):
    gripper = PiperSdkGripper(max_width_m=0.1, effort=2500, owner=arm)
    gripper.connect()
    gripper.move(1.7)
    gripper.move(-1)
    calls = [c for c in sdk(arm).calls if c[0] == "GripperCtrl"]
    assert calls == [("GripperCtrl", 100_000, 2500, 0x01, 0), ("GripperCtrl", 0, 2500, 0x01, 0)]


def test_gripper_width_readback(gripper, arm):
    sdk(arm).gripper_angle = 35_000
    assert gripper.width == pytest.approx(0.5)
    sdk(arm).gripper_angle = 90_000
    assert gripper.width == 1.0
    assert gripper.get_observation()["gripper_pos"].shape == (1,)


def test_gripper_stuck_at_zero_after_move(gripper, arm):
    gripper.move(1.0)
    sdk(arm).gripper_angle = 0
    with pytest.raises(DeviceFault) as info:
        gripper.width
    assert "configure the gripper stroke" in info.value.hint
    assert info.value.recoverable


def test_gripper_zero_without_nonzero_command_is_fine(gripper, arm):
    assert gripper.width == 0.0
    gripper.close()
    assert gripper.width == 0.0


def test_gripper_move_before_connect_is_error():
    from robot_infra.devices import DeviceNotConnected

    with pytest.raises(DeviceNotConnected):
        PiperSdkGripper(owner=PiperSdkArm()).move(0.5)


def test_gripper_own_open_is_guarded():
    with pytest.raises(RuntimeError):
        PiperSdkGripper()._open()


# -- fault gating / feedback completeness / motion settle (review H2-H4, M6b, M8) -----


def test_servo_joints_refused_while_arm_status_faulted(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    sdk(arm).calls.clear()
    sdk(arm).arm_status = 0x07
    with pytest.raises(DeviceFault, match="refused") as info:
        arm.servo_joints(np.zeros(6))
    assert info.value.recoverable
    assert sdk(arm).calls == []
    assert "arm_status: collision" in arm.health().errors


def test_servo_cartesian_refused_on_joint_flag(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    sdk(arm).calls.clear()
    sdk(arm).err_status = SimpleNamespace(communication_status_joint_3=True)
    with pytest.raises(DeviceFault, match="refused"):
        arm.servo_cartesian(np.array([0.1, 0.0, 0.2, 0, 0, 0, 1.0]))
    assert sdk(arm).calls == []


def test_servo_refused_on_stale_feedback(arm, monkeypatch):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    monkeypatch.setattr(PiperSdkArm, "FEEDBACK_STALE_S", 0.0)
    sdk(arm).advance_joint_stamp = False
    sdk(arm).calls.clear()
    with pytest.raises(DeviceFault, match="stale"):
        arm.servo_joints(np.zeros(6))
    assert sdk(arm).calls == []


def test_blocking_move_ignores_stale_reached_status(arm):
    sdk(arm).advance_status_stamp = False
    sdk(arm).motion_status_sequence = [0]
    with pytest.raises(DeviceFault, match="did not reach"):
        arm.move_to_joints(np.zeros(6), blocking=True)


def test_blocking_move_waits_for_settle(arm, monkeypatch):
    monkeypatch.setattr(PiperSdkArm, "MOTION_SETTLE_S", 0.05)
    sdk(arm).motion_status_sequence = [0]
    start = arm_module.time.monotonic()
    arm.move_to_joints(np.zeros(6), blocking=True)
    assert arm_module.time.monotonic() - start >= 0.05


def test_connect_needs_joint_stamp_to_advance_for_every_frame(piper_env):
    original = FakePiper.__init__

    def cached_frames_only(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.advance_joint_stamp = False  # singleton cache from an earlier session, no new frames

    piper_env.monkeypatch.setattr(FakePiper, "__init__", cached_frames_only)
    arm = PiperSdkArm()
    with pytest.raises(DeviceFault, match="no joint feedback"):
        arm.connect()
    assert not arm.is_connected
    assert "DisconnectPort" in piper_env.created[0].names()


def test_stop_refuses_stale_feedback(arm, monkeypatch):
    arm.stop()
    monkeypatch.setattr(PiperSdkArm, "FEEDBACK_STALE_S", 0.0)
    sdk(arm).advance_joint_stamp = False
    sdk(arm).calls.clear()
    with pytest.raises(DeviceFault, match="stale"):
        arm.stop()
    assert sdk(arm).calls == []


def test_stop_refuses_without_joint_feedback(arm):
    piper = sdk(arm)
    piper.joints = [0] * 6
    piper.joint_stamp = 0.0
    piper.advance_joint_stamp = False
    piper.calls.clear()
    with pytest.raises(DeviceFault):
        arm.set_control_mode(ControlMode.JOINT_SERVO)
    assert piper.calls == []


def test_disconnect_passes_join_timeout_and_joins_reader(piper_env):
    joined = []
    reader = SimpleNamespace(is_alive=lambda: True, join=lambda timeout=None: joined.append(timeout))
    arm = PiperSdkArm()
    arm.connect()
    piper = sdk(arm)
    piper._C_PiperInterface_V2__can_deal_th = reader
    arm.disconnect()
    assert piper.names()[-1] == "DisconnectPort"
    assert piper.disconnect_kwargs == {"thread_timeout": PiperSdkArm.DISCONNECT_JOIN_TIMEOUT_S}
    assert joined == [PiperSdkArm.DISCONNECT_JOIN_TIMEOUT_S]


def test_feedback_age_only_renewed_by_new_joint_frame(arm):
    piper = sdk(arm)
    piper.advance_joint_stamp = False
    arm.read_state()
    first = arm._last_feedback_time
    arm.read_state()
    assert arm._last_feedback_time == first
    piper.joint_stamp += 1.0
    arm.read_state()
    assert arm._last_feedback_time > first


def test_recover_keeps_errors_while_still_faulted(arm):
    piper = sdk(arm)
    piper.arm_status = 0x01
    assert arm.health().errors == ["arm_status: emergency stop"]
    arm.recover()
    health = arm.health()
    assert health.errors == ["arm_status: emergency stop"]
    assert not health.ok
    piper.arm_status = 0x00
    arm.recover()
    assert arm.health().ok


def test_joint_flag_change_with_same_arm_status_is_reported(arm):
    piper = sdk(arm)
    piper.arm_status = 0x04
    assert arm.health().errors == ["arm_status: target angle over limit"]
    piper.err_status = SimpleNamespace(joint_4_angle_limit=True)
    assert "joint 4 over angle limit" in arm.health().errors

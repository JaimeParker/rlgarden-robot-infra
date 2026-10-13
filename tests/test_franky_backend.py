"""Hardware-free tests for the franky Franka arm/gripper backends against a fake ``franky`` module."""
from __future__ import annotations

import enum
import sys
import types

import numpy as np
import pytest

import robot_infra.devices.backends.arms.franky as franky_arm_module
from robot_infra.devices import (
    Arm,
    ControlMode,
    DeviceFault,
    DeviceNotFound,
    Gripper,
    GripperMode,
    SdkMissing,
    VersionMismatch,
)
from robot_infra.devices.backends.arms.franky import FrankyArm
from robot_infra.devices.backends.grippers.franky import FrankyGripper
from robot_infra.devices.robots.franka import FrankaConfig

QUAT_Z90_XYZW = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
JACOBIAN = np.arange(42, dtype=np.float64).reshape(6, 7) / 100.0


def make_fake_franky(log: list[str]) -> types.ModuleType:
    franky = types.ModuleType("franky")

    class FrankyException(Exception):
        pass

    for exc_name in (
        "CommandException",
        "ControlException",
        "IncompatibleVersionException",
        "NetworkException",
        "RealtimeException",
    ):
        setattr(franky, exc_name, type(exc_name, (FrankyException,), {}))
    franky.Exception = FrankyException
    franky.GripperException = type("GripperException", (RuntimeError,), {})
    franky.RealtimeConfig = enum.Enum("RealtimeConfig", "Enforce Ignore")
    franky.RobotMode = enum.Enum(
        "RobotMode", "Other Idle Move Guiding Reflex UserStopped AutomaticErrorRecovery"
    )
    franky.Frame = enum.Enum("Frame", "Joint1 EndEffector")

    class Affine:
        def __init__(self, translation=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
            self.translation = np.asarray(translation, dtype=np.float64)
            self.quaternion = np.asarray(quaternion, dtype=np.float64)

    class CartesianMotion:
        def __init__(self, target):
            self.target = target

    class JointMotion:
        def __init__(self, target):
            self.target = np.asarray(target)

    class RobotState:
        def __init__(self):
            self.O_T_EE = Affine([0.3, -0.1, 0.5], QUAT_Z90_XYZW)
            self.q = np.linspace(0.0, 0.6, 7)
            self.dq = np.full(7, 0.1)
            self.O_F_ext_hat_K = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
            self.robot_mode = franky.RobotMode.Idle
            self.current_errors = "Errors()"

    class Model:
        def zero_jacobian(self, frame, state):
            assert frame is franky.Frame.EndEffector
            return JACOBIAN

    class Robot:
        connect_error: Exception | None = None

        def __init__(self, fci_hostname, **kwargs):
            if Robot.connect_error is not None:
                raise Robot.connect_error
            log.append("Robot")
            self.fci_hostname = fci_hostname
            self.kwargs = kwargs
            self.state = RobotState()
            self.model = Model()
            self.is_in_control = False
            self.has_errors = False
            self.pending_motion_error: Exception | None = None
            self.moves: list[tuple[object, bool]] = []

        def move(self, motion, asynchronous=False):
            log.append(f"move:{type(motion).__name__}")
            self.moves.append((motion, asynchronous))
            self.is_in_control = isinstance(motion, _Tracker) or asynchronous

        def join_motion(self, timeout=None):
            log.append("join_motion")
            self.is_in_control = False
            error, self.pending_motion_error = self.pending_motion_error, None
            if error is not None:
                raise error
            return True

        def stop(self):
            log.append("robot.stop")
            if self.is_in_control:
                self.pending_motion_error = franky.ControlException("libfranka: Move command preempted!")
            self.is_in_control = False

        def recover_from_errors(self):
            log.append("recover_from_errors")
            self.state.robot_mode = franky.RobotMode.Idle
            return True

    class _Tracker:
        def __init__(self, robot):
            log.append(type(self).__name__)
            self.robot = robot
            self.targets: list[object] = []
            robot.move(self, asynchronous=True)

        def set_target(self, target):
            log.append(f"{type(self).__name__}.set_target")
            self.targets.append(target)

        def stop(self):
            log.append(f"{type(self).__name__}.stop")
            self.robot.join_motion()

    class Gripper:
        connect_error: Exception | None = None
        grasp_behavior = "ok"

        def __init__(self, fci_hostname):
            if Gripper.connect_error is not None:
                raise Gripper.connect_error
            self.fci_hostname = fci_hostname
            self.width = 0.04
            self.is_grasped = False
            self.calls: list[tuple] = []

        def open(self, speed):
            self.calls.append(("open", speed))
            return True

        def move(self, width, speed):
            self.calls.append(("move", width, speed))
            return True

        def grasp(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
            self.calls.append(("grasp", width, speed, force, epsilon_inner, epsilon_outer))
            if Gripper.grasp_behavior == "gripper_exception":
                raise franky.GripperException("nothing grasped")
            if Gripper.grasp_behavior == "command_exception":
                raise franky.CommandException("grasp command rejected")
            return Gripper.grasp_behavior == "ok"

    franky.Affine = Affine
    franky.CartesianMotion = CartesianMotion
    franky.JointMotion = JointMotion
    franky.Robot = Robot
    franky.Gripper = Gripper
    franky.CartesianImpedanceTracker = type("CartesianImpedanceTracker", (_Tracker,), {})
    franky.JointImpedanceTracker = type("JointImpedanceTracker", (_Tracker,), {})
    return franky


@pytest.fixture
def log():
    return []


@pytest.fixture
def franky(monkeypatch, log):
    fake = make_fake_franky(log)
    monkeypatch.setitem(sys.modules, "franky", fake)
    return fake


@pytest.fixture
def arm(franky):
    arm = FrankyArm(robot_ip="10.0.0.2", realtime="ignore")
    arm.connect()
    yield arm
    arm.disconnect()


@pytest.fixture
def gripper(franky):
    gripper = FrankyGripper(robot_ip="10.0.0.2", max_width_m=0.08, speed=0.1, force=30.0)
    gripper.connect()
    yield gripper
    gripper.disconnect()


def test_registry_resolves_franky():
    assert Arm.backend("franky") is FrankyArm
    assert Gripper.backend("franky") is FrankyGripper


def test_from_config_wiring():
    config = FrankaConfig(robot_ip="192.168.1.7", relative_dynamics_factor=0.1, realtime="enforce")
    arm = FrankyArm.from_config(config)
    assert (arm.name, arm.robot_ip, arm.relative_dynamics_factor, arm.realtime) == (
        "arm",
        "192.168.1.7",
        0.1,
        "enforce",
    )
    assert arm.home_joints == config.home_joints
    assert arm.num_joints == 7
    gripper = FrankyGripper.from_config(config)
    assert (gripper.robot_ip, gripper.max_width_m, gripper.mode) == (
        "192.168.1.7",
        0.08,
        GripperMode.BINARY,
    )


def test_missing_sdk_hints_distribution_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "franky", None)
    with pytest.raises(SdkMissing) as info:
        FrankyArm(robot_ip="10.0.0.2").connect()
    assert "franky-control" in info.value.hint


def test_connect_passes_settings_and_reads_first_feedback(arm, franky):
    robot = arm._handle.robot
    assert robot.fci_hostname == "10.0.0.2"
    assert robot.kwargs == {
        "relative_dynamics_factor": 0.2,
        "realtime_config": franky.RealtimeConfig.Ignore,
    }
    health = arm.health()
    assert health.first_feedback_received and health.ok
    assert health.robot_mode == "Idle"


def test_enforce_realtime_config(franky):
    arm = FrankyArm(robot_ip="10.0.0.2", realtime="enforce")
    arm.connect()
    assert arm._handle.robot.kwargs["realtime_config"] is franky.RealtimeConfig.Enforce


def test_incompatible_version_maps_to_version_mismatch(franky):
    franky.Robot.connect_error = franky.IncompatibleVersionException("server version: 5, library version: 9")
    with pytest.raises(VersionMismatch) as info:
        FrankyArm(robot_ip="10.0.0.2").connect()
    assert "server version: 5" in info.value.message
    assert "by-robot-server-version" in info.value.hint


@pytest.mark.parametrize("error_name", ["NetworkException", "RuntimeError"])
def test_unreachable_robot_maps_to_device_not_found(franky, error_name):
    error_cls = getattr(franky, error_name, RuntimeError)
    franky.Robot.connect_error = error_cls("UDP receive: Timeout")
    arm = FrankyArm(robot_ip="10.0.0.2")
    with pytest.raises(DeviceNotFound) as info:
        arm.connect()
    assert "10.0.0.2" in info.value.message
    assert "FCI" in info.value.hint
    assert not arm.is_connected


def test_realtime_exception_gets_realtime_hint(franky):
    franky.Robot.connect_error = franky.RealtimeException("kernel does not have realtime capabilities")
    with pytest.raises(DeviceFault) as info:
        FrankyArm(robot_ip="10.0.0.2", realtime="enforce").connect()
    assert "realtime: ignore" in info.value.hint


def test_read_state_shapes_and_values(arm):
    state = arm.read_state()
    assert {k: v.shape for k, v in state.items()} == {
        "tcp_pose": (7,),
        "tcp_vel": (6,),
        "joint_pos": (7,),
        "joint_vel": (7,),
        "tcp_force": (3,),
        "tcp_torque": (3,),
    }
    np.testing.assert_allclose(state["tcp_pose"][:3], [0.3, -0.1, 0.5])
    np.testing.assert_allclose(state["tcp_pose"][3:], QUAT_Z90_XYZW)  # xyzw, w last
    np.testing.assert_allclose(state["tcp_vel"], JACOBIAN @ np.full(7, 0.1))
    np.testing.assert_allclose(state["tcp_force"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state["tcp_torque"], [0.4, 0.5, 0.6])
    observation = arm.get_observation()
    assert observation["joint_pos"].dtype == np.float32


def test_cartesian_servo_starts_tracker_and_sets_target(arm, franky, log):
    log.clear()
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert log == [
        "join_motion",
        "recover_from_errors",
        "CartesianImpedanceTracker",
        "move:CartesianImpedanceTracker",
    ]
    pose = np.array([0.4, 0.0, 0.3, *QUAT_Z90_XYZW])
    arm.send_action({"tcp_pose": pose})
    target = arm._handle.tracker.targets[-1]
    assert isinstance(target, franky.Affine)
    np.testing.assert_allclose(target.translation, pose[:3])
    np.testing.assert_allclose(target.quaternion, pose[3:])


def test_switch_to_joint_servo_stops_old_tracker_and_recovers(arm, franky, log):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    old_tracker = arm._handle.tracker
    log.clear()
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    stop_index = log.index("CartesianImpedanceTracker.stop")
    assert log[stop_index:][-4:] == [
        "join_motion",
        "recover_from_errors",
        "JointImpedanceTracker",
        "move:JointImpedanceTracker",
    ]
    assert isinstance(arm._handle.tracker, franky.JointImpedanceTracker)
    assert arm._handle.tracker is not old_tracker
    q = np.full(7, 0.2)
    arm.send_action({"joint_pos": q})
    np.testing.assert_allclose(arm._handle.tracker.targets[-1], q)


def test_move_to_joints_in_servo_mode_is_a_fault(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    with pytest.raises(DeviceFault) as info:
        arm.move_to_joints(np.zeros(7))
    assert "set_control_mode" in info.value.hint


@pytest.mark.parametrize("blocking", [True, False])
def test_move_to_joints_in_idle(arm, franky, blocking):
    q = np.linspace(0.0, 0.3, 7)
    arm.move_to_joints(q, blocking=blocking)
    motion, asynchronous = arm._handle.robot.moves[-1]
    assert isinstance(motion, franky.JointMotion)
    np.testing.assert_allclose(motion.target, q)
    assert asynchronous is (not blocking)


def test_move_to_pose_in_idle(arm, franky):
    pose = np.array([0.4, 0.1, 0.2, *QUAT_Z90_XYZW])
    arm.move_to_pose(pose, blocking=False)
    motion, asynchronous = arm._handle.robot.moves[-1]
    assert isinstance(motion, franky.CartesianMotion)
    np.testing.assert_allclose(motion.target.quaternion, QUAT_Z90_XYZW)
    assert asynchronous is True


def test_stop_in_idle_is_a_noop(arm, log):
    log.clear()
    arm.stop()
    assert log == []


def test_stop_in_idle_halts_async_position_motion(arm, log):
    arm.move_to_joints(np.zeros(7), blocking=False)
    log.clear()
    arm.stop()
    assert log == ["robot.stop", "join_motion"]
    assert arm.health().ok  # the preemption caused by the deliberate stop is not a fault


def test_servo_after_stop_restarts_tracker(arm, log):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.stop()
    assert arm._handle.tracker is None
    arm.send_action({"tcp_pose": np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])})
    assert len(arm._handle.tracker.targets) == 1


def test_dead_tracker_raises_recoverable_fault(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm._handle.robot.is_in_control = False  # e.g. a reflex ended the torque motion
    with pytest.raises(DeviceFault) as info:
        arm.send_action({"joint_pos": np.zeros(7)})
    assert info.value.recoverable
    assert not arm.health().ok
    with pytest.raises(DeviceFault, match="unrecovered"):
        arm.send_action({"joint_pos": np.zeros(7)})
    arm.recover()
    arm.send_action({"joint_pos": np.zeros(7)})
    assert arm.health().ok


def test_disconnect_stops_tracker_and_never_powers_off(franky, log):
    arm = FrankyArm(robot_ip="10.0.0.2")
    arm.connect()
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    robot = arm._handle.robot
    log.clear()
    arm.disconnect()
    assert log[0] == "CartesianImpedanceTracker.stop"
    assert not any(word in entry.lower() for entry in log for word in ("power", "shutdown", "brake"))
    assert not robot.is_in_control
    assert not arm.is_connected


def test_health_records_robot_errors_once(arm):
    arm._handle.robot.has_errors = True
    arm.health()
    health = arm.health()
    assert not health.ok
    assert len(health.errors) == 1


@pytest.mark.parametrize(
    ("realtime", "flag", "ok"),
    [("enforce", "0", False), ("ignore", "0", True), ("enforce", "1", True), ("enforce", None, False)],
)
def test_preflight_realtime_kernel(franky, monkeypatch, tmp_path, realtime, flag, ok):
    flag_path = tmp_path / "realtime"
    if flag is not None:
        flag_path.write_text(f"{flag}\n")
    monkeypatch.setattr(franky_arm_module, "REALTIME_KERNEL_FLAG_PATH", str(flag_path))
    arm = FrankyArm(robot_ip="10.0.0.2", realtime=realtime)
    arm.connect()
    results = {result.name: result for result in arm.preflight()}
    check = results["mode:realtime_kernel"]
    assert check.layer == 5 and check.ok is ok
    if not ok:
        assert "PREEMPT_RT" in check.hint and "realtime: ignore" in check.hint
    if realtime == "ignore":
        assert "RealtimeConfig.Ignore" in check.message
    assert results["mode:libfranka_version"].ok


def test_preflight_robot_mode_flags_reflex_with_recover_autofix(arm, franky):
    arm._handle.robot.state.robot_mode = franky.RobotMode.Reflex
    arm.read_state()
    check = {result.name: result for result in arm.preflight()}["mode:robot_mode"]
    assert not check.ok
    assert "--fix" in check.hint
    assert check.autofix == arm.recover
    check.autofix()  # fake recover_from_errors returns the robot to Idle
    assert {result.name: result for result in arm.preflight()}["mode:robot_mode"].ok


def test_gripper_connect_failure_maps_to_device_not_found(franky):
    franky.Gripper.connect_error = franky.NetworkException("Connection refused")
    with pytest.raises(DeviceNotFound) as info:
        FrankyGripper(robot_ip="10.0.0.2").connect()
    assert "FCI" in info.value.hint


def test_gripper_open_close_move_width(gripper):
    hand = gripper._handle
    gripper.open()
    gripper.close()
    gripper.move(0.5)
    assert hand.calls == [
        ("open", 0.1),
        ("grasp", 0.0, 0.1, 30.0, 1.0, 1.0),
        ("move", 0.04, 0.1),
    ]
    assert gripper.width == pytest.approx(0.5)
    hand.width = 0.09
    assert gripper.width == 1.0
    hand.is_grasped = True
    assert gripper.is_grasped is True


@pytest.mark.parametrize("behavior", ["fail", "gripper_exception"])
def test_gripper_close_on_nothing_does_not_raise(gripper, franky, behavior):
    franky.Gripper.grasp_behavior = behavior
    gripper.close()
    assert gripper._handle.calls[-1][0] == "grasp"


def test_gripper_rejected_grasp_command_is_a_fault(gripper, franky):
    franky.Gripper.grasp_behavior = "command_exception"
    with pytest.raises(DeviceFault, match="grasp command rejected"):
        gripper.close()


def test_gripper_binary_send_action(gripper):
    gripper.send_action({"gripper": np.array([1.0])})
    gripper.send_action({"gripper": np.array([0.0])})
    assert [call[0] for call in gripper._handle.calls] == ["open", "grasp"]


def test_gripper_continuous_send_action(franky):
    gripper = FrankyGripper(robot_ip="10.0.0.2", mode=GripperMode.CONTINUOUS, speed=0.1)
    gripper.connect()
    gripper.send_action({"gripper": np.array([0.25])})
    assert gripper._handle.calls == [("move", pytest.approx(0.02), 0.1)]


def test_reflex_during_move_maps_to_recoverable_fault(arm, franky):
    def reflex(motion, asynchronous=False):
        raise franky.ControlException("libfranka: Move command aborted: motion aborted by reflex!")

    arm._handle.robot.move = reflex
    with pytest.raises(DeviceFault, match="reflex") as info:
        arm.move_to_joints(np.zeros(7))
    assert info.value.recoverable is True
    assert "recover()" in info.value.hint
    assert not arm.health().ok


def test_reflex_starting_tracker_maps_to_recoverable_fault(arm, franky):
    def reflex(motion, asynchronous=False):
        raise franky.ControlException("cartesian_reflex")

    arm._handle.robot.move = reflex
    with pytest.raises(DeviceFault, match="cartesian_reflex") as info:
        arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert info.value.recoverable is True
    assert any("cartesian_reflex" in error for error in arm.health().errors)


def test_reflex_of_async_move_survives_mode_switch_until_recover(arm, franky):
    arm.move_to_joints(np.zeros(7), blocking=False)
    robot = arm._handle.robot
    robot.is_in_control = False  # the motion already ended, with a reflex
    robot.pending_motion_error = franky.ControlException("joint_reflex")
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    health = arm.health()
    assert not health.ok
    assert any("joint_reflex" in error for error in health.errors)
    arm.recover()
    assert arm.health().ok


def test_deliberate_stop_before_mode_switch_is_not_an_error(arm):
    arm.move_to_joints(np.zeros(7), blocking=False)
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    assert arm.health().ok


def test_recover_failure_keeps_errors(arm):
    arm._note_error("reflex")
    arm._handle.robot.recover_from_errors = lambda: False
    with pytest.raises(DeviceFault, match="did not recover"):
        arm.recover()
    assert arm.health().errors == ["reflex"]


def test_gripper_from_config_rejects_borrowed_session(arm):
    with pytest.raises(ValueError, match="shares_arm_session: false"):
        FrankyGripper.from_config(FrankaConfig(robot_ip="10.0.0.2"), owner=arm)

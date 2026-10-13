"""Hardware-free tests for the Arm contract via FakeArm."""
from __future__ import annotations

import numpy as np
import pytest

from robot_infra.devices import Arm, ControlMode, DeviceFault, DeviceNotConnected
from robot_infra.devices.backends.arms.fake import FakeArm


@pytest.fixture
def arm():
    arm = FakeArm(num_joints=7)
    arm.connect()
    yield arm
    arm.disconnect()


def test_registry_resolves_fake():
    assert Arm.backend("fake") is FakeArm


@pytest.mark.parametrize("n", [7, 6])
def test_observation_features_shapes(n):
    features = FakeArm(num_joints=n).observation_features
    assert features["tcp_pose"]["shape"] == (7,)
    assert features["tcp_vel"]["shape"] == (6,)
    assert features["joint_pos"]["shape"] == (n,)
    assert features["joint_vel"]["shape"] == (n,)
    assert features["tcp_force"]["shape"] == (3,)
    assert features["tcp_torque"]["shape"] == (3,)


def test_narrowed_state_fields_only_exposed():
    class NarrowArm(FakeArm):
        SUPPORTED_STATE_FIELDS = ("tcp_pose", "joint_pos")

    arm = NarrowArm(num_joints=6)
    arm.connect()
    assert set(arm.observation_features) == {"tcp_pose", "joint_pos"}
    observation = arm.get_observation()
    assert set(observation) == {"tcp_pose", "joint_pos"}
    assert observation["joint_pos"].dtype == np.float32


def test_missing_state_field_names_field_and_backend():
    class BrokenArm(FakeArm):
        def read_state(self):
            state = super().read_state()
            del state["tcp_vel"]
            return state

    arm = BrokenArm()
    arm.connect()
    with pytest.raises(KeyError, match="tcp_vel.*BrokenArm"):
        arm.get_observation()


def test_get_observation_before_connect():
    with pytest.raises(DeviceNotConnected):
        FakeArm().get_observation()


def test_send_action_in_idle_is_a_fault_with_hint(arm):
    with pytest.raises(DeviceFault) as info:
        arm.send_action({"tcp_pose": np.zeros(7)})
    assert "set_control_mode" in info.value.hint
    assert arm.action_features == {}


def test_set_control_mode_stops_before_entering(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert arm.call_log == ["stop", "enter:cartesian_servo"]
    assert arm.control_mode is ControlMode.CARTESIAN_SERVO
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert arm.call_log == ["stop", "enter:cartesian_servo"]


def test_action_features_follow_mode(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert arm.action_features == {"tcp_pose": {"shape": (7,), "dtype": "float32"}}
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    assert arm.action_features == {"joint_pos": {"shape": (7,), "dtype": "float32"}}


def test_cartesian_wrong_shape_and_missing_key(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    with pytest.raises(ValueError):
        arm.send_action({"tcp_pose": np.zeros(6)})
    with pytest.raises(ValueError, match="tcp_pose"):
        arm.send_action({"joint_pos": np.zeros(7)})


def test_joint_servo_converges(arm):
    target = np.linspace(0.1, 0.7, 7)
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.send_action({"joint_pos": target})
    for _ in range(10):
        state = arm.read_state()
    assert np.abs(state["joint_pos"] - target).max() < 1e-2


def test_cartesian_servo_moves_position_and_snaps_quaternion(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    pose = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    arm.send_action({"tcp_pose": pose})
    for _ in range(10):
        state = arm.read_state()
    assert np.allclose(state["tcp_pose"][:3], pose[:3], atol=1e-2)
    assert np.array_equal(state["tcp_pose"][3:], pose[3:])


def test_move_to_joints_blocking_snaps(arm):
    target = np.full(7, 0.5)
    arm.move_to_joints(target, blocking=True)
    assert np.array_equal(arm.read_state()["joint_pos"], target)


def test_health_tracks_first_feedback(arm):
    assert arm.health().first_feedback_received is False
    arm.read_state()
    health = arm.health()
    assert health.first_feedback_received is True
    assert health.last_feedback_age_s < 1.0
    assert health.ok


def test_no_feedback_fault(arm):
    arm.inject("no_feedback")
    arm.read_state()
    assert arm.health().first_feedback_received is False


def test_fault_raises_recoverable_and_recover_clears(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.inject("fault")
    with pytest.raises(DeviceFault) as info:
        arm.send_action({"joint_pos": np.zeros(7)})
    assert info.value.recoverable is True
    health = arm.health()
    assert health.errors and not health.ok
    arm.recover()
    assert arm.health().errors == []
    arm.send_action({"joint_pos": np.zeros(7)})


def test_drop_commands_leaves_state_unchanged(arm):
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.inject("drop_commands")
    arm.send_action({"joint_pos": np.ones(7)})
    for _ in range(5):
        state = arm.read_state()
    assert "servo_joints" in arm.call_log
    assert np.array_equal(state["joint_pos"], np.zeros(7))

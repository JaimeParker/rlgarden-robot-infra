"""RemoteRobot over LocalClient over RobotService over a fake Franka."""
from __future__ import annotations

import numpy as np
import pytest

import robot_infra.devices.claims as claims
from robot_infra.devices import (
    Arm,
    CameraConfig,
    ControlMode,
    DeviceFault,
    Gripper,
    GripperConfig,
    Robot,
)
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.doctor.checks import CheckResult
from robot_infra.server import RemoteRobot, RemoteRobotConfig
from robot_infra.server.remote import RemoteCamera
from robot_infra.server.service import RobotService
from robot_infra.server.transport import LocalClient


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")
    robot = FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake"),
            cameras=[CameraConfig(name="wrist", backend="fake", fps=60, width=32, height=24)],
        )
    )
    # A long watchdog keeps the remote arm from idling between test steps.
    service = RobotService(robot, watchdog_s=30.0)
    yield service
    service.close()


@pytest.fixture
def remote(service):
    robot = RemoteRobot.from_client(LocalClient(service))
    yield robot
    robot.disconnect()


@pytest.fixture
def connected(remote):
    remote.connect()
    return remote


def test_not_connected_before_connect(remote):
    assert remote.children == {}
    assert remote.describe_text() == "remote (remote @ local) — not connected"
    assert remote.health().ok is False


def test_connect_builds_parts_from_describe(connected):
    assert isinstance(connected.arm, Arm) and isinstance(connected.gripper, Gripper)
    assert connected.arm.num_joints == 7
    assert connected.arm.SUPPORTED_STATE_FIELDS == (
        "tcp_pose",
        "tcp_vel",
        "joint_pos",
        "joint_vel",
        "tcp_force",
        "tcp_torque",
    )
    assert set(connected.cameras) == {"wrist"}
    assert isinstance(connected.cameras["wrist"], RemoteCamera)
    assert connected.arm.is_connected and connected.cameras["wrist"].is_connected
    assert connected.observation_features["arm"]["joint_pos"]["shape"] == (7,)
    assert connected.observation_features["cameras"]["wrist"]["color"]["shape"] == (24, 32, 3)
    assert "RemoteArm" in connected.describe_text()


def test_get_observation_keys_and_dtypes(connected):
    observation = connected.get_observation()
    assert observation.keys() == {"arm", "gripper", "cameras"}
    assert observation["arm"]["joint_pos"].dtype == np.float32
    assert observation["gripper"]["gripper_pos"].dtype == np.float32
    assert observation["cameras"]["wrist"]["color"].dtype == np.uint8
    assert observation["cameras"]["wrist"]["color"].shape == (24, 32, 3)
    assert connected.arm.get_observation()["tcp_pose"].shape == (7,)
    frame = connected.cameras["wrist"].read()
    assert connected.cameras["wrist"].latest() is frame
    assert connected.cameras["wrist"].is_ready()


def test_step_round_trip(service, connected):
    connected.arm.set_control_mode(ControlMode.JOINT_SERVO)
    observation = connected.step({"arm": {"joint_pos": np.ones(7)}, "gripper": {"gripper": np.array([0.0])}})
    assert observation["gripper"]["gripper_pos"][0] == 0.0
    assert service.dispatch("robot.health", {})["last_action_age_s"] is not None


def test_set_control_mode_mirrors_remote(service, connected):
    connected.arm.set_control_mode(ControlMode.JOINT_SERVO)
    assert service.robot.arm.control_mode is ControlMode.JOINT_SERVO
    assert connected.arm.control_mode is ControlMode.JOINT_SERVO
    assert connected.arm.health().control_mode == "joint_servo"


def test_send_action_dispatches_through_part_group(service, connected):
    connected.arm.set_control_mode(ControlMode.JOINT_SERVO)
    connected.send_action({"arm": {"joint_pos": np.full(7, 0.3)}, "gripper": {"gripper": [0.0]}})
    assert service.robot.gripper.width == 0.0
    connected.send_action({"arm": {"joint_pos": np.full(7, 0.3)}, "gripper": {"gripper": [0.9]}})
    assert service.robot.gripper.width == 1.0
    for _ in range(12):
        state = connected.arm.read_state()
    np.testing.assert_allclose(state["joint_pos"], 0.3, atol=1e-3)


def test_move_to_joints_blocking_snaps(connected):
    connected.arm.move_to_joints(np.full(7, 0.5), blocking=True)
    np.testing.assert_allclose(connected.arm.read_state()["joint_pos"], 0.5)


def test_remote_device_fault_surfaces_with_hint(service, connected):
    service.robot.arm.inject("fault")
    with pytest.raises(DeviceFault) as info:
        connected.arm.servo_joints(np.ones(7))
    assert "recover()" in info.value.hint
    connected.arm.recover()
    connected.arm.servo_joints(np.ones(7))


def test_preflight_and_health(connected):
    results = connected.arm.preflight()
    assert isinstance(results, list) and all(isinstance(r, CheckResult) for r in results)
    health = connected.health()
    assert health.ok is True
    assert "rtt=" in health.detail and "watchdog_trips=0" in health.detail
    assert connected.arm.health().ok is True
    assert connected.cameras["wrist"].health() is not None
    assert connected.remote_health()["connected"] is True


def test_disconnect_and_reconnect_without_duplicates(service, connected):
    connected.disconnect()
    assert service.dispatch("robot.health", {})["connected"] is False
    assert connected.children == {}
    assert not connected.is_connected
    assert connected.client.is_open is False
    connected.connect()
    assert list(connected.children) == ["arm", "gripper", "cameras"]
    assert set(connected.cameras) == {"wrist"}
    assert service.dispatch("robot.health", {})["connected"] is True
    connected.connect()  # idempotent
    assert list(connected.children) == ["arm", "gripper", "cameras"]


def test_from_dict_builds_without_network():
    robot = Robot.from_dict({"type": "remote", "address": "127.0.0.1:1"})
    assert isinstance(robot, RemoteRobot)
    assert robot.client is None and robot.children == {}
    assert "127.0.0.1:1" in robot.describe_text()


def test_config_validation():
    with pytest.raises(ValueError, match="transport"):
        RemoteRobotConfig(transport="zmq")
    with pytest.raises(ValueError, match="from_client"):
        RemoteRobot(RemoteRobotConfig(transport="local"))

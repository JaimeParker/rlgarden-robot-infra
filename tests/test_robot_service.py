"""RobotService over a fake Franka: method table, claims, health and the watchdog."""
from __future__ import annotations

import time

import numpy as np
import pytest

import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, DeviceBusy, DeviceFault, DeviceNotFound, GripperConfig
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.server.service import METHODS, RobotService


def franka_robot(**kwargs) -> FrankaRobot:
    config = FrankaConfig(
        arm_backend="fake",
        gripper=GripperConfig(backend="fake"),
        cameras=[CameraConfig(name="wrist", backend="fake", fps=60, width=32, height=24)],
        **kwargs,
    )
    return FrankaRobot(config)


@pytest.fixture(autouse=True)
def isolated_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


@pytest.fixture
def services():
    created: list[RobotService] = []

    def make(robot=None, **kwargs) -> RobotService:
        service = RobotService(robot or franka_robot(), **kwargs)
        created.append(service)
        return service

    yield make
    for service in created:
        service.close()


@pytest.fixture
def connected(services) -> RobotService:
    service = services()
    service.dispatch("robot.connect", {})
    return service


def test_method_table_matches_spec(services):
    service = services()
    assert set(service.methods) == (set(METHODS) - {"camera.<name>.read"}) | {"camera.wrist.read"}
    with pytest.raises(ValueError, match=r"unknown method 'arm.fly'.*arm.servo_joints"):
        service.dispatch("arm.fly", {})


def test_describe(services):
    description = services().dispatch("robot.describe", {})
    assert description["type"] == "franka"
    assert description["describe"]["type"] == "franka"
    assert description["arm"] == {
        "num_joints": 7,
        "supported_state_fields": [
            "tcp_pose",
            "tcp_vel",
            "joint_pos",
            "joint_vel",
            "tcp_force",
            "tcp_torque",
        ],
        "control_mode": "idle",
    }
    assert description["gripper"] == {"max_width_m": 0.08, "mode": "binary", "binary_threshold": 0.5}
    assert description["cameras"]["wrist"]["fps"] == 60
    assert description["cameras"]["wrist"]["backend"] == "fake"


def test_connect_disconnect_toggle_connected(services):
    service = services()
    assert service.dispatch("robot.health", {})["connected"] is False
    service.dispatch("robot.connect", {})
    service.dispatch("robot.connect", {})  # idempotent: the claim is not re-acquired
    assert service.dispatch("robot.health", {})["connected"] is True
    assert service.watchdog_running
    service.dispatch("robot.disconnect", {})
    assert service.dispatch("robot.health", {})["connected"] is False


def test_claim_contention_and_release(services):
    first = services()
    second = services()
    first.dispatch("robot.connect", {})
    with pytest.raises(DeviceBusy) as info:
        second.dispatch("robot.connect", {})
    assert "franka:172.16.0.2" in str(info.value) and "claim=False" in info.value.hint
    assert not second.robot.is_connected
    first.dispatch("robot.disconnect", {})
    second.dispatch("robot.connect", {})
    assert second.robot.is_connected


def test_claim_disabled_by_config(services):
    first = services(franka_robot(claim=False))
    second = services(franka_robot(claim=False))
    first.dispatch("robot.connect", {})
    second.dispatch("robot.connect", {})


def test_joint_servo_moves_arm(connected):
    connected.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    assert connected.dispatch("arm.control_mode", {}) == "joint_servo"
    connected.dispatch("arm.servo_joints", {"q": np.ones(7)})
    for _ in range(10):
        observation = connected.dispatch("robot.get_observation", {})
    np.testing.assert_allclose(observation["arm"]["joint_pos"], 1.0, atol=1e-2)
    assert observation["cameras"]["wrist"]["color"].shape == (24, 32, 3)


def test_set_control_mode_rejects_unknown(connected):
    with pytest.raises(ValueError, match="cartesian_servo"):
        connected.dispatch("arm.set_control_mode", {"mode": "dance"})


def test_step_returns_observation_and_marks_action(connected):
    assert connected.dispatch("robot.health", {})["last_action_age_s"] is None
    connected.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    observation = connected.dispatch(
        "robot.step", {"action": {"arm": {"joint_pos": np.ones(7)}, "gripper": {"gripper": np.array([0.0])}}}
    )
    assert observation.keys() == {"arm", "gripper", "cameras"}
    assert observation["gripper"]["gripper_pos"][0] == 0.0
    age = connected.dispatch("robot.health", {})["last_action_age_s"]
    assert age is not None and 0.0 <= age < 0.5


def test_arm_moves_health_preflight(connected):
    connected.dispatch("arm.move_to_joints", {"q": [0.5] * 7, "blocking": True})
    np.testing.assert_allclose(connected.dispatch("arm.read_state", {})["joint_pos"], 0.5)
    connected.dispatch("arm.stop", {})
    connected.dispatch("arm.recover", {})
    health = connected.dispatch("arm.health", {})
    assert health["ok"] is True and health["control_mode"] == "idle"
    assert connected.dispatch("arm.preflight", {}) == []


def test_gripper_methods(connected):
    connected.dispatch("gripper.close", {})
    assert connected.dispatch("gripper.width", {}) == 0.0
    connected.dispatch("gripper.open", {})
    assert connected.dispatch("gripper.width", {}) == 1.0
    connected.dispatch("gripper.move", {"width": 0.25})
    assert connected.dispatch("gripper.width", {}) == 0.25


def test_gripperless_robot_raises_device_not_found(services):
    robot = FrankaRobot(FrankaConfig(arm_backend="fake", gripper=None))
    service = services(robot)
    assert service.dispatch("robot.describe", {})["gripper"] is None
    with pytest.raises(DeviceNotFound) as info:
        service.dispatch("gripper.open", {})
    assert info.value.hint == "this robot has no gripper"


def test_camera_read(connected):
    frame = connected.dispatch("camera.wrist.read", {})
    assert frame.keys() == {"color", "depth", "timestamp"}
    assert frame["color"].shape == (24, 32, 3) and frame["depth"] is None


def test_health_layout(connected):
    health = connected.dispatch("robot.health", {})
    assert health["watchdog_trips"] == 0
    assert "last_action_age_s" in health
    assert health["robot"]["ok"] is True
    assert health["arm"]["control_mode"] == "idle"
    assert "ok" in health["gripper"] and "ok" in health["cameras"]["wrist"]


def test_watchdog_idles_quiet_servo(services):
    service = services(watchdog_s=0.1)
    service.dispatch("robot.connect", {})
    arm = service.robot.arm
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    service.dispatch("arm.servo_joints", {"q": np.ones(7)})
    calls_before_trip = len(arm.call_log)
    time.sleep(0.4)
    assert service.dispatch("arm.control_mode", {}) == "idle"
    assert service.watchdog_trips == 1
    assert service.dispatch("robot.health", {})["watchdog_trips"] == 1
    assert "stop" in arm.call_log[calls_before_trip:]


def test_watchdog_trip_explains_itself_until_mode_is_set_again(services):
    service = services(watchdog_s=0.1)
    service.dispatch("robot.connect", {})
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    service.dispatch("arm.servo_joints", {"q": np.ones(7)})
    time.sleep(0.4)
    with pytest.raises(DeviceFault, match="watchdog") as info:
        service.dispatch("arm.servo_joints", {"q": np.ones(7)})
    assert info.value.recoverable and "set_control_mode" in info.value.hint
    with pytest.raises(DeviceFault, match="watchdog"):
        service.dispatch("robot.step", {"action": {"arm": {"joint_pos": np.ones(7)}}})
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    service.dispatch("arm.servo_joints", {"q": np.ones(7)})


def test_watchdog_trips_servo_without_any_action(services):
    service = services(watchdog_s=0.1)
    service.dispatch("robot.connect", {})
    service.dispatch("arm.set_control_mode", {"mode": "cartesian_servo"})
    time.sleep(0.4)
    assert service.dispatch("arm.control_mode", {}) == "idle"
    assert service.watchdog_trips == 1


def test_watchdog_ignores_idle_and_disconnected_arm(services):
    service = services(watchdog_s=0.1)
    service.dispatch("robot.connect", {})
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    service.dispatch("robot.disconnect", {})
    time.sleep(0.3)
    assert service.watchdog_trips == 0


def test_close_disconnects_stops_watchdog_and_releases_claim(services):
    service = services()
    service.dispatch("robot.connect", {})
    assert service.watchdog_running
    service.close()
    assert not service.robot.is_connected
    assert not service.watchdog_running
    other = services()
    other.dispatch("robot.connect", {})
    assert other.robot.is_connected

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
    templates = {"camera.<name>.read", "teleop.<name>.read", "teleop.<name>.reset"}
    assert set(service.methods) == (set(METHODS) - templates) | {"camera.wrist.read"}
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
        "safety": None,
        "home_joints": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
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
    assert "franka:172.16.0.2" in str(info.value) and "`claim: false`" in info.value.hint
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


TELEOP_SCRIPT = [
    {"action": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0], "active": True, "buttons": {"left": True}},
    {"events": {"episode_end": True}},
]


def teleop_robot(**teleop) -> FrankaRobot:
    from robot_infra.devices import TeleopConfig

    config = TeleopConfig(backend="fake", options={"script": TELEOP_SCRIPT}, **teleop)
    return franka_robot(teleop=config)


def test_teleop_methods_describe_and_health(services):
    service = services(teleop_robot(hold_s=0.25))
    assert {"teleop.teleop.read", "teleop.teleop.reset"} <= set(service.methods)
    assert service.dispatch("robot.describe", {})["teleop"] == {
        "name": "teleop",
        "backend": "fake",
        "action_space": "cartesian_delta",
        "has_gripper": True,
        "hold_s": 0.25,
    }
    service.dispatch("robot.connect", {})
    sample = service.dispatch("teleop.teleop.read", {"timeout_s": 1.0})
    assert sample.keys() == {
        "action", "action_space", "gripper", "buttons", "events", "intervening", "timestamp", "seq"
    }
    assert sample["action"].dtype == np.float32 and sample["intervening"] is True
    assert sample["buttons"] == {"left": True}
    assert service.dispatch("teleop.teleop.read", {})["events"]["episode_end"] is True
    service.dispatch("teleop.teleop.reset", {})
    health = service.dispatch("robot.health", {})
    assert health["teleop"]["ok"] is True and health["teleop"]["thread_alive"] is True
    observation = service.dispatch("robot.get_observation", {})
    assert observation["teleop"]["action"].shape == (6,)


@pytest.mark.parametrize("requested, expected", [(1e9, 2.0), (None, 2.0), (-5.0, 0.0), (0.25, 0.25)])
def test_teleop_read_timeout_is_clamped(services, monkeypatch, requested, expected):
    from robot_infra.server.service import TELEOP_READ_MAX_TIMEOUT_S

    assert TELEOP_READ_MAX_TIMEOUT_S == 2.0
    service = services(teleop_robot())
    service.dispatch("robot.connect", {})
    seen = []
    real_read = service.robot.teleop.read
    monkeypatch.setattr(service.robot.teleop, "read", lambda timeout_s=None: seen.append(timeout_s) or real_read(1.0))
    service.dispatch("teleop.teleop.read", {} if requested is None else {"timeout_s": requested})
    assert seen == [expected]


def test_no_teleop_means_no_teleop_methods(services):
    service = services()
    assert not any(method.startswith("teleop.") for method in service.methods)
    assert service.dispatch("robot.describe", {})["teleop"] is None


def test_teleop_read_does_not_feed_the_watchdog(services):
    service = services(teleop_robot(), watchdog_s=0.1)
    service.dispatch("robot.connect", {})
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    service.dispatch("arm.servo_joints", {"q": np.ones(7)})
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        service.dispatch("teleop.teleop.read", {"timeout_s": 1.0})
    assert service.watchdog_trips == 1
    assert service.dispatch("arm.control_mode", {}) == "idle"

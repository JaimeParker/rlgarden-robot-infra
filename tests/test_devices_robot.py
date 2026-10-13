"""Hardware-free tests for the Robot composition layer using fake backends."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from robot_infra.devices import CameraConfig, ControlMode, GripperConfig, Robot
from robot_infra.devices.backends.grippers.fake import FakeGripper
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.devices.robots.piper import PiperRobot


def franka_config(**kwargs) -> FrankaConfig:
    return FrankaConfig(
        arm_backend="fake",
        gripper=GripperConfig(backend="fake"),
        cameras=[CameraConfig(name="wrist", backend="fake", fps=60)],
        **kwargs,
    )


def piper_dict() -> dict:
    return {
        "type": "piper",
        "arm_backend": "fake",
        "gripper": {
            "backend": "fake",
            "shares_arm_session": True,
            "mode": "continuous",
            "max_width_m": 0.07,
        },
    }


def test_build_does_not_touch_hardware():
    robot = FrankaRobot(franka_config())
    assert robot.arm.num_joints == 7
    assert isinstance(robot.gripper, FakeGripper)
    assert robot.cameras.keys() == {"wrist"}
    parts = [robot, robot.arm, robot.gripper, *robot.cameras.values()]
    assert not any(part.is_connected for part in parts)
    text = robot.describe_text()
    assert "franka" in text and "wrist" in text and "FakeArm" in text
    assert not any(part.is_connected for part in parts)
    assert robot.describe()["type"] == "franka"


def test_piper_from_dict_shared_session_lifecycle():
    robot = Robot.from_dict(piper_dict())
    assert isinstance(robot, PiperRobot)
    assert robot.arm.num_joints == 6
    assert robot.gripper.owner is robot.arm
    robot.connect()
    assert robot.arm.is_connected and robot.gripper.is_connected
    observation = robot.get_observation()
    assert observation.keys() == {"arm", "gripper"}
    assert observation["arm"]["joint_pos"].shape == (6,)
    robot.disconnect()
    assert not robot.arm.is_connected and not robot.gripper.is_connected
    assert not robot.is_connected


@pytest.mark.parametrize("wrap", [False, True])
def test_yaml_round_trip(tmp_path, wrap):
    config = franka_config()
    before = config.to_dict()
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": before} if wrap else before))
    robot = Robot.from_yaml(path)
    assert isinstance(robot, FrankaRobot)
    assert robot.config.to_dict() == before


def test_unknown_and_missing_type():
    with pytest.raises(ValueError, match=r"unknown robot type 'ur5'.*franka.*piper"):
        Robot.from_dict({"type": "ur5"})
    with pytest.raises(ValueError, match="type: franka"):
        Robot.from_dict({"name": "x"})


def test_unknown_arm_backend_lists_registered():
    with pytest.raises(ValueError, match=r"'nonexistent'.*registered: \[.*'fake'.*\]"):
        FrankaRobot(FrankaConfig(arm_backend="nonexistent", gripper=None))


def test_config_validation():
    with pytest.raises(ValueError, match="home_joints"):
        FrankaConfig(home_joints=[0.0] * 3)
    with pytest.raises(ValueError, match="safety_box"):
        FrankaConfig(safety_box=[0.0] * 5)
    with pytest.raises(ValueError, match="sideways"):
        FrankaConfig(gripper=GripperConfig(mode="sideways"))
    with pytest.raises(ValueError, match="realtime"):
        FrankaConfig(realtime="maybe")
    with pytest.raises(ValueError, match="bogus.*FrankaConfig"):
        Robot.from_dict({"type": "franka", "bogus": 1})


def test_full_flow_with_fakes():
    robot = FrankaRobot(franka_config())
    robot.connect()
    try:
        robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
        robot.send_action(
            {"arm": {"joint_pos": np.ones(7)}, "gripper": {"gripper": np.array([0.9])}}
        )
        for _ in range(10):
            observation = robot.get_observation()
        np.testing.assert_allclose(observation["arm"]["joint_pos"], 1.0, atol=1e-2)
        assert observation["gripper"]["gripper_pos"][0] == 1.0
        assert observation["cameras"]["wrist"]["color"].shape == (480, 640, 3)
    finally:
        robot.disconnect()


def test_type_registry():
    assert {"franka", "piper"} <= Robot.types().keys()

    class MismatchedConfig(FrankaConfig):
        TYPE = "other"

    class Mismatched(Robot):
        ROBOT_TYPE = "mismatch"
        CONFIG = MismatchedConfig

        def build(self) -> None:
            pass

    with pytest.raises(ValueError, match="mismatch"):
        Robot.register_type(Mismatched)
    assert "mismatch" not in Robot.types()


def test_unknown_camera_key_names_key_and_config():
    from robot_infra.devices import RobotConfig

    with pytest.raises(ValueError, match="unknown key 'fpss' for CameraConfig"):
        RobotConfig.from_dict({"cameras": [{"name": "wrist", "fpss": 30}]})


def piper_with_teleop(**teleop) -> dict:
    return {**piper_dict(), "teleop": {"backend": "fake", **teleop}}


def test_teleop_config_parses_and_is_the_last_child():
    from robot_infra.devices import TeleopConfig
    from robot_infra.devices.backends.teleops.fake import FakeTeleop

    robot = Robot.from_dict(piper_with_teleop(name="sm", hold_s=0.2, options={"on_end": "idle"}))
    assert isinstance(robot.config.teleop, TeleopConfig) and robot.config.teleop.hold_s == 0.2
    assert isinstance(robot.teleop, FakeTeleop) and robot.teleop.name == "sm"
    assert list(robot.children) == ["arm", "gripper", "teleop"]
    assert Robot.from_dict(piper_dict()).teleop is None
    assert Robot.from_dict({**piper_dict(), "teleop": None}).teleop is None


def test_teleop_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown key 'deadzone' for TeleopConfig"):
        Robot.from_dict(piper_with_teleop(deadzone=0.1))


def test_teleop_yaml_round_trip(tmp_path):
    robot = Robot.from_dict(piper_with_teleop(options={"script": [{"active": True}]}))
    before = robot.config.to_dict()
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(before))
    assert Robot.from_yaml(path).config.to_dict() == before


def test_teleop_disconnects_before_the_arm(monkeypatch):
    from robot_infra.devices.backends.arms.fake import FakeArm
    from robot_infra.devices.backends.teleops.fake import FakeTeleop

    order: list[str] = []
    arm_release = FakeArm._release
    teleop_close = FakeTeleop._close_device
    monkeypatch.setattr(FakeArm, "_release", lambda self, h: (order.append("arm"), arm_release(self, h))[1])
    monkeypatch.setattr(
        FakeTeleop, "_close_device", lambda self, h: (order.append("teleop"), teleop_close(self, h))[1]
    )
    robot = Robot.from_dict(piper_with_teleop(options={"script": [{"action": [0.1] * 6}]}))
    robot.connect()
    assert robot.teleop.is_connected
    observation = robot.get_observation()
    assert observation.keys() == {"arm", "gripper", "teleop"}
    assert observation["teleop"]["action"].shape == (6,)
    assert float(robot.teleop.read().action[0]) == pytest.approx(0.1)
    robot.disconnect()
    assert order == ["teleop", "arm"]
    assert not robot.teleop.is_connected

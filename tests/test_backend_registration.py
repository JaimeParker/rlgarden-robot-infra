"""Importing robot_infra.devices registers all backends without any vendor SDK installed."""
from __future__ import annotations

import pytest

from robot_infra.devices import Arm, Camera, Gripper, Robot

SDK_BACKENDS = {"fake", "franky", "serl_http", "pyagxarm", "piper_sdk"}


def test_arm_and_gripper_backends_registered():
    assert SDK_BACKENDS <= set(Arm.backends())
    assert SDK_BACKENDS <= set(Gripper.backends())


def test_camera_backends_registered():
    assert {"fake", "realsense"} <= set(Camera.backends())


@pytest.mark.parametrize("robot_type", ["franka", "piper"])
def test_default_backends_build_without_sdk_or_hardware(robot_type):
    robot = Robot.from_dict({"type": robot_type})
    assert not robot.arm.is_connected
    if robot.gripper is not None:
        assert not robot.gripper.is_connected
    assert robot.describe_text()

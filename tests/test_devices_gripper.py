"""Hardware-free tests for the Gripper contract via FakeGripper."""
from __future__ import annotations

import numpy as np
import pytest

from robot_infra.devices import DeviceNotConnected, Gripper, GripperMode, PartGroup
from robot_infra.devices.backends.arms.fake import FakeArm
from robot_infra.devices.backends.grippers.fake import FakeGripper


def connected(**kwargs) -> FakeGripper:
    gripper = FakeGripper(**kwargs)
    gripper.connect()
    return gripper


def test_binary_threshold_opens_and_closes():
    gripper = connected()
    gripper.send_action({"gripper": np.array([0.9])})
    assert gripper.width == 1.0
    gripper.send_action({"gripper": np.array([0.1])})
    assert gripper.width == 0.0


def test_continuous_moves_and_clips():
    gripper = connected(mode=GripperMode.CONTINUOUS)
    gripper.send_action({"gripper": np.array([0.3])})
    assert gripper.width == pytest.approx(0.3)
    gripper.send_action({"gripper": np.array([1.7])})
    assert gripper.width == 1.0
    gripper.send_action({"gripper": np.array([-0.4])})
    assert gripper.width == 0.0


def test_get_observation_dtype_and_shape():
    observation = connected().get_observation()["gripper_pos"]
    assert observation.dtype == np.float32
    assert observation.shape == (1,)


def test_send_action_before_connect():
    with pytest.raises(DeviceNotConnected):
        FakeGripper().send_action({"gripper": np.array([1.0])})


def test_owner_borrowing_connects_arm():
    arm = FakeArm()
    gripper = FakeGripper(owner=arm)
    assert not arm.is_connected
    gripper.connect()
    assert arm.is_connected and gripper.is_connected
    assert gripper.call_log == ["opened"]
    gripper.disconnect()
    assert gripper.call_log == ["opened", "closing"]


def test_part_group_observation_has_both_keys():
    arm = FakeArm()
    group = PartGroup("robot")
    group.add("arm", arm)
    group.add("gripper", FakeGripper(owner=arm))
    group.connect()
    observation = group.get_observation()
    assert "tcp_pose" in observation["arm"]
    assert "gripper_pos" in observation["gripper"]
    group.disconnect()


def test_closed_action_value_and_registry():
    assert Gripper.closed_action_value == 0.0
    assert FakeGripper().is_grasped is None
    assert Gripper.backend("fake") is FakeGripper

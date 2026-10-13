"""LocalClient over RobotService with a codec round trip on every call."""
from __future__ import annotations

import time

import numpy as np
import pytest

import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, DeviceFault, GripperConfig
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.server.service import RobotService
from robot_infra.server.transport import LocalClient, LocalTransport, TransportError, is_read_only


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
    service = RobotService(robot)
    LocalTransport().start(service, "local")
    yield service
    service.close()


@pytest.fixture
def client(service):
    client = LocalClient(service)
    client.open()
    client.call("robot.connect")
    yield client
    client.close()


def test_observation_arrays_survive_round_trip(client):
    observation = client.call("robot.get_observation")
    assert isinstance(observation["arm"]["joint_pos"], np.ndarray)
    assert observation["arm"]["joint_pos"].dtype == np.float32
    assert observation["gripper"]["gripper_pos"].dtype == np.float32
    assert observation["cameras"]["wrist"]["color"].dtype == np.uint8
    assert observation["cameras"]["wrist"]["color"].shape == (24, 32, 3)
    assert client.last_rtt_ms is not None and client.last_rtt_ms >= 0.0


def test_servo_kwargs_round_trip(client):
    client.call("arm.set_control_mode", mode="joint_servo")
    client.call("arm.servo_joints", q=np.ones(7, dtype=np.float32))
    for _ in range(10):
        state = client.call("arm.read_state")
    np.testing.assert_allclose(state["joint_pos"], 1.0, atol=1e-2)


def test_remote_device_error_keeps_type_and_hint(service, client):
    client.call("arm.set_control_mode", mode="joint_servo")
    service.robot.arm.inject("fault")
    with pytest.raises(DeviceFault) as info:
        client.call("arm.servo_joints", q=np.ones(7))
    assert info.value.recoverable and "recover()" in info.value.hint


def test_closed_client_raises_transport_error(service):
    client = LocalClient(service)
    with pytest.raises(TransportError) as info:
        client.call("robot.describe")
    assert "open()" in info.value.hint


def test_subscribe_rate_and_stop_on_close(client):
    client.call("camera.wrist.read")  # wait for the capture thread's first frame
    stream = client.subscribe("robot.get_observation", rate_hz=50)
    start = time.monotonic()
    received = [next(stream) for _ in range(3)]
    assert time.monotonic() - start < 0.2
    assert all(obs["arm"]["joint_pos"].shape == (7,) for obs in received)
    client.close()
    with pytest.raises(StopIteration):
        next(stream)


def test_is_read_only():
    assert is_read_only("robot.get_observation") and is_read_only("camera.wrist.read")
    assert not is_read_only("arm.servo_joints") and not is_read_only("camera.read")

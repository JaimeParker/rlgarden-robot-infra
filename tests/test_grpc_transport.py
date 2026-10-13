"""GrpcTransport/GrpcClient over loopback with a fake-backend Franka RobotService."""
from __future__ import annotations

import logging
import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("grpc")

import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, DeviceFault, GripperConfig
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.server.service import RobotService
from robot_infra.server.transport import GrpcClient, GrpcTransport, TransportError


def make_service(width: int = 32, height: int = 24) -> RobotService:
    robot = FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake"),
            cameras=[CameraConfig(name="wrist", backend="fake", fps=60, width=width, height=height)],
        )
    )
    return RobotService(robot)


@pytest.fixture(autouse=True)
def isolated_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


@pytest.fixture
def cleanup():
    transports: list[GrpcTransport] = []
    closeables: list = []
    yield transports, closeables
    for closeable in closeables:
        closeable.close()
    for transport in transports:
        transport.stop(grace_s=0.1)


@pytest.fixture
def served(cleanup):
    """Start a daemon over ``service`` and return (transport, port)."""
    transports, closeables = cleanup

    def start(service: RobotService, *, token: str | None = None, port: int = 0):
        transport = GrpcTransport()
        transport.start(service, f"127.0.0.1:{port}", token=token)
        transports.append(transport)
        if service not in closeables:
            closeables.append(service)
        return transport, transport.port

    return start


@pytest.fixture
def clients(cleanup):
    _, closeables = cleanup

    def make(port: int, **kwargs) -> GrpcClient:
        client = GrpcClient(f"127.0.0.1:{port}", **kwargs)
        closeables.insert(0, client)  # clients close before the service
        return client

    return make


@pytest.fixture
def service():
    return make_service()


@pytest.fixture
def client(service, served, clients):
    _, port = served(service)
    client = clients(port)
    client.open()
    client.call("robot.connect")
    return client


def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_describe_and_observation_arrays(client):
    description = client.call("robot.describe")
    assert description["type"] == "franka" and description["arm"]["num_joints"] == 7
    observation = client.call("robot.get_observation")
    assert isinstance(observation["arm"]["joint_pos"], np.ndarray)
    assert observation["arm"]["joint_pos"].dtype == np.float32
    assert observation["cameras"]["wrist"]["color"].dtype == np.uint8
    assert observation["cameras"]["wrist"]["color"].shape == (24, 32, 3)
    assert client.is_open
    assert client.last_rtt_ms is not None and client.last_rtt_ms >= 0.0


def test_step(client):
    client.call("arm.set_control_mode", mode="joint_servo")
    observation = client.call(
        "robot.step", action={"arm": {"joint_pos": np.ones(7)}, "gripper": {"gripper": np.array([0.0])}}
    )
    assert observation.keys() == {"arm", "gripper", "cameras"}
    assert observation["gripper"]["gripper_pos"][0] == 0.0


def test_servo_changes_observation(client):
    client.call("arm.set_control_mode", mode="joint_servo")
    assert client.call("arm.control_mode") == "joint_servo"
    client.call("arm.servo_joints", q=np.ones(7, dtype=np.float32))
    for _ in range(10):
        observation = client.call("robot.get_observation")
    np.testing.assert_allclose(observation["arm"]["joint_pos"], 1.0, atol=1e-2)


def test_remote_device_error_keeps_type_and_hint(service, client):
    client.call("arm.set_control_mode", mode="joint_servo")
    service.robot.arm.inject("fault")
    with pytest.raises(DeviceFault) as info:
        client.call("arm.servo_joints", q=np.ones(7))
    assert not isinstance(info.value, TransportError)
    assert info.value.recoverable and "recover()" in info.value.hint


def test_unknown_method_raises_value_error(client):
    with pytest.raises(ValueError, match="unknown method 'arm.fly'"):
        client.call("arm.fly")


def test_token_mismatch_and_match(service, served, clients):
    _, port = served(service, token="abc")
    anonymous = clients(port)
    anonymous.open()
    with pytest.raises(TransportError, match="token mismatch") as info:
        anonymous.call("robot.describe")
    assert "--token" in info.value.hint
    authorized = clients(port, token="abc")
    authorized.open()
    assert authorized.call("robot.describe")["type"] == "franka"


def test_unreachable_daemon_open_raises_with_hint(clients):
    client = clients(closed_port(), timeout_s=0.5)
    start = time.monotonic()
    with pytest.raises(TransportError, match="unreachable") as info:
        client.open()
    assert time.monotonic() - start < 2.0
    assert "python -m robot_infra.server" in info.value.hint
    assert not client.is_open


def test_read_only_call_retried_after_reconnect(service, served, clients, caplog):
    transport, port = served(service)
    client = clients(port, timeout_s=1.0, reconnect_attempts=10, reconnect_backoff_s=0.1)
    client.open()
    client.call("robot.connect")
    transport.stop(grace_s=0.1)
    restart = threading.Timer(0.3, lambda: served(service, port=port))
    restart.start()
    try:
        with caplog.at_level(logging.WARNING, logger="robot_infra.transport.grpc"):
            health = client.call("robot.health")
    finally:
        restart.join()
    assert health["connected"] is True
    assert "reconnecting" in caplog.text
    assert client.is_open


def test_command_never_retried_after_disconnect(service, served, clients, caplog):
    transport, port = served(service)
    client = clients(port, timeout_s=0.3, reconnect_attempts=2, reconnect_backoff_s=0.05)
    client.open()
    client.call("robot.connect")
    transport.stop(grace_s=0.1)
    with caplog.at_level(logging.WARNING, logger="robot_infra.transport.grpc"):
        with pytest.raises(TransportError, match="lost connection") as info:
            client.call("arm.stop")
    assert "never retried" in info.value.hint
    assert "reconnecting" in caplog.text


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


def test_server_survives_client_disconnect_mid_stream(service, served, clients):
    _, port = served(service)
    first = clients(port)
    first.open()
    stream = first.subscribe("robot.health", rate_hz=100)
    next(stream)
    first.close()
    second = clients(port)
    second.open()
    assert second.call("robot.describe")["type"] == "franka"


def test_large_camera_frame_round_trips(served, clients):
    service = make_service(width=640, height=480)
    _, port = served(service)
    client = clients(port)
    client.open()
    client.call("robot.connect")
    frame = client.call("camera.wrist.read", timeout_s=2.0)
    assert frame["color"].shape == (480, 640, 3) and frame["color"].dtype == np.uint8
    observation = client.call("robot.get_observation")
    assert observation["cameras"]["wrist"]["color"].shape == (480, 640, 3)

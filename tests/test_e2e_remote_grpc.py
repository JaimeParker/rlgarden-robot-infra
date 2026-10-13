"""End-to-end: Robot.from_dict({"type": "remote"}) -> real gRPC over loopback -> RobotService -> fake Franka."""
from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("grpc")

import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, DeviceBusy, DeviceFault, GripperConfig, Robot
from robot_infra.devices.arm import ControlMode
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.server import RemoteRobot
from robot_infra.server.service import RobotService
from robot_infra.server.transport import GrpcTransport, TransportError

ACTION = {"arm": {"joint_pos": np.ones(7)}, "gripper": {"gripper": [0.9]}}


def franka_robot() -> FrankaRobot:
    return FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake"),
            cameras=[CameraConfig(name="wrist", backend="fake", fps=60, width=32, height=24)],
        )
    )


@pytest.fixture(autouse=True)
def isolated_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


@pytest.fixture
def daemon():
    """start(...) -> (service, transport); everything is torn down client-first."""
    transports: list[GrpcTransport] = []
    services: list[RobotService] = []
    remotes: list[RemoteRobot] = []

    def start(*, watchdog_s: float = 0.5, token: str | None = None, port: int = 0, robot=None):
        service = RobotService(robot or franka_robot(), watchdog_s=watchdog_s)
        transport = GrpcTransport()
        transport.start(service, f"127.0.0.1:{port}", token=token)
        services.append(service)
        transports.append(transport)
        return service, transport

    def remote(port: int, **overrides) -> RemoteRobot:
        config = {
            "type": "remote",
            "address": f"127.0.0.1:{port}",
            "timeout_s": 2.0,
            "reconnect_attempts": 3,
            "reconnect_backoff_s": 0.1,
            **overrides,
        }
        robot = Robot.from_dict(config)
        remotes.append(robot)
        return robot

    start.remote = remote
    yield start
    for robot in remotes:
        try:
            robot.disconnect()
        except Exception:
            pass
    for transport in transports:
        transport.stop(grace_s=0.1)
    for service in services:
        service.close()


def test_full_session(daemon):
    _, transport = daemon()
    robot = daemon.remote(transport.port)
    robot.connect()
    text = robot.describe_text()
    assert "arm" in text and "gripper" in text and "wrist" in text

    observation = robot.get_observation()
    assert observation["arm"]["joint_pos"].dtype == np.float32
    assert observation["arm"]["tcp_pose"].shape == (7,)
    assert observation["gripper"]["gripper_pos"].dtype == np.float32
    assert observation["cameras"]["wrist"]["color"].dtype == np.uint8
    assert observation["cameras"]["wrist"]["color"].shape == (24, 32, 3)

    robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
    for _ in range(5):
        observation = robot.step(ACTION)
    np.testing.assert_allclose(observation["arm"]["joint_pos"], 1.0, atol=0.1)
    assert observation["gripper"]["gripper_pos"][0] == 1.0

    health = robot.health()
    assert health.ok and "rtt" in health.detail
    robot.disconnect()
    assert not robot.is_connected


def test_watchdog_through_the_wire(daemon):
    _, transport = daemon(watchdog_s=0.2)
    robot = daemon.remote(transport.port)
    robot.connect()
    robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
    robot.step(ACTION)
    time.sleep(0.6)
    with pytest.raises(DeviceFault, match="watchdog") as info:
        robot.step(ACTION)
    assert info.value.recoverable
    robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
    robot.step(ACTION)
    assert robot.remote_health()["watchdog_trips"] == 1


def test_daemon_restart_mid_session(daemon):
    service, transport = daemon()
    robot = daemon.remote(transport.port)
    robot.connect()
    robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
    robot.step(ACTION)

    port = transport.port
    transport.stop(grace_s=0.1)
    service.close()  # a dead daemon releases its device claim
    with pytest.raises(TransportError) as info:
        robot.step(ACTION)
    assert "reconnect" in info.value.hint and "never retried" in info.value.hint

    robot.disconnect()  # the dead daemon cannot answer robot.disconnect; logged, not raised

    _, new_transport = daemon(port=port)
    assert new_transport.port == port
    robot.connect()
    assert robot.get_observation()["arm"]["joint_pos"].shape == (7,)
    robot.arm.set_control_mode(ControlMode.JOINT_SERVO)  # the fresh daemon robot starts IDLE
    robot.step(ACTION)


def test_token_mismatch(daemon):
    _, transport = daemon(token="s3cret")
    with pytest.raises(TransportError, match="token"):
        daemon.remote(transport.port).connect()
    robot = daemon.remote(transport.port, token="s3cret")
    robot.connect()
    assert robot.is_connected


def test_fault_with_hint_over_the_wire(daemon):
    service, transport = daemon()
    robot = daemon.remote(transport.port)
    robot.connect()
    robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
    service.robot.arm.inject("fault")
    with pytest.raises(DeviceFault) as info:
        robot.arm.servo_joints(np.ones(7))
    assert info.value.recoverable
    assert "recover()" in info.value.hint
    robot.arm.recover()
    robot.arm.servo_joints(np.ones(7))


def test_device_claim_over_the_wire(daemon):
    _, first_transport = daemon()
    _, second_transport = daemon()
    first = daemon.remote(first_transport.port)
    second = daemon.remote(second_transport.port)
    first.connect()
    with pytest.raises(DeviceBusy) as info:
        second.connect()
    assert info.value.hint
    first.disconnect()
    second.connect()
    assert second.is_connected


def test_teleop_over_the_wire(daemon):
    from robot_infra.devices import TeleopConfig, TeleopSample

    script = [
        {"action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0], "active": True, "buttons": {"right": True}},
        {"events": {"episode_end": True}},
    ]
    teleop_robot = FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake"),
            teleop=TeleopConfig(backend="fake", options={"script": script}),
        )
    )
    _, transport = daemon(watchdog_s=30.0, robot=teleop_robot)
    robot = daemon.remote(transport.port)
    robot.connect()
    teleop = robot.teleop
    assert teleop.action_space == "cartesian_delta" and teleop.has_gripper is True
    sample = teleop.read(timeout_s=1.0)
    assert isinstance(sample, TeleopSample)
    assert sample.action.dtype == np.float32 and sample.action.shape == (6,)
    np.testing.assert_allclose(sample.action, [0.1, 0.2, 0.3, 0, 0, 0], atol=1e-6)
    assert sample.intervening is True and sample.buttons == {"right": True}
    assert isinstance(sample.gripper, float) and isinstance(sample.seq, int)
    ended = teleop.read()
    assert ended.events["episode_end"] is True and ended.intervening is False
    assert teleop.read().events["episode_end"] is False
    assert teleop.health().ok
    robot.disconnect()


def test_no_teleop_over_the_wire(daemon):
    _, transport = daemon()
    robot = daemon.remote(transport.port)
    robot.connect()
    assert robot.teleop is None
    robot.disconnect()

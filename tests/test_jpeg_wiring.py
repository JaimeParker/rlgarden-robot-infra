"""--jpeg-quality wiring: service encodes camera color on every camera path; clients restore it."""
from __future__ import annotations

import sys
import time

import numpy as np
import pytest

import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, Frame, GripperConfig
from robot_infra.devices.backends.cameras.fake import FakeCamera
from robot_infra.devices.base import Health
from robot_infra.devices.errors import SdkMissing
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.server import RemoteRobot
from robot_infra.server.codec import pack, unpack
from robot_infra.server.remote import _rebuild
from robot_infra.server.service import RobotService
from robot_infra.server.transport import LocalClient

pytest.importorskip("PIL")

HEIGHT, WIDTH = 96, 128
ACTION = {"arm": {"joint_pos": np.zeros(7)}, "gripper": {"gripper": [0.5]}}


def textured_frame() -> np.ndarray:
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    gradient = np.stack([xs, ys * 2, xs + ys], axis=-1)
    noise = np.random.default_rng(0).integers(0, 8, size=gradient.shape)
    return (gradient + noise).astype(np.uint8)


FRAME = textured_frame()


@pytest.fixture(autouse=True)
def fixed_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")

    def capture(self, handle):
        time.sleep(1.0 / self._config.fps)
        depth = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32) if self._config.enable_depth else None
        return Frame(color=FRAME.copy(), depth=depth, timestamp=123.0)

    monkeypatch.setattr(FakeCamera, "_capture_frame", capture)


def make_robot(enable_depth: bool = False) -> FrankaRobot:
    return FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake"),
            cameras=[
                CameraConfig(
                    name="wrist", backend="fake", fps=60, width=WIDTH, height=HEIGHT, enable_depth=enable_depth
                )
            ],
        )
    )


@pytest.fixture
def make_service():
    services = []

    def build(jpeg_quality=None, enable_depth=False) -> RobotService:
        service = RobotService(make_robot(enable_depth), watchdog_s=30.0, jpeg_quality=jpeg_quality)
        services.append(service)
        service.dispatch("robot.connect")
        return service

    yield build
    for service in services:
        service.close()


def wire_roundtrip(reply):
    wire = pack(reply)
    return wire, unpack(wire)


def test_raw_when_quality_none(make_service):
    service = make_service()
    _, reply = wire_roundtrip(service.dispatch("camera.wrist.read", {"timeout_s": 2.0}))
    assert np.array_equal(reply["color"], FRAME)


def test_jpeg_camera_read_is_lossy_and_small(make_service):
    raw_size = len(pack({"color": FRAME}))
    service = make_service(jpeg_quality=90)
    reply = service.dispatch("camera.wrist.read", {"timeout_s": 2.0})
    assert reply["color"]["jpeg"] is True
    wire, decoded = wire_roundtrip(reply)
    assert decoded["color"].dtype == np.uint8 and decoded["color"].shape == FRAME.shape
    diff = np.abs(decoded["color"].astype(int) - FRAME.astype(int))
    assert 0 < diff.max() <= 40
    assert len(wire) < raw_size / 3
    assert decoded["timestamp"] == 123.0


@pytest.mark.parametrize("method", ["robot.get_observation", "robot.step"])
def test_observation_paths_encode_color_only(make_service, method):
    service = make_service(jpeg_quality=90, enable_depth=True)
    kwargs = None
    if method == "robot.step":
        service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
        kwargs = {"action": ACTION}
    reply = service.dispatch(method, kwargs)
    assert reply["cameras"]["wrist"]["color"]["jpeg"] is True
    _, decoded = wire_roundtrip(reply)
    camera = decoded["cameras"]["wrist"]
    assert camera["color"].dtype == np.uint8 and camera["color"].shape == FRAME.shape
    assert not np.array_equal(camera["color"], FRAME)
    assert camera["depth"].dtype == np.float32
    assert np.array_equal(camera["depth"], np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32))
    assert decoded["arm"].keys() == service.dispatch("robot.get_observation")["arm"].keys()


def test_camera_read_keeps_depth_and_timestamp(make_service):
    service = make_service(jpeg_quality=90, enable_depth=True)
    _, reply = wire_roundtrip(service.dispatch("camera.wrist.read", {"timeout_s": 2.0}))
    assert reply["depth"].dtype == np.float32 and np.all(reply["depth"] == 0.5)
    assert reply["timestamp"] == 123.0


@pytest.mark.parametrize("quality, expected", [(None, None), (90, {"jpeg": 90})])
def test_describe_and_remote_camera_report_encoding(make_service, quality, expected):
    service = make_service(jpeg_quality=quality)
    description = service.dispatch("robot.describe")
    assert description["image_encoding"] == expected
    # An older client unpacks each camera dict as CameraConfig(**dict): no extra keys allowed.
    assert CameraConfig(**description["cameras"]["wrist"]).name == "wrist"
    remote = RemoteRobot.from_client(LocalClient(service))
    remote.connect()
    try:
        assert remote.cameras["wrist"].image_encoding == expected
        color = remote.cameras["wrist"].read().color
        assert color.dtype == np.uint8 and color.shape == FRAME.shape
        assert np.array_equal(color, FRAME) == (quality is None)
        assert remote.get_observation()["cameras"]["wrist"]["color"].shape == FRAME.shape
    finally:
        remote.disconnect()


def test_grpc_loopback_decodes_jpeg(make_service):
    pytest.importorskip("grpc")
    from robot_infra.devices import Robot
    from robot_infra.server.transport import GrpcTransport

    service = make_service(jpeg_quality=90)
    transport = GrpcTransport()
    transport.start(service, "127.0.0.1:0")
    remote = Robot.from_dict({"type": "remote", "address": f"127.0.0.1:{transport.port}", "timeout_s": 5.0})
    try:
        remote.connect()
        assert remote.cameras["wrist"].image_encoding == {"jpeg": 90}
        color = remote.cameras["wrist"].read().color
        assert color.dtype == np.uint8 and color.shape == FRAME.shape
        observed = remote.get_observation()["cameras"]["wrist"]["color"]
        assert observed.dtype == np.uint8 and observed.shape == FRAME.shape
    finally:
        remote.disconnect()
        transport.stop(grace_s=0.1)


@pytest.mark.parametrize("quality", [0, 101, -1])
def test_invalid_quality_rejected(quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        RobotService(make_robot(), jpeg_quality=quality)


def test_missing_encoder_libraries_fail_at_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(SdkMissing) as info:
        RobotService(make_robot(), jpeg_quality=90)
    assert "[jpeg]" in info.value.hint
    RobotService(make_robot()).close()  # raw mode needs no encoder


def test_unknown_health_keys_are_ignored(make_service):
    assert _rebuild(Health, {"ok": True, "detail": "x", "future_field": 1}) == Health(ok=True, detail="x")
    service = make_service()
    original = service._methods["arm.health"]
    service._methods["arm.health"] = lambda: {**original(), "future_field": 1}
    remote = RemoteRobot.from_client(LocalClient(service))
    remote.connect()
    try:
        assert remote.arm.health().ok
    finally:
        remote.disconnect()


def block_jpeg_libraries(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("PIL", "cv2"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_client_without_jpeg_libraries_raises_sdk_missing_on_unpack(make_service, monkeypatch):
    wire = pack(make_service(jpeg_quality=90).dispatch("camera.wrist.read", {"timeout_s": 2.0}))
    block_jpeg_libraries(monkeypatch)
    with pytest.raises(SdkMissing) as info:
        unpack(wire)
    assert "[jpeg]" in info.value.hint


def test_client_without_jpeg_libraries_fails_at_connect(make_service, monkeypatch):
    service = make_service(jpeg_quality=90)
    block_jpeg_libraries(monkeypatch)
    remote = RemoteRobot.from_client(LocalClient(service))
    with pytest.raises(SdkMissing) as info:
        remote.connect()
    assert "[jpeg]" in info.value.hint

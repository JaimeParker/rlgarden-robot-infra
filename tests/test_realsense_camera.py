"""Hardware-free tests for RealSenseCamera against an injected fake pyrealsense2."""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_infra.devices import Camera, CameraConfig, DeviceFault
from robot_infra.devices.backends.cameras.realsense import RealSenseCamera
from robot_infra.devices.errors import DeviceNotFound

WIDTH, HEIGHT, FPS = 8, 6, 60
DEPTH_SCALE = 0.001


class FakeFrames:
    def get_color_frame(self):
        return SimpleNamespace(get_data=lambda: np.full((HEIGHT, WIDTH, 3), 7, np.uint8))

    def get_depth_frame(self):
        return SimpleNamespace(get_data=lambda: np.full((HEIGHT, WIDTH), 1500, np.uint16))


class FakePipeline:
    def __init__(self, state):
        self.state = state

    def start(self, cfg):
        if self.state.start_error is not None:
            raise self.state.start_error
        self.state.started_with = cfg
        sensor = SimpleNamespace(get_depth_scale=lambda: DEPTH_SCALE)
        device = SimpleNamespace(first_depth_sensor=lambda: sensor)
        return SimpleNamespace(get_device=lambda: device)

    def stop(self):
        self.state.stopped = True

    def wait_for_frames(self, timeout_ms=0):
        time.sleep(0.002)
        return FakeFrames()


class FakeConfig:
    def __init__(self, state):
        self.state = state

    def enable_device(self, serial):
        self.state.enabled_device = serial

    def enable_stream(self, *args):
        self.state.streams.append(args)


def make_fake_rs(serials, start_error=None):
    state = SimpleNamespace(
        start_error=start_error, started_with=None, stopped=False,
        enabled_device=None, streams=[],
    )
    devices = [
        SimpleNamespace(get_info=lambda key, s=s: s if key == "SERIAL_KEY" else "")
        for s in serials
    ]
    rs = SimpleNamespace(
        context=lambda: SimpleNamespace(query_devices=lambda: devices),
        camera_info=SimpleNamespace(serial_number="SERIAL_KEY"),
        pipeline=lambda: FakePipeline(state),
        config=lambda: FakeConfig(state),
        stream=SimpleNamespace(color="COLOR", depth="DEPTH"),
        format=SimpleNamespace(rgb8="RGB8", z16="Z16"),
        align=lambda target: SimpleNamespace(process=lambda frames: frames),
    )
    return rs, state


@pytest.fixture
def install_rs(monkeypatch):
    def install(serials, start_error=None):
        rs, state = make_fake_rs(serials, start_error)
        monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
        return state

    return install


def config(**kwargs) -> CameraConfig:
    return CameraConfig(name="cam", width=WIDTH, height=HEIGHT, fps=FPS, **kwargs)


def test_discover_returns_serials(install_rs):
    install_rs(["A1", "B2"])
    assert RealSenseCamera.discover() == {"A1", "B2"}


def test_discover_without_sdk_is_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    assert RealSenseCamera.discover() == set()


def test_open_by_serial_color_only(install_rs):
    state = install_rs(["A1", "B2"])
    camera = RealSenseCamera(config(serial="B2"))
    camera.connect()
    try:
        assert state.enabled_device == "B2"
        assert state.streams == [("COLOR", WIDTH, HEIGHT, "RGB8", FPS)]
    finally:
        camera.disconnect()


def test_open_with_depth_enables_depth_stream(install_rs):
    state = install_rs(["A1"])
    camera = RealSenseCamera(config(serial="A1", enable_depth=True))
    camera.connect()
    try:
        assert state.streams == [
            ("COLOR", WIDTH, HEIGHT, "RGB8", FPS),
            ("DEPTH", WIDTH, HEIGHT, "Z16", FPS),
        ]
    finally:
        camera.disconnect()


def test_empty_serial_single_device_is_used(install_rs):
    state = install_rs(["ONLY"])
    camera = RealSenseCamera(config())
    camera.connect()
    try:
        assert state.enabled_device == "ONLY"
    finally:
        camera.disconnect()


def test_empty_serial_two_devices_raises_with_both_serials(install_rs):
    install_rs(["A1", "B2"])
    with pytest.raises(DeviceNotFound) as info:
        RealSenseCamera(config()).connect()
    assert "A1" in info.value.hint and "B2" in info.value.hint


def test_empty_serial_no_devices_points_to_discover(install_rs):
    install_rs([])
    with pytest.raises(DeviceNotFound) as info:
        RealSenseCamera(config()).connect()
    assert "robot_infra.discover" in info.value.hint


def test_serial_not_attached_lists_discovered(install_rs):
    install_rs(["A1", "B2"])
    with pytest.raises(DeviceNotFound) as info:
        RealSenseCamera(config(serial="ZZ")).connect()
    assert "A1" in info.value.hint and "B2" in info.value.hint


def test_start_failure_is_recoverable_fault_with_udev_hint(install_rs):
    install_rs(["A1"], start_error=RuntimeError("no frames"))
    camera = RealSenseCamera(config(serial="A1"))
    with pytest.raises(DeviceFault) as info:
        camera.connect()
    assert info.value.recoverable is True
    assert "udev" in info.value.hint
    assert not camera.is_connected


def test_read_color_and_scaled_depth_then_disconnect_stops_pipeline(install_rs):
    state = install_rs(["A1"])
    camera = RealSenseCamera(config(serial="A1", enable_depth=True))
    camera.connect()
    try:
        frame = camera.read()
        assert frame.color.shape == (HEIGHT, WIDTH, 3)
        assert frame.color.dtype == np.uint8
        assert frame.depth.dtype == np.float32
        expected = np.full((HEIGHT, WIDTH), 1500, np.uint16).astype(np.float32) * DEPTH_SCALE
        np.testing.assert_allclose(frame.depth, expected)
    finally:
        camera.disconnect()
    assert state.stopped


def test_read_color_only_has_no_depth(install_rs):
    install_rs(["A1"])
    camera = RealSenseCamera(config(serial="A1"))
    camera.connect()
    try:
        assert camera.read().depth is None
    finally:
        camera.disconnect()


def test_registry_and_sdk_name():
    assert Camera.backend("realsense") is RealSenseCamera
    assert RealSenseCamera.SDK == "pyrealsense2"

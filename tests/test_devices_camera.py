"""Hardware-free tests for the Camera contract via FakeCamera."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from robot_infra.devices import Camera, CameraConfig, DeviceFault
from robot_infra.devices.backends.cameras.fake import FakeCamera
from robot_infra.devices.camera import Frame


def config(**kwargs) -> CameraConfig:
    return CameraConfig(name="cam", width=8, height=6, fps=60, backend="fake", **kwargs)


def counter_of(frame: Frame) -> int:
    return int(frame.color[0, 0, 0]) + 256 * int(frame.color[0, 0, 1])


@pytest.fixture
def camera():
    camera = FakeCamera(config())
    camera.connect()
    yield camera
    camera.disconnect()


def test_read_returns_fresh_increasing_frames(camera):
    counters = [counter_of(camera.read()) for _ in range(5)]
    assert counters == sorted(set(counters))


def test_is_ready_and_latest(camera):
    assert camera.is_ready()
    assert camera.latest() is not None


def test_observation_features_depth_only_when_enabled():
    assert set(FakeCamera(config()).observation_features) == {"color"}
    features = FakeCamera(config(enable_depth=True)).observation_features
    assert features["depth"] == {"shape": (6, 8), "dtype": "float32"}
    assert features["color"] == {"shape": (6, 8, 3), "dtype": "uint8"}


def test_get_observation_with_depth():
    camera = FakeCamera(config(enable_depth=True))
    camera.connect()
    try:
        observation = camera.get_observation()
        assert observation["color"].shape == (6, 8, 3)
        assert observation["color"].dtype == np.uint8
        assert observation["depth"].dtype == np.float32
        assert observation["depth"].shape == (6, 8)
    finally:
        camera.disconnect()


def test_get_observation_color_only(camera):
    assert set(camera.get_observation()) == {"color"}


def test_reopen_after_capture_failures():
    class CountingCamera(FakeCamera):
        open_calls = 0

        def _open_device(self):
            type(self).open_calls += 1
            return super()._open_device()

    camera = CountingCamera(
        config(), fail_after=2, read_timeout_s=0.3, max_consecutive_failures=2
    )
    camera.connect()
    try:
        camera.read()
        camera.read()
        frame = camera.read()
        assert counter_of(frame) <= 2
        assert CountingCamera.open_calls == 2
        assert camera.is_connected
    finally:
        camera.disconnect()


def test_hung_capture_raises_recoverable_fault():
    release = threading.Event()

    class HungCamera(FakeCamera):
        def _capture_frame(self, handle):
            release.wait()
            raise RuntimeError("released")

    camera = HungCamera(config(), read_timeout_s=0.2)
    camera.connect()
    try:
        with pytest.raises(DeviceFault) as info:
            camera.read()
        assert info.value.recoverable is True
        assert "doctor" in info.value.hint
    finally:
        release.set()
        camera.disconnect()


def test_disconnect_joins_capture_thread():
    camera = FakeCamera(config())
    camera.connect()
    thread = camera._capture_thread
    assert thread.is_alive()
    camera.disconnect()
    assert not thread.is_alive()


def test_health_reports_fresh_frames(camera):
    camera.read()
    assert camera.health().ok


def test_registry_and_discover():
    assert Camera.backend("fake") is FakeCamera
    assert FakeCamera.discover() == {"FAKE0", "FAKE1"}

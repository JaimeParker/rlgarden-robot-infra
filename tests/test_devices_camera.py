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


def test_reopen_after_disconnect_stays_closed():
    camera = FakeCamera(config())
    camera.connect()
    thread = camera._capture_thread
    camera.disconnect()
    camera.reopen()
    assert not camera.is_connected
    assert camera._capture_thread is None
    assert not thread.is_alive()


def test_disconnect_during_reopen_leaves_no_live_device():
    closed = []

    class RacingCamera(FakeCamera):
        opens = 0

        def _open_device(self):
            type(self).opens += 1
            handle = {"open": type(self).opens}
            if type(self).opens == 2:  # inside reopen(): another thread disconnects now
                racer = threading.Thread(target=self.disconnect)
                racer.start()
                racer.join(timeout=0.2)  # blocks on the lifecycle lock until reopen returns
                self.racer = racer
            return handle

        def _close_device(self, handle):
            closed.append(handle["open"])

    camera = RacingCamera(config())
    camera.connect()
    camera.reopen()
    camera.racer.join(timeout=2.0)
    assert not camera.racer.is_alive()
    assert not camera.is_connected
    assert camera._capture_thread is None
    assert sorted(closed) == [1, 2]


def test_concurrent_read_timeouts_reopen_once():
    class CountingCamera(FakeCamera):
        opens = 0

        def _open_device(self):
            type(self).opens += 1
            return super()._open_device()

    camera = CountingCamera(config())
    camera.connect()
    try:
        session = camera._handle
        camera._reopen_session(session)
        camera._reopen_session(session)  # a second reader that timed out on the same session
        assert CountingCamera.opens == 2
    finally:
        camera.disconnect()


def test_failed_reopen_leaves_camera_disconnected():
    class FlakyCamera(FakeCamera):
        opens = 0

        def _open_device(self):
            type(self).opens += 1
            if type(self).opens == 2:
                raise RuntimeError("usb gone")
            return super()._open_device()

    camera = FlakyCamera(config(), fail_after=0, read_timeout_s=0.1, max_consecutive_failures=1)
    camera.connect()
    with pytest.raises(RuntimeError, match="usb gone"):
        camera.read()
    assert not camera.is_connected
    assert camera._capture_thread is None

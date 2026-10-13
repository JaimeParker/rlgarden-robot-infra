"""SpaceMouse teleop backend against a mocked ``pyspacemouse`` module (no hardware)."""
from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from robot_infra.devices import DeviceNotFound, SdkMissing, Teleop, TeleopConfig, VersionMismatch
from robot_infra.devices.backends.teleops import spacemouse as spacemouse_backend
from robot_infra.devices.backends.teleops.spacemouse import SpaceMouseTeleop
from robot_infra.discovery import enumerate as probes
from robot_infra.doctor import checks


def make_pyspacemouse() -> types.ModuleType:
    fake = types.ModuleType("pyspacemouse")
    fake.twist = (0.0,) * 6
    fake.buttons = (0, 0)
    fake.devices = []

    class Device:
        def __init__(self):
            self.closed = False

        def read(self):
            if self.closed:
                raise OSError("read from a closed SpaceMouse")
            time.sleep(0.001)
            x, y, z, roll, pitch, yaw = fake.twist
            return types.SimpleNamespace(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, buttons=list(fake.buttons)
            )

        def close(self):
            self.closed = True

    def open_device(**kwargs):
        device = Device()
        device.kwargs = kwargs
        fake.devices.append(device)
        return device

    fake.open = open_device
    return fake


@pytest.fixture
def pyspacemouse(monkeypatch):
    fake = make_pyspacemouse()
    monkeypatch.setitem(sys.modules, "pyspacemouse", fake)
    return fake


@pytest.fixture
def spacemouse(pyspacemouse):
    created = []

    def build(**options):
        teleop = Teleop.backend("spacemouse")(TeleopConfig(backend="spacemouse", options=options))
        teleop.connect()
        created.append(teleop)
        return teleop

    yield build
    for teleop in created:
        teleop.disconnect()


def settled_read(teleop, predicate, attempts=200):
    """Read until the mocked state has propagated through the poll thread."""
    for _ in range(attempts):
        sample = teleop.read()
        if predicate(sample):
            return sample
    raise AssertionError("mocked state never reached the sample")


def test_registered_and_contract():
    assert Teleop.backend("spacemouse") is SpaceMouseTeleop
    assert SpaceMouseTeleop.ACTION_SPACE == "cartesian_delta" and SpaceMouseTeleop.HAS_GRIPPER


def test_missing_sdk_raises_sdk_missing_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyspacemouse", None)  # makes `import pyspacemouse` fail
    teleop = SpaceMouseTeleop(TeleopConfig(backend="spacemouse"))
    with pytest.raises(SdkMissing) as info:
        teleop.connect()
    assert "pip install" in info.value.hint and "pyspacemouse" in info.value.hint
    assert not teleop.is_connected


def test_legacy_api_without_open_is_version_mismatch(monkeypatch):
    legacy = types.ModuleType("pyspacemouse")
    legacy.read_all = lambda: []
    monkeypatch.setitem(sys.modules, "pyspacemouse", legacy)
    with pytest.raises(VersionMismatch) as info:
        SpaceMouseTeleop(TeleopConfig(backend="spacemouse")).connect()
    assert "pyspacemouse>=2.0,<3" in info.value.hint


def test_legacy_open_returning_bool_is_version_mismatch(monkeypatch):
    legacy = types.ModuleType("pyspacemouse")
    legacy.open = lambda **kwargs: True
    legacy.read_all = lambda: []
    monkeypatch.setitem(sys.modules, "pyspacemouse", legacy)
    with pytest.raises(VersionMismatch):
        SpaceMouseTeleop(TeleopConfig(backend="spacemouse")).connect()


def test_open_failure_is_device_not_found(pyspacemouse):
    pyspacemouse.open = lambda **kwargs: None
    with pytest.raises(DeviceNotFound):
        SpaceMouseTeleop(TeleopConfig(backend="spacemouse")).connect()


def test_device_index_is_forwarded(spacemouse, pyspacemouse):
    spacemouse(device_index=2)
    assert pyspacemouse.devices[0].kwargs == {"device_index": 2}


def test_axis_mapping_signs_and_order(spacemouse, pyspacemouse):
    teleop = spacemouse()
    pyspacemouse.twist = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)  # x, y, z, roll, pitch, yaw
    sample = settled_read(teleop, lambda s: s.action[0] != 0)
    np.testing.assert_allclose(sample.action, [-0.2, 0.1, 0.3, -0.4, -0.5, -0.6], atol=1e-6)
    assert sample.action.dtype == np.float32 and sample.intervening


def test_invert_axes_flips_selected_outputs(spacemouse, pyspacemouse):
    teleop = spacemouse(invert_axes=[0, 5])
    pyspacemouse.twist = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    sample = settled_read(teleop, lambda s: s.action[0] != 0)
    np.testing.assert_allclose(sample.action, [0.2, 0.1, 0.3, -0.4, -0.5, 0.6], atol=1e-6)


def test_invalid_options_rejected():
    with pytest.raises(ValueError, match="unknown spacemouse option"):
        SpaceMouseTeleop(TeleopConfig(backend="spacemouse", options={"bogus": 1}))
    with pytest.raises(ValueError, match="invert_axes"):
        SpaceMouseTeleop(TeleopConfig(backend="spacemouse", options={"invert_axes": [6]}))


def test_below_deadzone_is_not_intervening_and_action_stays_raw(spacemouse, pyspacemouse):
    teleop = spacemouse(deadzone=0.05)
    pyspacemouse.twist = (0.0, 0.0, 0.01, 0.0, 0.0, 0.0)
    sample = settled_read(teleop, lambda s: s.action[2] != 0)
    assert not sample.intervening
    assert sample.action[2] == pytest.approx(0.01)  # legacy semantics: raw twist, flag only
    pyspacemouse.twist = (0.0, 0.0, 0.2, 0.0, 0.0, 0.0)
    assert settled_read(teleop, lambda s: s.action[2] > 0.1).intervening


def test_idle_sample_has_zero_action_and_open_gripper(spacemouse):
    sample = spacemouse().read()
    assert not sample.intervening and not sample.action.any() and sample.gripper == 1.0


def press(pyspacemouse, teleop, buttons):
    pyspacemouse.buttons = buttons
    return settled_read(teleop, lambda s: [s.buttons["button_0"], s.buttons["button_1"]] == [bool(b) for b in buttons])


def test_button_zero_toggles_sticky_gripper_and_reset_reopens(spacemouse, pyspacemouse):
    teleop = spacemouse()
    sample = press(pyspacemouse, teleop, (1, 0))
    assert sample.intervening  # a pressed button alone counts as activity
    assert settled_read(teleop, lambda s: s.gripper == 0.0).gripper == 0.0
    press(pyspacemouse, teleop, (0, 0))
    assert teleop.read().gripper == 0.0  # sticky after release
    press(pyspacemouse, teleop, (1, 0))
    assert settled_read(teleop, lambda s: s.gripper == 1.0).gripper == 1.0
    press(pyspacemouse, teleop, (0, 0))
    press(pyspacemouse, teleop, (1, 0))
    settled_read(teleop, lambda s: s.gripper == 0.0)
    press(pyspacemouse, teleop, (0, 0))
    teleop.reset()
    assert settled_read(teleop, lambda s: not s.buttons["button_0"] and s.gripper == 1.0).gripper == 1.0


def test_held_button_toggles_only_once(spacemouse, pyspacemouse):
    teleop = spacemouse()
    press(pyspacemouse, teleop, (1, 0))
    time.sleep(0.05)  # many polls with the button still held
    assert teleop.read().gripper == 0.0


def test_button_one_latches_episode_end_once(spacemouse, pyspacemouse):
    teleop = spacemouse()
    pyspacemouse.buttons = (0, 1)
    seen = []
    for _ in range(200):
        sample = teleop.read()
        seen.append(sample.events["episode_end"])
        if sample.buttons["button_1"] and len(seen) > 1 and any(seen):
            break
    pyspacemouse.buttons = (0, 0)
    time.sleep(0.05)
    assert seen.count(True) == 1  # the held button fires one edge
    assert teleop.read().events["episode_end"] is False
    assert teleop.read().events["episode_end"] is False


def test_disconnect_closes_device_and_stops_thread(spacemouse, pyspacemouse):
    teleop = spacemouse()
    teleop.read()
    device = pyspacemouse.devices[0]
    assert any(t.name == f"teleop-{teleop.name}" for t in threading.enumerate())
    teleop.disconnect()
    assert device.closed
    assert not any(t.name == f"teleop-{teleop.name}" for t in threading.enumerate())


def test_reconnect_opens_a_fresh_device(spacemouse, pyspacemouse):
    teleop = spacemouse()
    teleop.disconnect()
    teleop.connect()
    assert len(pyspacemouse.devices) == 2 and pyspacemouse.devices[0].closed
    assert not pyspacemouse.devices[1].closed
    teleop.disconnect()


# -- discovery ---------------------------------------------------------------


def fake_hid(entries):
    module = types.ModuleType("hid")
    module.enumerate = lambda: entries
    return module


def test_discover_filters_vendor_ids_via_hidapi(monkeypatch):
    entries = [
        {"vendor_id": 0x256F, "product_id": 0xC635, "path": b"/dev/hidraw3"},
        {"vendor_id": 0x046D, "product_id": 0xC626, "path": b"/dev/hidraw4"},
        {"vendor_id": 0x1234, "product_id": 0x0001, "path": b"/dev/hidraw5"},
    ]
    monkeypatch.setitem(sys.modules, "hid", fake_hid(entries))
    assert SpaceMouseTeleop.discover() == {"256f:c635:/dev/hidraw3", "046d:c626:/dev/hidraw4"}


def test_discover_falls_back_to_hidraw_nodes_without_hidapi(monkeypatch, tmp_path):
    for name in ("hidraw0", "hidraw1"):
        (tmp_path / name).touch()
    monkeypatch.setitem(sys.modules, "hid", None)
    monkeypatch.setattr(spacemouse_backend, "HIDRAW_GLOB", str(tmp_path / "hidraw*"))
    assert SpaceMouseTeleop.discover() == {str(tmp_path / "hidraw0"), str(tmp_path / "hidraw1")}
    for path in tmp_path.iterdir():
        path.unlink()
    assert SpaceMouseTeleop.discover() == set()


def test_discover_teleops_includes_spacemouse(monkeypatch):
    monkeypatch.setitem(sys.modules, "hid", fake_hid([]))
    assert probes.discover_teleops(hardware_only=True)["spacemouse"] == set()


# -- doctor os:hid -----------------------------------------------------------


def run_hid(pattern):
    return checks._make_check(2, "os:hid", checks._hid_outcome(pattern))()


def test_hid_check_ok_when_nodes_are_readable_and_writable(tmp_path):
    (tmp_path / "hidraw0").touch()
    result = run_hid(str(tmp_path / "hidraw*"))
    assert result.ok and "hidraw0" in result.message


def test_hid_check_reports_udev_hint_when_access_denied(tmp_path, monkeypatch):
    (tmp_path / "hidraw0").touch()
    monkeypatch.setattr(checks.os, "access", lambda path, mode: False)
    result = run_hid(str(tmp_path / "hidraw*"))
    assert not result.ok and "hidraw0" in result.message
    assert 'ATTRS{idVendor}=="256f"' in result.hint and "udevadm control --reload" in result.hint
    assert "--device=/dev/hidrawN" in result.hint


def test_hid_check_reports_missing_devices(tmp_path):
    result = run_hid(str(tmp_path / "hidraw*"))
    assert not result.ok and "no HID device" in result.message and "USB" in result.hint


def test_os_hid_check_only_for_spacemouse_backend(monkeypatch):
    monkeypatch.setattr(checks, "_HIDRAW_GLOB", "/nonexistent/hidraw*")
    from robot_infra.devices.robots.piper import PiperConfig

    def names(backend):
        config = PiperConfig(arm_backend="fake", teleop=TeleopConfig(backend=backend))
        return {name for name, _ in checks._os_checks(config)}

    assert "os:hid" in names("spacemouse")
    assert "os:hid" not in names("fake")


def test_poll_interval_sleeps_once_per_poll(spacemouse, monkeypatch):
    sleeps = []
    monkeypatch.setattr(spacemouse_backend, "_sleep", lambda seconds: sleeps.append(seconds))
    teleop = spacemouse(poll_interval_s=0.02)
    for _ in range(3):
        teleop.read()
    teleop.disconnect()
    assert len(sleeps) >= 3 and set(sleeps) == {0.02}


def test_poll_interval_defaults_to_200hz_and_rejects_negative():
    assert SpaceMouseTeleop(TeleopConfig(backend="spacemouse"))._poll_interval_s == 0.005
    with pytest.raises(ValueError, match="poll_interval_s"):
        SpaceMouseTeleop(TeleopConfig(backend="spacemouse", options={"poll_interval_s": -1}))

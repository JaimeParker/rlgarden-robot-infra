"""Hardware-free tests for the device layer: lifecycle, owner borrowing, PartGroup,
backend registry, claims, and error round trips."""
from __future__ import annotations

import numpy as np
import pytest

from robot_infra.devices import (
    BackendRegistry,
    Connection,
    ControllablePart,
    DeviceBusy,
    DeviceClaim,
    DeviceError,
    DeviceFault,
    DeviceNotConnected,
    Part,
    PartGroup,
    SdkMissing,
    VersionMismatch,
)


class FakeConnection(Connection):
    def __init__(self, name, *, owner=None, log=None, open_error=None):
        super().__init__(name, owner=owner)
        self.log = log if log is not None else []
        self.open_error = open_error
        self.open_calls = 0
        self.release_calls = 0
        self.opened_calls = 0
        self.closing_calls = 0

    def _open(self):
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error
        self.log.append(f"open:{self.name}")
        return object()

    def _release(self, handle):
        self.release_calls += 1
        self.log.append(f"release:{self.name}")

    def _opened(self):
        self.opened_calls += 1

    def _closing(self):
        self.closing_calls += 1


class FakePart(FakeConnection, Part):
    @property
    def observation_features(self):
        return {"x": {"shape": (1,), "dtype": "float32"}}

    def get_observation(self):
        return {"x": np.zeros(1, dtype=np.float32)}


class FakeControllablePart(FakePart, ControllablePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = None

    @property
    def action_features(self):
        return {"cmd": {"shape": (2,), "dtype": "float32"}}

    def send_action(self, action):
        self.received = action


def test_connect_and_disconnect_are_idempotent():
    conn = FakeConnection("c")
    conn.connect()
    conn.connect()
    assert conn.is_connected and conn.open_calls == 1
    conn.disconnect()
    conn.disconnect()
    assert not conn.is_connected and conn.release_calls == 1


def test_failed_open_leaves_disconnected_and_propagates():
    conn = FakeConnection("c", open_error=OSError("boom"))
    with pytest.raises(OSError, match="boom"):
        conn.connect()
    assert not conn.is_connected


def test_borrower_connects_owner_but_never_disconnects_it():
    owner = FakeConnection("owner")
    borrower = FakeConnection("borrower", owner=owner)
    assert not borrower.is_connected
    borrower.connect()
    assert owner.is_connected and borrower.is_connected and borrower.opened_calls == 1
    borrower.disconnect()
    assert owner.is_connected and not borrower.is_connected
    assert borrower.closing_calls == 1 and owner.release_calls == 0
    owner.disconnect()
    borrower.connect()
    assert borrower.is_connected and owner.open_calls == 2


def test_borrower_mirrors_owner_disconnect():
    owner = FakeConnection("owner")
    borrower = FakeConnection("borrower", owner=owner)
    borrower.connect()
    owner.disconnect()
    assert not borrower.is_connected


def test_require_sdk_missing_raises_sdk_missing_with_pip_hint():
    class NeedsSdk(FakeConnection):
        SDK = "definitely_not_installed_pkg_xyz"

    with pytest.raises(SdkMissing) as info:
        NeedsSdk.require_sdk()
    assert "pip install definitely_not_installed_pkg_xyz" in info.value.hint


def test_require_connected_raises_with_connect_hint():
    conn = FakeConnection("c")
    with pytest.raises(DeviceNotConnected) as info:
        conn._require_connected()
    assert "connect()" in info.value.hint
    assert info.value.device == "c"


def test_part_group_rolls_back_connected_children_on_open_failure():
    first = FakePart("first")
    second = FakePart("second", open_error=RuntimeError("second failed"))
    group = PartGroup("g")
    group.add("first", first)
    group.add("second", second)
    with pytest.raises(RuntimeError, match="second failed"):
        group.connect()
    assert not first.is_connected and not group.is_connected
    assert first.release_calls == 1


def test_part_group_releases_in_reverse_insertion_order():
    log = []
    group = PartGroup("g")
    for name in ("a", "b", "c"):
        group.add(name, FakePart(name, log=log))
    group.connect()
    log.clear()
    group.disconnect()
    assert log == ["release:c", "release:b", "release:a"]


def test_part_group_brings_up_sibling_owner_for_borrower():
    owner = FakePart("owner")
    borrower = FakePart("borrower", owner=owner)
    group = PartGroup("g")
    group.add("owner", owner)
    group.add("borrower", borrower)
    group.connect()
    assert owner.is_connected and borrower.is_connected and borrower.opened_calls == 1
    group.disconnect()
    assert not owner.is_connected and not borrower.is_connected


def test_part_group_send_action_dispatch_and_unknown_key():
    ctrl = FakeControllablePart("arm")
    group = PartGroup("g")
    group.add("arm", ctrl)
    group.add("cam", FakePart("cam"))
    command = {"cmd": np.ones(2, dtype=np.float32)}
    group.send_action({"arm": command})
    assert ctrl.received is command
    with pytest.raises(ValueError, match=r"\['arm'\]"):
        group.send_action({"cam": command})


def test_part_group_feature_nesting_and_duplicate_and_unknown_part():
    group = PartGroup("g")
    group.add("arm", FakeControllablePart("arm"))
    group.add("cam", FakePart("cam"))
    assert group.observation_features == {
        "arm": {"x": {"shape": (1,), "dtype": "float32"}},
        "cam": {"x": {"shape": (1,), "dtype": "float32"}},
    }
    assert group.action_features == {"arm": {"cmd": {"shape": (2,), "dtype": "float32"}}}
    assert set(group.get_observation()) == {"arm", "cam"}
    with pytest.raises(ValueError):
        group.add("arm", FakePart("again"))
    with pytest.raises(KeyError, match="arm"):
        group.part("missing")
    assert group.describe()["children"]["arm"]["name"] == "arm"


def test_backend_registry_register_lookup_and_errors():
    class Root(BackendRegistry):
        _BACKENDS = {}

    class SubRoot(Root):
        pass

    @Root.register("Foo")
    class Foo:
        pass

    assert Root.backend("foo") is Foo
    assert SubRoot.backend("FOO") is Foo
    assert Root.backends() == {"foo": Foo}
    with pytest.raises(ValueError):
        Root.register("foo")(type("Other", (), {}))
    with pytest.raises(ValueError, match=r"registered: \['foo'\]"):
        Root.backend("bar")


def test_device_claim_is_exclusive_and_context_managed(tmp_path):
    first = DeviceClaim("/dev/ttyUSB0", lock_dir=tmp_path)
    second = DeviceClaim("/dev/ttyUSB0", lock_dir=tmp_path)
    first.acquire()
    with pytest.raises(DeviceBusy) as info:
        second.acquire()
    assert "claim=False" in info.value.hint
    first.release()
    first.release()
    second.acquire()
    second.release()
    with DeviceClaim("/dev/ttyUSB0", lock_dir=tmp_path):
        with pytest.raises(DeviceBusy):
            second.acquire()
    second.acquire()
    second.release()


def test_device_error_round_trip_preserves_subclass_and_fields():
    error = VersionMismatch("old firmware", hint="update it", device="arm")
    restored = DeviceError.from_dict(error.to_dict())
    assert type(restored) is VersionMismatch
    assert (restored.message, restored.hint, restored.device) == ("old firmware", "update it", "arm")
    assert "\n  hint: update it" in str(restored)
    assert str(DeviceError("plain")) == "plain"


def test_device_fault_round_trips_recoverable():
    assert DeviceError.from_dict(DeviceFault("f", recoverable=True).to_dict()).recoverable is True
    assert DeviceError.from_dict(DeviceFault("f").to_dict()).recoverable is False


def test_unknown_error_kind_becomes_plain_device_error():
    restored = DeviceError.from_dict({"kind": "nope", "message": "m", "hint": "", "device": ""})
    assert type(restored) is DeviceError and restored.message == "m"

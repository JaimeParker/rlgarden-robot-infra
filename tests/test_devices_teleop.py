"""Hardware-free tests for the Teleop contract via the scripted FakeTeleop."""
from __future__ import annotations

import numpy as np
import pytest

import robot_infra.devices.teleop as teleop_module
from robot_infra.devices import DeviceFault, DeviceNotConnected, Teleop, TeleopConfig, TeleopSample
from robot_infra.devices.backends.teleops.fake import FakeTeleop
from robot_infra.devices.teleop import EVENT_KEYS, TeleopHealth


def twist(x: float) -> list[float]:
    return [x, 0.0, 0.0, 0.0, 0.0, 0.0]


def make(**options) -> FakeTeleop:
    hold_s = options.pop("hold_s", 0.0)
    return FakeTeleop(TeleopConfig(backend="fake", hold_s=hold_s, options=options))


@pytest.fixture
def connected():
    created: list[FakeTeleop] = []

    def build(**options) -> FakeTeleop:
        teleop = make(**options)
        teleop.connect()
        created.append(teleop)
        return teleop

    yield build
    for teleop in created:
        teleop.disconnect()


def xs(teleop: FakeTeleop, reads: int) -> list[float]:
    return [float(teleop.read().action[0]) for _ in range(reads)]


def test_script_advances_one_step_per_read_and_holds_last(connected):
    teleop = connected(script=[{"action": twist(0.1)}, {"action": twist(0.2), "hold": 2}])
    assert xs(teleop, 5) == pytest.approx([0.1, 0.2, 0.2, 0.2, 0.2])


def test_on_end_idle_and_loop(connected):
    script = [{"action": twist(0.1), "active": True}, {"action": twist(0.2), "active": True}]
    idle = connected(script=script, on_end="idle")
    samples = [idle.read() for _ in range(4)]
    assert [float(s.action[0]) for s in samples] == pytest.approx([0.1, 0.2, 0.0, 0.0])
    assert [s.intervening for s in samples] == [True, True, False, False]
    loop = connected(script=script, on_end="loop")
    assert xs(loop, 5) == pytest.approx([0.1, 0.2, 0.1, 0.2, 0.1])


def test_push_injects_a_sample(connected):
    teleop = connected()
    teleop.push({"action": twist(0.7), "gripper": 0.25, "buttons": {"left": True}, "active": True})
    sample = teleop.read()
    assert isinstance(sample, TeleopSample)
    assert sample.action.dtype == np.float32 and float(sample.action[0]) == pytest.approx(0.7)
    assert sample.gripper == 0.25 and sample.buttons == {"left": True} and sample.intervening
    following = teleop.read()  # idle fake after the push; the gripper target is sticky
    assert not following.intervening and following.gripper == 0.25
    assert np.all(following.action == 0.0)


def test_sample_fields_and_gripper_stickiness(connected):
    teleop = connected(script=[{"gripper": 0.0}, {}, {"gripper": 0.6}])
    first, second, third = (teleop.read() for _ in range(3))
    assert first.action_space == "cartesian_delta" and first.action.shape == (6,)
    assert [first.gripper, second.gripper, third.gripper] == [0.0, 0.0, 0.6]
    assert set(EVENT_KEYS) <= set(first.events)


def test_intervening_maps_active_without_hold(connected, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(teleop_module, "_now", lambda: clock["t"])
    teleop = connected(script=[{"active": True}, {"active": False}], on_end="idle")
    assert teleop.read().intervening is True
    clock["t"] = 0.01
    assert teleop.read().intervening is False


def test_hold_s_keeps_intervening_after_activity(connected, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(teleop_module, "_now", lambda: clock["t"])
    teleop = connected(hold_s=0.5, script=[{"active": True}, {"active": False, "hold": 5}])
    assert teleop.read().intervening is True
    clock["t"] = 0.3
    assert teleop.read().intervening is True
    clock["t"] = 0.5
    assert teleop.latest().intervening is True
    clock["t"] = 0.9
    assert teleop.read().intervening is False
    assert teleop.latest().intervening is False


def test_reset_clears_hold_and_events(connected, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(teleop_module, "_now", lambda: clock["t"])
    teleop = connected(hold_s=10.0, script=[{"active": True}, {"events": {"success": True}}])
    assert teleop.read().intervening
    teleop.read()
    teleop.push({"events": {"failure": True}})
    assert teleop.latest().events["failure"] is True
    teleop.reset()
    sample = teleop.read()
    assert not sample.intervening
    assert not any(sample.events.values())


def test_events_latch_until_one_read_consumes_them(connected):
    teleop = connected(script=[{"events": {"episode_end": True}, "hold": 3}])
    first = teleop.read()
    assert teleop.latest().events["episode_end"] is False  # consumed by the read
    assert first.events["episode_end"] is True
    assert teleop.read().events["episode_end"] is False
    assert teleop.read().events["episode_end"] is False


def test_latest_does_not_consume_events(connected):
    teleop = connected()
    teleop.push({"events": {"rerecord": True}})
    assert teleop.latest().events["rerecord"] is True
    assert teleop.latest().events["rerecord"] is True
    assert teleop.get_observation()["action"].shape == (6,)
    assert teleop.read().events["rerecord"] is True
    assert teleop.read().events["rerecord"] is False


def test_short_press_between_reads_is_not_lost(connected):
    teleop = connected()
    teleop.push({"events": {"success": True}})
    teleop.push({})  # the press is over before the client reads
    sample = teleop.read()
    assert sample.events["success"] is True


def test_seq_is_monotonic(connected):
    teleop = connected(script=[{"action": twist(0.1)}], on_end="loop")
    seqs = [teleop.read().seq for _ in range(6)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 6


class SilentTeleop(FakeTeleop):
    def _poll_device(self, handle):
        return None


def test_read_timeout_raises_recoverable_device_fault():
    teleop = SilentTeleop(TeleopConfig())
    teleop.connect()
    try:
        with pytest.raises(DeviceFault, match="no sample") as info:
            teleop.read(timeout_s=0.05)
        assert info.value.recoverable
        assert teleop.latest() is None
        observation = teleop.get_observation()
        assert observation["action"].shape == (6,) and not observation["intervening"][0]
    finally:
        teleop.disconnect()


def test_blocked_read_wakes_with_device_not_connected_on_disconnect():
    import threading

    teleop = SilentTeleop(TeleopConfig())
    teleop.connect()
    outcome: list[BaseException] = []

    def blocked_read():
        try:
            teleop.read(timeout_s=30.0)
        except BaseException as error:
            outcome.append(error)

    reader = threading.Thread(target=blocked_read, daemon=True)
    reader.start()
    reader.join(timeout=0.05)  # let it reach the wait; a late start still raises DeviceNotConnected
    teleop.disconnect()
    reader.join(timeout=0.5)
    assert not reader.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], DeviceNotConnected)


def test_action_space_and_has_gripper_properties():
    teleop = make(action_space="joint_abs", num_joints=3, has_gripper=False)
    assert teleop.action_space == "joint_abs" and teleop.has_gripper is False
    assert make().action_space == "cartesian_delta" and make().has_gripper is True


def test_unconnected_raises_device_not_connected():
    teleop = make()
    for call in (teleop.read, teleop.reset, teleop.get_observation, lambda: teleop.push({})):
        with pytest.raises(DeviceNotConnected):
            call()
    assert teleop.health().ok is False


def test_disconnect_stops_the_poll_thread():
    teleop = make()
    teleop.connect()
    thread = teleop._poll_thread
    assert thread.is_alive() and teleop.health().ok and teleop.health().thread_alive
    teleop.disconnect()
    assert not thread.is_alive()
    assert not teleop.is_connected


def test_fail_after_raises_recoverable_fault_and_reconnect_recovers():
    teleop = make(script=[{"action": twist(0.3)}], fail_after=2)
    teleop.connect()
    try:
        teleop.read()
        teleop.read()
        with pytest.raises(DeviceFault, match="polls in a row") as info:
            teleop.read(timeout_s=2.0)
        assert info.value.recoverable and "reconnect" in info.value.hint
        health = teleop.health()
        assert isinstance(health, TeleopHealth)
        assert health.ok is False and health.error_count >= Teleop.MAX_CONSECUTIVE_FAILURES
        assert "fake teleop failure" in health.detail
        teleop.disconnect()
        teleop.connect()
        assert float(teleop.read().action[0]) == pytest.approx(0.3)
        assert teleop.health().ok
    finally:
        teleop.disconnect()


def test_action_space_and_gripper_queryable_unconnected():
    assert FakeTeleop.ACTION_SPACE == "cartesian_delta" and FakeTeleop.HAS_GRIPPER is True
    joint = make(action_space="joint_abs", num_joints=6, has_gripper=False)
    assert joint.ACTION_SPACE == "joint_abs" and joint.HAS_GRIPPER is False
    assert joint.action_dim == 6
    assert joint.observation_features == {
        "action": {"shape": (6,), "dtype": "float32"},
        "intervening": {"shape": (1,), "dtype": "bool"},
    }
    assert not joint.is_connected


def test_joint_space_action_shape(connected):
    teleop = connected(
        action_space="joint_delta", num_joints=7, has_gripper=False, script=[{"action": [0.1] * 7}]
    )
    sample = teleop.read()
    assert sample.action.shape == (7,) and sample.action_space == "joint_delta"
    assert sample.gripper is None
    assert "gripper" not in teleop.get_observation()


def test_options_and_config_validation():
    with pytest.raises(ValueError, match="num_joints"):
        make(action_space="joint_abs")
    with pytest.raises(ValueError, match="unknown fake teleop option 'speed'"):
        make(speed=1)
    with pytest.raises(ValueError, match="on_end"):
        make(on_end="stop")
    with pytest.raises(ValueError, match="expected"):
        make(script=[{"action": [0.0, 0.0]}])
    with pytest.raises(ValueError, match="unknown key 'activ'"):
        make(script=[{"activ": True}])
    with pytest.raises(ValueError, match="has_gripper"):
        make(has_gripper=False, script=[{"gripper": 0.5}])
    with pytest.raises(ValueError, match="unknown key 'deadzone' for TeleopConfig"):
        TeleopConfig.from_dict({"backend": "fake", "deadzone": 0.1})
    with pytest.raises(ValueError, match="hold_s"):
        TeleopConfig(hold_s=-1.0)
    assert TeleopConfig.from_dict({"name": "sm", "hold_s": 0.2}).name == "sm"


def test_wire_dict_round_trip(connected):
    teleop = connected(script=[{"action": twist(0.4), "buttons": {"a": True}, "events": {"success": True}}])
    sample = teleop.read()
    rebuilt = TeleopSample.from_dict(sample.to_dict())
    assert rebuilt.seq == sample.seq and rebuilt.events == sample.events
    np.testing.assert_array_equal(rebuilt.action, sample.action)


def test_discover_and_registry():
    assert "fake" in Teleop.backends()
    assert Teleop.backend("fake") is FakeTeleop
    assert FakeTeleop.discover() == {"fake"}

"""Hardware-free tests for the layered doctor; OS probes are monkeypatched."""
from __future__ import annotations

import json

import pytest
import yaml

from robot_infra.devices import CameraConfig, DeviceFault, GripperConfig
from robot_infra.devices.claims import DeviceClaim, claim_key
from robot_infra.devices.robot import Robot
from robot_infra.devices.backends.arms.fake import FakeArm
from robot_infra.devices.backends.cameras.fake import FakeCamera
from robot_infra.devices.robots.franka import FrankaConfig
from robot_infra.devices.robots.piper import PiperConfig
from robot_infra.discovery import enumerate as probes
from robot_infra.doctor import cli
from robot_infra.doctor.checks import exit_code, run_doctor, summarize


def franka_config(**kwargs) -> FrankaConfig:
    return FrankaConfig(
        arm_backend="fake",
        gripper=GripperConfig(backend="fake"),
        cameras=[CameraConfig(name="wrist", backend="fake", fps=60)],
        **kwargs,
    )


@pytest.fixture(autouse=True)
def reachable(monkeypatch):
    monkeypatch.setattr(probes, "ping", lambda ip, timeout_s=1.0: True)
    monkeypatch.setattr(probes, "tcp_open", lambda ip, port, timeout_s=0.5: False)
    monkeypatch.setattr(FakeCamera, "discover", classmethod(lambda cls: {"S1"}))


@pytest.fixture
def sdk_backed_fakes(monkeypatch):
    """Make the fakes look like SDK-backed hardware so the OS probes run (os is always importable)."""
    monkeypatch.setattr(FakeArm, "SDK", "os")
    monkeypatch.setattr(FakeCamera, "SDK", "os")


def by_layer(results, layer):
    return [result for result in results if result.layer == layer]


def test_all_green(sdk_backed_fakes):
    results = run_doctor(franka_config())
    assert exit_code(results) == 0
    assert {result.layer for result in results} == {1, 2, 3, 4, 5}
    assert all(result.ok and not result.skipped for result in results)
    names = {result.name for result in results}
    assert {"os:ping", "handshake", "feedback:arm", "feedback:gripper", "feedback:camera:wrist"} <= names
    assert "mode:none" in names


def test_ping_failure_does_not_skip_later_layers(monkeypatch, sdk_backed_fakes):
    monkeypatch.setattr(probes, "ping", lambda ip, timeout_s=1.0: False)
    results = run_doctor(franka_config())
    ping_result = next(result for result in results if result.name == "os:ping")
    assert not ping_result.ok and "Ethernet cable" in ping_result.hint
    assert exit_code(results) == 1
    assert all(r.ok and not r.skipped for layer in (3, 4, 5) for r in by_layer(results, layer))


def test_unknown_backend_skips_later_layers():
    config = franka_config()
    config.arm_backend = "nonexistent"
    results = run_doctor(config)
    sdk_arm = next(result for result in results if result.name == "sdk:arm")
    assert not sdk_arm.ok and "nonexistent" in sdk_arm.message and sdk_arm.hint
    assert all(r.skipped for layer in (3, 4, 5) for r in by_layer(results, layer))
    assert exit_code(results) == 1


def test_handshake_failure_skips_feedback_and_mode(monkeypatch):
    def open_fails(self):
        raise DeviceFault("fci down", hint="enable FCI", device="arm", recoverable=True)

    monkeypatch.setattr(FakeArm, "_open", open_fails)
    results = run_doctor(franka_config())
    handshake = by_layer(results, 3)[0]
    assert not handshake.ok and handshake.hint == "enable FCI" and handshake.autofix is None
    assert all(r.skipped for layer in (4, 5) for r in by_layer(results, layer))


def test_mock_piper_skips_os_and_passes():
    results = run_doctor(PiperConfig(), mock=True)
    assert all(r.skipped for r in by_layer(results, 2)) and by_layer(results, 2)
    assert all(r.message == "no SDK needed" for r in by_layer(results, 1))
    assert exit_code(results) == 0


def test_fix_reopens_camera_and_flips_result(monkeypatch):
    calls = {"reopen": 0}
    original_reopen = FakeCamera.reopen

    def reopen(self):
        calls["reopen"] += 1
        original_reopen(self)

    monkeypatch.setattr(FakeCamera, "reopen", reopen)
    state = {"first": True}
    original_ready = FakeCamera.is_ready

    def is_ready(self, timeout_s=2.0):
        if state["first"]:
            state["first"] = False
            return False
        return original_ready(self, timeout_s)

    monkeypatch.setattr(FakeCamera, "is_ready", is_ready)
    unfixed = run_doctor(franka_config())
    assert exit_code(unfixed) == 1
    state["first"] = True
    fixed = run_doctor(franka_config(), fix=True)
    assert calls["reopen"] == 1
    assert next(r for r in fixed if r.name == "feedback:camera:wrist").ok
    assert exit_code(fixed) == 0


def test_fix_without_autofix_leaves_failure(monkeypatch, sdk_backed_fakes):
    monkeypatch.setattr(probes, "ping", lambda ip, timeout_s=1.0: False)
    results = run_doctor(franka_config(), fix=True)
    assert not next(r for r in results if r.name == "os:ping").ok


def test_no_feedback_arm_fails_with_hint(monkeypatch):
    original = FakeArm._open

    def open_and_inject(self):
        self.inject("no_feedback")
        return original(self)

    monkeypatch.setattr(FakeArm, "_open", open_and_inject)
    feedback = next(r for r in run_doctor(franka_config()) if r.name == "feedback:arm")
    assert not feedback.ok and "follower/slave" in feedback.hint


def test_summarize_counts():
    results = run_doctor(PiperConfig(), mock=True)
    ok, failed, skipped = summarize(results)
    assert failed == 0 and skipped == 3 and ok + skipped == len(results)


def write_config(tmp_path) -> str:
    path = tmp_path / "robot.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "type": "franka",
                "arm_backend": "fake",
                "gripper": {"backend": "fake"},
                "cameras": [{"name": "wrist", "backend": "fake", "fps": 60}],
            }
        )
    )
    return str(path)


def test_cli_text_and_json(tmp_path, capsys):
    path = write_config(tmp_path)
    assert cli.main(["--config", path, "--mock"]) == 0
    text = capsys.readouterr().out
    assert "summary:" in text and "[3/5] handshake ... OK" in text
    assert cli.main(["--config", path, "--mock", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["layer"] for entry in payload} == {1, 2, 3, 4, 5}
    assert all("autofix" not in entry for entry in payload)


def test_fake_backends_skip_os_probes_and_pass(monkeypatch):
    monkeypatch.setattr(probes, "ping", lambda ip, timeout_s=1.0: False)
    results = run_doctor(franka_config(robot_ip="10.0.0.2"))
    assert exit_code(results) == 0
    os_results = {r.name: r for r in by_layer(results, 2)}
    assert {"os:ping", "os:tcp", "os:camera:wrist"} <= set(os_results)
    assert all(r.skipped and r.message == "backend 'fake' needs no OS device" for r in os_results.values())


def test_cli_fills_robot_ip_from_env(tmp_path, monkeypatch, capsys, sdk_backed_fakes):
    pinged = []
    monkeypatch.setattr(probes, "ping", lambda ip, timeout_s=1.0: pinged.append(ip) or True)
    monkeypatch.setenv("ROBOT_IP", "10.9.8.7")
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"type": "franka", "arm_backend": "fake", "robot_ip": "", "gripper": {"backend": "fake"}}))
    assert cli.main(["--config", str(path)]) == 0
    assert pinged == ["10.9.8.7"]
    capsys.readouterr()


def test_can_hints_fill_both_halves(monkeypatch, sdk_backed_fakes):
    monkeypatch.setattr(probes, "list_can_interfaces", lambda: {})
    results = run_doctor(PiperConfig(arm_backend="fake", can_interface="can3", bitrate=500000, gripper=None))
    hint = next(r for r in results if r.name == "os:can_exists").hint
    assert "<iface>" not in hint and "<bitrate>" not in hint
    assert "bash can_activate.sh can3 500000" in hint and "ip link set can3 up type can bitrate 500000" in hint


def test_camera_check_messages_with_unset_serial(monkeypatch, sdk_backed_fakes):
    def camera_result():
        return next(r for r in run_doctor(franka_config()) if r.name == "os:camera:wrist")

    monkeypatch.setattr(FakeCamera, "discover", classmethod(lambda cls: set()))
    none_found = camera_result()
    assert not none_found.ok and none_found.message == "no RealSense device found (serial unset)"
    monkeypatch.setattr(FakeCamera, "discover", classmethod(lambda cls: {"A1", "B2"}))
    several = camera_result()
    assert several.message == "serial unset and 2 devices attached; set serial to one of ['A1', 'B2']"


@pytest.fixture
def claim_dir(tmp_path, monkeypatch):
    import robot_infra.devices.claims as claims

    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


def test_doctor_fails_handshake_when_robot_is_claimed(claim_dir):
    config = franka_config()
    with DeviceClaim(claim_key(Robot.from_config(config))):
        results = run_doctor(config)
    handshake = next(r for r in results if r.name == "handshake")
    assert not handshake.ok and "claimed by another process" in handshake.message
    assert "claim: false" in handshake.hint
    assert exit_code(results) == 1


def test_doctor_releases_claim_afterwards(claim_dir):
    config = franka_config()
    assert exit_code(run_doctor(config)) == 0
    with DeviceClaim(claim_key(Robot.from_config(config))):
        pass

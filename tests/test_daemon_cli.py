"""Daemon CLI: build without serving, preflight gate, signal-driven shutdown."""
from __future__ import annotations

import signal

import pytest

import robot_infra.devices.claims as claims
import robot_infra.server.daemon as daemon
from robot_infra.devices.robots.franka import FrankaRobot
from robot_infra.doctor.checks import CheckResult
from robot_infra.server.transport import Transport

_FAKE_FRANKA_YAML = """\
robot:
  type: franka
  arm_backend: fake
  gripper: {backend: fake}
  cameras:
    - {name: wrist, backend: fake, width: 32, height: 24, fps: 60}
"""


class StubTransport(Transport):
    def __init__(self, on_serve=None) -> None:
        self.served = None
        self.stops = 0
        self._on_serve = on_serve

    def serve(self, service, bind, *, token=None) -> None:
        self.served = (service, bind, token)
        if self._on_serve is not None:
            self._on_serve()

    def start(self, service, bind, *, token=None) -> None:
        self.served = (service, bind, token)

    def stop(self, grace_s: float = 1.0) -> None:
        self.stops += 1


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")
    path = tmp_path / "robot.yaml"
    path.write_text(_FAKE_FRANKA_YAML)
    return str(path)


@pytest.fixture
def stub_transport(monkeypatch):
    transport = StubTransport()
    monkeypatch.setattr(daemon, "_make_transport", lambda name: transport)
    return transport


def test_build_daemon_does_not_serve(config_path, stub_transport):
    service, transport = daemon.build_daemon(
        ["--config", config_path, "--no-preflight", "--bind", "127.0.0.1:0", "--watchdog-s", "0.25"]
    )
    try:
        assert isinstance(service.robot, FrankaRobot)
        assert service.watchdog_s == 0.25
        assert service.jpeg_quality is None
        assert transport is stub_transport and transport.served is None
        assert not service.robot.is_connected
    finally:
        service.close()


def test_jpeg_quality_stored_on_service(config_path, stub_transport):
    service, _ = daemon.build_daemon(["--config", config_path, "--no-preflight", "--jpeg-quality", "80"])
    service.close()
    assert service.jpeg_quality == 80


def test_build_daemon_with_grpc_transport(config_path):
    grpc_transport = pytest.importorskip("robot_infra.server.transport.grpc_transport")
    service, transport = daemon.build_daemon(["--config", config_path, "--no-preflight", "--bind", "127.0.0.1:0"])
    try:
        assert isinstance(transport, grpc_transport.GrpcTransport)
        transport.start(service, "127.0.0.1:0")
    finally:
        transport.stop()
        service.close()


def test_main_serves_and_cleans_up(config_path, stub_transport, capsys):
    assert daemon.main(["--config", config_path, "--no-preflight", "--bind", "127.0.0.1:0", "--token", "t"]) == 0
    out = capsys.readouterr().out
    assert "serving franka 'franka' on 127.0.0.1:0 (token required)" in out
    assert "franka (franka)" in out
    service, bind, token = stub_transport.served
    assert (bind, token) == ("127.0.0.1:0", "t")
    assert not service.robot.is_connected and not service.watchdog_running


def test_sigterm_stops_transport_and_restores_handler(config_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    transport = StubTransport(on_serve=lambda: signal.raise_signal(signal.SIGTERM))
    monkeypatch.setattr(daemon, "_make_transport", lambda name: transport)
    assert daemon.main(["--config", config_path, "--no-preflight"]) == 0
    assert transport.stops >= 2  # once from the handler, once in cleanup
    assert signal.getsignal(signal.SIGTERM) == previous


def test_failed_preflight_exits_1(config_path, stub_transport, monkeypatch, capsys):
    monkeypatch.setattr(
        daemon, "run_doctor", lambda config: [CheckResult(1, "sdk:arm", False, "boom", "install it")]
    )
    assert daemon.main(["--config", config_path]) == 1
    out = capsys.readouterr().out
    assert "sdk:arm ... FAIL  boom" in out and "hint: install it" in out
    assert "preflight failed; fix the items above or start with --no-preflight" in out
    assert stub_transport.served is None


def test_passing_preflight_serves(config_path, stub_transport, monkeypatch, capsys):
    monkeypatch.setattr(daemon, "run_doctor", lambda config: [CheckResult(1, "sdk:arm", True, "fine")])
    assert daemon.main(["--config", config_path]) == 0
    assert "summary: 1 ok, 0 failed, 0 skipped" in capsys.readouterr().out
    assert stub_transport.served is not None

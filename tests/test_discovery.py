"""Hardware-free tests for discovery probes and env-based auto-fill."""
from __future__ import annotations

import pytest

from robot_infra.devices import CameraConfig
from robot_infra.devices.backends.cameras.fake import FakeCamera
from robot_infra.devices.backends.cameras.realsense import RealSenseCamera
from robot_infra.devices.robots.franka import FrankaConfig
from robot_infra.devices.robots.piper import PiperConfig
from robot_infra.discovery import enumerate as enum
from robot_infra.discovery.autoconfig import fill_from_env


def write_iface(root, name, iface_type, operstate):
    iface = root / name
    iface.mkdir()
    (iface / "type").write_text(f"{iface_type}\n")
    (iface / "operstate").write_text(f"{operstate}\n")


def test_list_can_interfaces_keeps_only_can(tmp_path, monkeypatch):
    write_iface(tmp_path, "can0", 280, "up")
    write_iface(tmp_path, "eth0", 1, "up")
    monkeypatch.setattr(enum, "can_bitrate", lambda name: 1_000_000)
    interfaces = enum.list_can_interfaces(tmp_path)
    assert interfaces == {"can0": enum.CanInfo("can0", "up", 1_000_000)}


def test_list_can_interfaces_missing_root(tmp_path):
    assert enum.list_can_interfaces(tmp_path / "absent") == {}


def test_ping_without_binary_is_false(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(enum.subprocess, "run", missing)
    assert enum.ping("10.0.0.1") is False
    assert enum.can_bitrate("can0") is None


def test_discover_all_and_format(monkeypatch):
    monkeypatch.setattr(enum, "ping", lambda ip, timeout_s=1.0: True)
    monkeypatch.setattr(enum, "tcp_open", lambda ip, port, timeout_s=0.5: port == 1337)
    monkeypatch.setattr(
        enum, "list_can_interfaces", lambda: {"can7": enum.CanInfo("can7", "up", None)}
    )
    monkeypatch.setattr(RealSenseCamera, "discover", classmethod(lambda cls: {"RS123"}))
    report = enum.discover_all(["172.16.0.2"])
    assert "fake" not in report.cameras
    assert report.cameras["realsense"] == {"RS123"}
    assert report.hosts["172.16.0.2"] == enum.HostInfo("172.16.0.2", True, [1337])
    text = report.format()
    assert "can7" in text and "172.16.0.2" in text and "RS123" in text
    assert "FAKE0" not in text and "fake" not in text


def test_empty_report_formats_none_lines():
    assert enum.DiscoveryReport().format().count("none") == 4


def test_camera_discover_exception_gives_empty_set(monkeypatch):
    def boom(cls):
        raise RuntimeError("usb")

    monkeypatch.setattr(FakeCamera, "discover", classmethod(boom))
    assert enum.discover_cameras()["fake"] == set()


def test_hosts_from_config():
    assert enum.hosts_from_config(FrankaConfig(robot_ip="1.2.3.4")) == ["1.2.3.4"]
    assert enum.hosts_from_config(PiperConfig()) == []


def test_fill_from_env_casts_and_fills_only_empty():
    franka = fill_from_env(
        FrankaConfig(arm_backend="fake", robot_ip=""), {"ROBOT_IP": "10.0.0.5"}
    )
    assert franka.robot_ip == "10.0.0.5"
    kept = fill_from_env(FrankaConfig(robot_ip="1.1.1.1"), {"ROBOT_IP": "10.0.0.5"})
    assert kept.robot_ip == "1.1.1.1"
    piper = fill_from_env(PiperConfig(can_interface=""), {"CAN_INTERFACE": "can3", "BITRATE": "5"})
    assert piper.can_interface == "can3" and piper.bitrate == 1_000_000
    assert piper.arm_backend == "pyagxarm"


@pytest.mark.parametrize(
    "serials, expected", [({"ONLY"}, "ONLY"), ({"A", "B"}, "")], ids=["single", "ambiguous"]
)
def test_fill_from_env_camera_serial(monkeypatch, serials, expected):
    monkeypatch.setattr(FakeCamera, "discover", classmethod(lambda cls: set(serials)))
    config = FrankaConfig(
        arm_backend="fake", cameras=[CameraConfig(name="wrist", backend="fake")]
    )
    assert fill_from_env(config, {}).cameras[0].serial == expected

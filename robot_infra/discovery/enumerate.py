"""Read-only probes: cameras, teleop devices, CAN interfaces, reachable hosts and HID devices."""
from __future__ import annotations

import glob
import re
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from robot_infra.devices import Camera, RobotConfig, Teleop

_ARPHRD_CAN = "280"
_DEFAULT_PORTS = (22, 80, 443, 1337, 50051)


@dataclass
class CanInfo:
    name: str
    operstate: str
    bitrate: int | None


@dataclass
class HostInfo:
    ip: str
    ping_ok: bool
    open_ports: list[int]


@dataclass
class DiscoveryReport:
    cameras: dict[str, set[str]] = field(default_factory=dict)
    teleops: dict[str, set[str]] = field(default_factory=dict)
    can_interfaces: dict[str, CanInfo] = field(default_factory=dict)
    hosts: dict[str, HostInfo] = field(default_factory=dict)
    hid_devices: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = ["cameras:"]
        if any(self.cameras.values()):
            for backend, serials in sorted(self.cameras.items()):
                if serials:
                    lines.append(f"  {backend}: {', '.join(sorted(serials))}")
        else:
            lines.append("  none")
        lines.append("teleops:")
        if any(self.teleops.values()):
            for backend, devices in sorted(self.teleops.items()):
                if devices:
                    lines.append(f"  {backend}: {', '.join(sorted(devices))}")
        else:
            lines.append("  none")
        lines.append("can interfaces:")
        if self.can_interfaces:
            for info in self.can_interfaces.values():
                bitrate = "unknown" if info.bitrate is None else str(info.bitrate)
                lines.append(f"  {info.name}: operstate={info.operstate} bitrate={bitrate}")
        else:
            lines.append("  none")
        lines.append("hosts:")
        if self.hosts:
            for host in self.hosts.values():
                ports = ",".join(str(port) for port in host.open_ports) or "none"
                lines.append(
                    f"  {host.ip}: ping={'ok' if host.ping_ok else 'no reply'} open ports={ports}"
                )
        else:
            lines.append("  none")
        lines.append("hid devices:")
        lines.extend(f"  {path}" for path in self.hid_devices or ["none"])
        return "\n".join(lines)


def can_bitrate(name: str) -> int | None:
    try:
        output = subprocess.run(
            ["ip", "-details", "link", "show", name],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        ).stdout
        match = re.search(r"bitrate (\d+)", output)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def list_can_interfaces(sys_class_net: Path = Path("/sys/class/net")) -> dict[str, CanInfo]:
    interfaces: dict[str, CanInfo] = {}
    if not sys_class_net.is_dir():
        return interfaces
    for iface in sorted(sys_class_net.iterdir()):
        try:
            if (iface / "type").read_text().strip() != _ARPHRD_CAN:
                continue
            operstate = (iface / "operstate").read_text().strip()
        except OSError:
            continue
        interfaces[iface.name] = CanInfo(iface.name, operstate, can_bitrate(iface.name))
    return interfaces


def ping(ip: str, timeout_s: float = 1.0) -> bool:
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(timeout_s)), ip],
            capture_output=True,
            timeout=timeout_s + 2.0,
        )
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def tcp_open(ip: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def discover_cameras(hardware_only: bool = False) -> dict[str, set[str]]:
    """Serials per camera backend; ``hardware_only`` skips backends without a vendor SDK (the fake)."""
    cameras: dict[str, set[str]] = {}
    for name, backend_cls in sorted(Camera.backends().items()):
        if hardware_only and backend_cls.SDK is None:
            continue
        try:
            cameras[name] = set(backend_cls.discover())
        except Exception:
            cameras[name] = set()
    return cameras


def discover_teleops(hardware_only: bool = False) -> dict[str, set[str]]:
    """Device ids per teleop backend; ``hardware_only`` skips backends without a vendor SDK (the fake)."""
    teleops: dict[str, set[str]] = {}
    for name, backend_cls in sorted(Teleop.backends().items()):
        if hardware_only and backend_cls.SDK is None:
            continue
        try:
            teleops[name] = set(backend_cls.discover())
        except Exception:
            teleops[name] = set()
    return teleops


def discover_all(
    host_ips: Iterable[str] = (), ports: Iterable[int] = _DEFAULT_PORTS
) -> DiscoveryReport:
    ports = list(ports)
    hosts = {
        ip: HostInfo(ip, ping(ip), [port for port in ports if tcp_open(ip, port)])
        for ip in host_ips
    }
    return DiscoveryReport(
        cameras=discover_cameras(hardware_only=True),
        teleops=discover_teleops(hardware_only=True),
        can_interfaces=list_can_interfaces(),
        hosts=hosts,
        hid_devices=sorted(glob.glob("/dev/hidraw*")),
    )


def hosts_from_config(config: RobotConfig) -> list[str]:
    ip = getattr(config, "robot_ip", None)
    return [ip] if ip else []

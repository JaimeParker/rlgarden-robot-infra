"""Hardware discovery and config auto-fill; probes never connect a robot."""
from robot_infra.discovery.autoconfig import fill_from_env, load_config
from robot_infra.discovery.enumerate import (
    CanInfo,
    DiscoveryReport,
    HostInfo,
    discover_all,
    hosts_from_config,
)

__all__ = [
    "CanInfo",
    "DiscoveryReport",
    "HostInfo",
    "discover_all",
    "fill_from_env",
    "hosts_from_config",
    "load_config",
]

"""``python -m robot_infra.discovery [--config robot.yaml]``: print what hardware is visible."""
from __future__ import annotations

import argparse

from robot_infra.discovery.autoconfig import load_config
from robot_infra.discovery.enumerate import discover_all, hosts_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot_infra.discovery", description=__doc__)
    parser.add_argument("--config", help="robot yaml; its robot IP is probed (never connected)")
    args = parser.parse_args(argv)
    host_ips = hosts_from_config(load_config(args.config)) if args.config else []
    print(discover_all(host_ips).format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Robot daemon: ``python -m robot_infra.server --config robot.yaml`` serves one robot over a
transport until SIGINT/SIGTERM, then disconnects it (never powers it off)."""
from __future__ import annotations

import argparse
import signal
import threading
from typing import Sequence

from robot_infra.devices import Robot, RobotConfig
from robot_infra.discovery.autoconfig import load_config_with_env
from robot_infra.doctor.checks import exit_code, run_doctor
from robot_infra.doctor.cli import format_text
from robot_infra.server.service import RobotService
from robot_infra.server.transport.base import Transport

_TRANSPORT_NAMES = ("grpc",)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="robot_infra.server", description="Serve one robot over the network")
    parser.add_argument("--config", required=True, help="robot yaml")
    parser.add_argument("--bind", default="0.0.0.0:50051", help="host:port to listen on")
    parser.add_argument("--token", default=None, help="shared secret clients must send")
    parser.add_argument("--no-preflight", action="store_true", help="skip the doctor before serving")
    parser.add_argument("--watchdog-s", type=float, default=0.5, help="idle a servoing arm after this long without actions")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="lossy-JPEG the color image of every camera frame at this quality (1-100); depth stays raw. "
        "Train and deploy with the same value; default sends raw frames")
    parser.add_argument("--transport", choices=_TRANSPORT_NAMES, default="grpc")
    return parser.parse_args(argv)


def _make_transport(name: str) -> Transport:
    if name == "grpc":
        from robot_infra.server.transport.grpc_transport import GrpcTransport

        return GrpcTransport()
    raise ValueError(f"unknown transport '{name}'; choose from {list(_TRANSPORT_NAMES)}")


def load_daemon_config(args: argparse.Namespace) -> RobotConfig:
    return load_config_with_env(args.config)


def preflight_passed(config: RobotConfig) -> bool:
    results = run_doctor(config)
    print(format_text(results))
    if exit_code(results):
        print("preflight failed; fix the items above or start with --no-preflight")
        return False
    return True


def build_daemon(
    args: Sequence[str] | argparse.Namespace, *, config: RobotConfig | None = None
) -> tuple[RobotService, Transport]:
    """Build the service and an unstarted transport without serving."""
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)
    config = load_daemon_config(args) if config is None else config
    service = RobotService(Robot.from_config(config), watchdog_s=args.watchdog_s, jpeg_quality=args.jpeg_quality)
    return service, _make_transport(args.transport)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_daemon_config(args)
    if not args.no_preflight and not preflight_passed(config):
        return 1
    service, transport = build_daemon(args, config=config)
    robot = service.robot
    print(robot.describe_text())
    print(f"serving {robot.ROBOT_TYPE} '{robot.name}' on {args.bind}" + (" (token required)" if args.token else ""))

    def request_stop(signum: int, frame: object) -> None:
        transport.stop()

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        transport.serve(service, args.bind, token=args.token)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        transport.stop()
        service.close()
    return 0

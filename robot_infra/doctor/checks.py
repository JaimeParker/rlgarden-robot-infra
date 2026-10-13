"""Layered diagnostic: 1 sdk, 2 os, 3 handshake, 4 feedback, 5 mode. Each check is a small
closure so ``fix`` can run its autofix and then re-run exactly that check."""
from __future__ import annotations

import dataclasses
import glob
import os
from dataclasses import dataclass, field
from typing import Callable

from robot_infra.devices import (
    Arm,
    Camera,
    DeviceError,
    DeviceFault,
    Gripper,
    Robot,
    RobotConfig,
    SdkMissing,
    Teleop,
)
from robot_infra.devices.claims import DeviceClaim, claim_key
from robot_infra.discovery import enumerate as probes

LAYER_NAMES = {1: "sdk", 2: "os", 3: "handshake", 4: "feedback", 5: "mode"}
_FRANKA_FCI_PORT = 1337
_CAMERA_READY_TIMEOUT_S = 2.0
_HIDRAW_GLOB = "/dev/hidraw*"


@dataclass
class CheckResult:
    layer: int
    name: str
    ok: bool
    message: str
    hint: str = ""
    skipped: bool = False
    autofix: Callable[[], None] | None = field(default=None, repr=False, compare=False)


Outcome = tuple  # (ok, message) or (ok, message, hint)
Check = Callable[[], CheckResult]


def _make_check(layer: int, name: str, fn: Callable[[], Outcome]) -> Check:
    def run() -> CheckResult:
        try:
            ok, message, hint = (*fn(), "")[:3]
        except DeviceError as error:
            return CheckResult(layer, name, False, error.message, error.hint)
        except Exception as error:
            return CheckResult(layer, name, False, repr(error))
        return CheckResult(layer, name, ok, message, "" if ok else hint)

    return run


def _skipped(layer: int, name: str, message: str) -> CheckResult:
    return CheckResult(layer, name, True, message, skipped=True)


def _mocked(config: RobotConfig) -> RobotConfig:
    gripper = config.gripper
    return dataclasses.replace(
        config,
        arm_backend="fake",
        gripper=dataclasses.replace(gripper, backend="fake") if gripper else None,
        cameras=[dataclasses.replace(camera, backend="fake") for camera in config.cameras],
        teleop=_mocked_teleop(config.teleop),
    )


def _mocked_teleop(teleop):
    if teleop is None or teleop.backend == "fake":
        return teleop
    # Another backend's private options mean nothing to the fake.
    return dataclasses.replace(teleop, backend="fake", options={})


def _sdk_outcome(backend_cls: type) -> Outcome:
    if backend_cls.SDK is None:
        return True, "no SDK needed"
    try:
        backend_cls.require_sdk()
    except SdkMissing as error:
        return False, error.message, error.hint
    return True, f"'{backend_cls.SDK}' importable"


def _sdk_layer(config: RobotConfig) -> tuple[list[Check], bool]:
    """Returns the layer-1 checks and whether every backend name resolved."""
    targets: list[tuple[str, Callable[[], type]]] = [
        ("sdk:arm", lambda: Arm.backend(config.arm_backend))
    ]
    if config.gripper:
        targets.append(("sdk:gripper", lambda: Gripper.backend(config.gripper.backend)))
    for camera in config.cameras:
        targets.append(
            (f"sdk:camera:{camera.name}", lambda camera=camera: Camera.backend(camera.backend))
        )
    if config.teleop:
        targets.append(
            (f"sdk:teleop:{config.teleop.backend}", lambda: Teleop.backend(config.teleop.backend))
        )
    checks: list[Check] = []
    resolved = True
    for name, resolve in targets:
        try:
            backend_cls = resolve()
            fn = lambda backend_cls=backend_cls: _sdk_outcome(backend_cls)
        except ValueError as error:
            resolved = False
            fn = lambda error=error: (
                False,
                str(error),
                "pip install the backend extra or check arm_backend",
            )
        checks.append(_make_check(1, name, fn))
    return checks, resolved


def _os_checks(config: RobotConfig) -> list[tuple[str, Check]]:
    named: list[tuple[str, Callable[[], Outcome]]] = []
    robot_ip = getattr(config, "robot_ip", None)
    if robot_ip:
        named += [("os:ping", _ping_outcome(robot_ip)), ("os:tcp", _tcp_outcome(robot_ip))]
    can_interface = getattr(config, "can_interface", None)
    if can_interface:
        named += _can_outcomes(can_interface, getattr(config, "bitrate", None))
    arm_needs_no_device = _backend_needs_no_device(Arm, config.arm_backend)
    checks: list[tuple[str, Check]] = []
    for name, fn in named:
        if arm_needs_no_device:
            checks.append((name, _no_device_check(name, config.arm_backend)))
        else:
            checks.append((name, _make_check(2, name, fn)))
    for camera in config.cameras:
        name = f"os:camera:{camera.name}"
        if _backend_needs_no_device(Camera, camera.backend):
            checks.append((name, _no_device_check(name, camera.backend)))
        else:
            checks.append((name, _make_check(2, name, _camera_outcome(camera))))
    if config.teleop:
        name = f"os:teleop:{config.teleop.name}"
        if _backend_needs_no_device(Teleop, config.teleop.backend):
            checks.append((name, _no_device_check(name, config.teleop.backend)))
        else:
            checks.append((name, _make_check(2, name, _teleop_outcome(config.teleop))))
        if config.teleop.backend == "spacemouse":
            checks.append(("os:hid", _make_check(2, "os:hid", _hid_outcome())))
    return checks


def _backend_needs_no_device(part_cls: type, backend: str) -> bool:
    try:
        return part_cls.backend(backend).SDK is None
    except ValueError:
        return False


def _no_device_check(name: str, backend: str) -> Check:
    return lambda: _skipped(2, name, f"backend '{backend}' needs no OS device")


def _ping_outcome(ip: str) -> Callable[[], Outcome]:
    def outcome() -> Outcome:
        reachable = probes.ping(ip)
        return (
            reachable,
            f"{ip} {'replies to ping' if reachable else 'does not reply'}",
            "check the Ethernet cable to the robot's Control port and the robot IP in Desk",
        )

    return outcome


def _tcp_outcome(ip: str) -> Callable[[], Outcome]:
    def outcome() -> Outcome:
        state = "open" if probes.tcp_open(ip, _FRANKA_FCI_PORT) else "closed"
        return True, f"port {_FRANKA_FCI_PORT} {state} (informational)"

    return outcome


def _can_outcomes(iface: str, bitrate: int | None) -> list[tuple[str, Callable[[], Outcome]]]:
    activate_hint = (
        f"run `bash can_activate.sh {iface} {bitrate}` from piper_sdk, or "
        f"`sudo ip link set {iface} up type can bitrate {bitrate}`"
    )

    def exists() -> Outcome:
        found = iface in probes.list_can_interfaces()
        return found, f"CAN interface {iface} {'exists' if found else 'not found'}", activate_hint

    def up() -> Outcome:
        info = probes.list_can_interfaces().get(iface)
        state = info.operstate if info else "missing"
        return state == "up", f"{iface} operstate={state}", activate_hint

    def bitrate_matches() -> Outcome:
        info = probes.list_can_interfaces().get(iface)
        actual = info.bitrate if info else None
        return (
            actual == bitrate,
            f"{iface} bitrate={actual}, expected {bitrate}",
            f"expected bitrate {bitrate} but found {actual}; " + activate_hint,
        )

    return [("os:can_exists", exists), ("os:can_up", up), ("os:can_bitrate", bitrate_matches)]


def _camera_outcome(camera) -> Callable[[], Outcome]:
    def outcome() -> Outcome:
        discovered = sorted(Camera.backend(camera.backend).discover())
        if not camera.serial and len(discovered) == 1:
            return True, f"will use {discovered[0]}"
        if camera.serial in discovered:
            return True, f"serial {camera.serial} found"
        if not camera.serial:
            if not discovered:
                return False, "no RealSense device found (serial unset)", "plug in the camera and run `python -m robot_infra.discovery`"
            return (
                False,
                f"serial unset and {len(discovered)} devices attached; set serial to one of {discovered}",
                f"set serial; discovered: {discovered}",
            )
        return (
            False,
            f"camera '{camera.name}' serial '{camera.serial}' not found",
            f"set serial; discovered: {discovered}",
        )

    return outcome


def _teleop_outcome(teleop) -> Callable[[], Outcome]:
    def outcome() -> Outcome:
        discovered = sorted(Teleop.backend(teleop.backend).discover())
        if discovered:
            message = f"teleop '{teleop.name}' devices: {discovered}"
        else:
            message = f"no '{teleop.backend}' teleop device found"
        return (
            bool(discovered),
            message,
            "plug in the teleop device and run `python -m robot_infra.discovery`",
        )

    return outcome


def _hid_outcome(pattern: str | None = None) -> Callable[[], Outcome]:
    def outcome() -> Outcome:
        paths = sorted(glob.glob(pattern or _HIDRAW_GLOB))
        if not paths:
            return (
                False,
                "no HID device found",
                "no /dev/hidraw* node: check the USB cable and port; inside Docker pass it with "
                "`docker run --device=/dev/hidrawN`",
            )
        denied = [path for path in paths if not os.access(path, os.R_OK | os.W_OK)]
        if denied:
            return (
                False,
                f"no read/write access to {denied}",
                'add a udev rule `SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"` to '
                "/etc/udev/rules.d/99-spacemouse.rules, then `sudo udevadm control --reload && "
                "sudo udevadm trigger` and replug; inside Docker pass the node with "
                "`docker run --device=/dev/hidrawN`",
            )
        return True, f"{len(paths)} HID device(s) readable and writable: {paths}"

    return outcome


def run_doctor(config: RobotConfig, *, mock: bool = False, fix: bool = False) -> list[CheckResult]:
    if mock:
        config = _mocked(config)
    results: list[CheckResult] = []
    reruns: dict[int, Check] = {}

    def record(check: Check) -> CheckResult:
        result = check()
        reruns[len(results)] = check
        results.append(result)
        return result

    sdk_checks, backends_resolved = _sdk_layer(config)
    for check in sdk_checks:
        record(check)

    for name, check in _os_checks(config):
        if mock:
            results.append(_skipped(2, name, "skipped (--mock)"))
        else:
            record(check)

    if not backends_resolved:
        for layer in (3, 4, 5):
            results.append(_skipped(layer, LAYER_NAMES[layer], "skipped: unknown backend"))
        return results

    robot: Robot | None = None
    claim: DeviceClaim | None = None
    try:
        robot = Robot.from_config(config)
        claim = DeviceClaim(claim_key(robot)) if config.claim else None
        handshake = record(_handshake_check(robot, claim))
        if not handshake.ok:
            for layer in (4, 5):
                results.append(_skipped(layer, LAYER_NAMES[layer], "skipped: handshake failed"))
        else:
            _feedback_and_mode_layers(robot, record, results, reruns)
        if fix:
            _apply_fixes(results, reruns)
    except Exception as error:
        results.append(CheckResult(3, "handshake", False, repr(error)))
    finally:
        if robot is not None:
            _disconnect(robot, results)
        if claim is not None and claim.held:
            claim.release()
    return results


def _handshake_check(robot: Robot, claim: DeviceClaim | None) -> Check:
    def run() -> CheckResult:
        try:
            if claim is not None:
                claim.acquire()
            robot.connect()
        except DeviceFault as error:
            return CheckResult(3, "handshake", False, error.message, error.hint)
        except DeviceError as error:
            return CheckResult(3, "handshake", False, error.message, error.hint)
        except Exception as error:
            return CheckResult(3, "handshake", False, repr(error))
        return CheckResult(3, "handshake", True, "robot connected")

    return run


def _feedback_and_mode_layers(robot: Robot, record, results: list[CheckResult], reruns) -> None:
    arm = robot.arm

    def arm_feedback() -> Outcome:
        arm.get_observation()
        received = arm.health().first_feedback_received
        return (
            received,
            "arm feedback received" if received else "arm reports no feedback",
            "the arm reports no feedback: check power, e-stop and, for CAN arms, follower/slave mode",
        )

    record(_make_check(4, "feedback:arm", arm_feedback))
    if robot.gripper is not None:
        gripper = robot.gripper
        record(_make_check(4, "feedback:gripper", lambda: (gripper.get_observation() is not None, "gripper readable")))
    for name, camera in robot.cameras.items():
        record(_camera_feedback_check(name, camera))
    if robot.teleop is not None:
        record(_make_check(4, f"feedback:teleop:{robot.teleop.name}", _teleop_feedback(robot.teleop)))

    try:
        mode_results = arm.preflight()
    except Exception as error:
        mode_results = [CheckResult(5, "mode:preflight", False, repr(error))]
    if not mode_results:
        record(_make_check(5, "mode:none", lambda: (True, "backend declares no mode checks")))
    for mode_result in mode_results:
        reruns[len(results)] = _preflight_rerun(arm, mode_result)
        results.append(mode_result)


def _camera_feedback_check(name: str, camera: Camera) -> Check:
    def run() -> CheckResult:
        ready = camera.is_ready(_CAMERA_READY_TIMEOUT_S)
        if ready:
            return CheckResult(4, f"feedback:camera:{name}", True, "frames arriving")
        return CheckResult(
            4,
            f"feedback:camera:{name}",
            False,
            "no frames within 2s",
            "no frames within 2s: check USB 3 bandwidth; try another port; run with --fix to reopen",
            autofix=camera.reopen,
        )

    return run


def _teleop_feedback(teleop: Teleop) -> Callable[[], Outcome]:
    def outcome() -> Outcome:
        health = teleop.health()
        teleop.latest()
        problem = health.detail or "poll thread not running"
        return (
            health.ok,
            "teleop poll thread alive" if health.ok else f"teleop unhealthy: {problem}",
            "check the teleop device cable and permissions, then reconnect",
        )

    return outcome


def _preflight_rerun(arm: Arm, original: CheckResult) -> Check:
    def run() -> CheckResult:
        for result in arm.preflight():
            if result.name == original.name:
                return result
        return original

    return run


def _apply_fixes(results: list[CheckResult], reruns: dict[int, Check]) -> None:
    for index, result in enumerate(list(results)):
        if result.ok or result.autofix is None or index not in reruns:
            continue
        try:
            result.autofix()
        except Exception as error:
            result.message = f"{result.message}; autofix failed: {error!r}"
            continue
        results[index] = reruns[index]()


def _disconnect(robot: Robot, results: list[CheckResult]) -> None:
    try:
        robot.disconnect()
    except Exception as error:
        results.append(CheckResult(3, "disconnect", False, repr(error)))


def summarize(results: list[CheckResult]) -> tuple[int, int, int]:
    skipped = sum(result.skipped for result in results)
    failed = sum(not result.ok and not result.skipped for result in results)
    return len(results) - skipped - failed, failed, skipped


def exit_code(results: list[CheckResult]) -> int:
    return 1 if summarize(results)[1] else 0

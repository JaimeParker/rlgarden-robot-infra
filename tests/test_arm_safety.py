"""Device-layer safety limits: ArmSafety, the Arm command templates, the daemon and remote clients."""
from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path

import numpy as np
import pytest

import robot_infra.devices.claims as claims
from robot_infra.devices import ArmHealth, ControlMode, DeviceFault, GripperConfig, Robot
from robot_infra.devices.backends.arms.fake import FakeArm
from robot_infra.devices.errors import DeviceError, SafetyViolation
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.devices.robots.piper import PiperConfig
from robot_infra.devices.safety import ArmSafety
from robot_infra.server import RemoteRobot
from robot_infra.server.service import RobotService
from robot_infra.server.transport import LocalClient

BOX = [0.2, -0.3, 0.1, 0.6, 0.3, 0.5]
JOINT_LIMITS = [[-1.0, 1.0]] * 7
QUAT = [0.0, 0.0, 0.0, 1.0]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def connected_arm(safety: ArmSafety | None, arm_cls: type[FakeArm] = FakeArm) -> FakeArm:
    arm = arm_cls(num_joints=7, safety=safety)
    arm.connect()
    return arm


def safe_franka(**overrides) -> FrankaRobot:
    values = dict(
        arm_backend="fake",
        gripper=GripperConfig(backend="fake"),
        home_joints=[0.0] * 7,
        safety_box=BOX,
        joint_limits=JOINT_LIMITS,
    )
    values.update(overrides)
    return FrankaRobot(FrankaConfig(**values))


class CountingFakeArm(FakeArm):
    reads = 0

    def read_state(self):
        self.reads += 1
        return super().read_state()


@pytest.fixture(autouse=True)
def isolated_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


@pytest.fixture
def services():
    created: list[RobotService] = []

    def make(robot=None, **kwargs) -> RobotService:
        service = RobotService(robot or safe_franka(), **kwargs)
        created.append(service)
        return service

    yield make
    for service in created:
        service.close()


# ------------------------------------------------------------------ ArmSafety


def test_arm_safety_from_config_none_when_unlimited():
    assert ArmSafety.from_config(FrankaConfig()) is None
    assert ArmSafety.from_config(FrankaConfig(max_delta_per_step={})) is None


def test_arm_safety_dict_round_trip():
    config = FrankaConfig(
        home_joints=[0.0] * 7,
        safety_box=BOX,
        joint_limits=JOINT_LIMITS,
        max_delta_per_step={"xyz_m": 0.05},
    )
    limits = ArmSafety.from_config(config).to_dict()
    assert limits == {
        "safety_box": BOX,
        "joint_limits": JOINT_LIMITS,
        "max_delta_per_step": {"xyz_m": 0.05},
    }
    assert ArmSafety.from_dict(limits).to_dict() == limits


def test_arm_safety_require_names_dimension_value_and_bound():
    safety = ArmSafety(safety_box=np.array(BOX), joint_limits=np.array(JOINT_LIMITS))
    with pytest.raises(SafetyViolation, match=r"z=0\.9000 not in \[0\.1000, 0\.5000\]"):
        safety.require_pose_inside(np.array([0.4, 0.0, 0.9, *QUAT]))
    with pytest.raises(SafetyViolation, match=r"joint 3=1\.5000 not in \[-1\.0000, 1\.0000\]"):
        safety.require_joints_inside(np.array([0, 0, 0, 1.5, 0, 0, 0.0]))
    safety.require_pose_inside(np.array([0.6, 0.3, 0.5, *QUAT]))  # boundary is inside


def test_safety_violation_is_not_a_device_fault():
    error = SafetyViolation("outside", hint="h", device="arm")
    assert not isinstance(error, DeviceFault)
    restored = DeviceError.from_dict(error.to_dict())
    assert type(restored) is SafetyViolation and restored.hint == "h"


# ------------------------------------------------------------------ Arm templates


def test_servo_cartesian_outside_box_is_clipped_and_counted():
    arm = connected_arm(ArmSafety(safety_box=np.array(BOX)))
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    quat = np.array([0.0, 0.0, 0.70710678, 0.70710678])
    arm.servo_cartesian(np.array([1.0, -0.5, 0.3, *quat]))
    np.testing.assert_allclose(arm._pose_target[:3], [0.6, -0.3, 0.3])
    np.testing.assert_allclose(arm._pose_target[3:], quat / np.linalg.norm(quat))
    assert arm.health().safety_clips == 1
    arm.servo_cartesian(np.array([0.4, 0.0, 0.3, *QUAT]))
    assert arm.health().safety_clips == 1


def test_servo_joints_outside_limits_is_clipped_and_counted():
    arm = connected_arm(ArmSafety(joint_limits=np.array(JOINT_LIMITS)))
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.servo_joints(np.array([2.0, -3.0, 0.5, 0, 0, 0, 0]))
    np.testing.assert_allclose(arm._joint_target, [1.0, -1.0, 0.5, 0, 0, 0, 0])
    assert arm.health().safety_clips == 1


@pytest.mark.parametrize("via_send_action", [False, True])
def test_send_action_and_direct_servo_are_limited_alike(via_send_action):
    arm = connected_arm(ArmSafety(joint_limits=np.array(JOINT_LIMITS)))
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    q = np.full(7, 5.0)
    if via_send_action:
        arm.send_action({"joint_pos": q})
    else:
        arm.servo_joints(q)
    np.testing.assert_allclose(arm._joint_target, 1.0)


@pytest.mark.parametrize(
    "method, target",
    [
        ("move_to_pose", np.array([0.4, 0.0, 0.9, *QUAT])),
        ("move_to_joints", np.array([0, 0, 0, 1.5, 0, 0, 0.0])),
    ],
)
def test_move_outside_limits_is_rejected_before_the_backend(method, target):
    arm = connected_arm(ArmSafety(safety_box=np.array(BOX), joint_limits=np.array(JOINT_LIMITS)))
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.call_log.clear()
    with pytest.raises(SafetyViolation) as info:
        getattr(arm, method)(target)
    assert info.value.device == arm.name
    assert method not in arm.call_log
    health = arm.health()
    assert health.safety_rejects == 1 and health.errors == [] and health.ok
    assert arm.control_mode is ControlMode.CARTESIAN_SERVO


def test_move_inside_limits_reaches_the_backend():
    arm = connected_arm(ArmSafety(safety_box=np.array(BOX), joint_limits=np.array(JOINT_LIMITS)))
    arm.move_to_joints(np.full(7, 0.5))
    arm.move_to_pose(np.array([0.4, 0.0, 0.3, *QUAT]))
    assert arm.call_log[-2:] == ["move_to_joints", "move_to_pose"]
    assert arm.health().safety_rejects == 0


def test_no_guard_still_sanitizes_direct_calls():
    arm = connected_arm(None)
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.call_log.clear()
    with pytest.raises(ValueError, match="non-finite"):
        arm.servo_cartesian(np.array([0.4, np.nan, 0.3, *QUAT]))
    with pytest.raises(ValueError, match="quaternion norm"):
        arm.move_to_pose(np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="shape"):
        arm.servo_joints(np.zeros(6))
    assert arm.call_log == []
    arm.servo_cartesian(np.array([5.0, 0.0, 0.3, *QUAT]))
    assert arm._pose_target[0] == 5.0
    assert arm.safety_limits is None


def test_max_delta_xyz_shortens_the_step_and_reads_only_when_stale():
    arm = connected_arm(ArmSafety(max_delta_xyz_m=0.05), CountingFakeArm)
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.servo_cartesian(np.array([0.5, 0.0, 0.0, *QUAT]))  # empty cache -> one read
    assert arm.reads == 1
    assert np.linalg.norm(arm._pose_target[:3] - arm._last_state["tcp_pose"][:3]) == pytest.approx(0.05)
    assert arm.health().safety_clips == 1

    arm.read_state()
    reads = arm.reads
    arm.servo_cartesian(np.array([0.0, 0.5, 0.0, *QUAT]))  # fresh cache -> no read
    assert arm.reads == reads
    step = arm._pose_target[:3] - arm._last_state["tcp_pose"][:3]
    assert np.linalg.norm(step) == pytest.approx(0.05)

    arm._last_state_time -= 2 * arm.MAX_DELTA_STATE_AGE_S
    arm.servo_cartesian(np.array([0.0, 0.5, 0.0, *QUAT]))  # stale cache -> one read
    assert arm.reads == reads + 1
    assert np.linalg.norm(arm._pose_target[:3] - arm._last_state["tcp_pose"][:3]) == pytest.approx(0.05)
    np.testing.assert_allclose(arm._pose_target[3:], QUAT)


def test_max_delta_joint_clips_each_joint_around_the_measured_state():
    arm = connected_arm(ArmSafety(max_delta_joint_rad=0.1), CountingFakeArm)
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    arm.read_state()
    reads = arm.reads
    arm.servo_joints(np.array([1.0, -1.0, 0.05, 0, 0, 0, 0]))
    assert arm.reads == reads
    np.testing.assert_allclose(arm._joint_target, [0.1, -0.1, 0.05, 0, 0, 0, 0])
    arm._last_state_time -= 2 * arm.MAX_DELTA_STATE_AGE_S
    arm.servo_joints(np.ones(7))
    assert arm.reads == reads + 1
    current = arm._last_state["joint_pos"]
    np.testing.assert_allclose(arm._joint_target, current + 0.1)


def test_step_limit_without_max_delta_never_reads():
    arm = connected_arm(ArmSafety(safety_box=np.array(BOX)), CountingFakeArm)
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.servo_cartesian(np.array([0.4, 0.0, 0.3, *QUAT]))
    assert arm.reads == 0


def test_step_then_box_counts_one_clip():
    arm = connected_arm(ArmSafety(safety_box=np.array(BOX), max_delta_xyz_m=0.05))
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.servo_cartesian(np.array([1.0, 0.0, 0.3, *QUAT]))
    assert arm.health().safety_clips == 1


def test_from_config_passes_the_guard():
    robot = safe_franka(max_delta_per_step={"joint_rad": 0.2})
    assert robot.arm.safety_limits == {
        "safety_box": BOX,
        "joint_limits": JOINT_LIMITS,
        "max_delta_per_step": {"joint_rad": 0.2},
    }


def test_joint_limit_rows_must_match_num_joints():
    with pytest.raises(ValueError, match="joint_limits has 7 rows"):
        FakeArm(num_joints=6, safety=ArmSafety(joint_limits=np.array(JOINT_LIMITS)))


# ------------------------------------------------------------------ daemon


def test_service_rejects_unsanitized_commands_before_the_backend(services):
    service = services()
    service.dispatch("robot.connect", {})
    arm = service.robot.arm
    service.dispatch("arm.set_control_mode", {"mode": "cartesian_servo"})
    arm.call_log.clear()
    with pytest.raises(ValueError, match="non-finite"):
        service.dispatch("arm.servo_cartesian", {"pose": [0.4, np.nan, 0.3, *QUAT]})
    with pytest.raises(ValueError, match="quaternion norm"):
        service.dispatch("arm.servo_cartesian", {"pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5]})
    assert arm.call_log == []
    assert service.last_action_age_s is None
    with pytest.raises(SafetyViolation):
        service.dispatch("arm.move_to_joints", {"q": [2.0] * 7})
    assert "move_to_joints" not in arm.call_log
    assert service.dispatch("robot.health", {})["arm"]["safety_rejects"] == 1


def test_service_step_is_clipped(services):
    service = services()
    service.dispatch("robot.connect", {})
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    service.dispatch("robot.step", {"action": {"arm": {"joint_pos": np.full(7, 3.0)}}})
    np.testing.assert_allclose(service.robot.arm._joint_target, 1.0)
    assert service.dispatch("robot.health", {})["arm"]["safety_clips"] == 1


def test_rejected_commands_do_not_feed_the_watchdog(services):
    service = services(watchdog_s=0.1)
    service.dispatch("robot.connect", {})
    service.dispatch("arm.set_control_mode", {"mode": "joint_servo"})
    deadline = time.monotonic() + 0.4
    while time.monotonic() < deadline:
        for method, kwargs in (
            ("arm.move_to_joints", {"q": [2.0] * 7}),
            ("arm.servo_joints", {"q": [np.nan] * 7}),
        ):
            with pytest.raises((SafetyViolation, ValueError, DeviceFault)):
                service.dispatch(method, kwargs)
        time.sleep(0.01)
    assert service.watchdog_trips == 1
    assert service.dispatch("arm.control_mode", {}) == "idle"
    with pytest.raises(DeviceFault, match="watchdog"):
        service.dispatch("arm.servo_joints", {"q": [0.0] * 7})


def test_describe_exposes_safety_and_home(services):
    configured = services().dispatch("robot.describe", {})["arm"]
    assert configured["safety"] == {
        "safety_box": BOX,
        "joint_limits": JOINT_LIMITS,
        "max_delta_per_step": None,
    }
    assert configured["home_joints"] == [0.0] * 7
    plain = FrankaRobot(FrankaConfig(arm_backend="fake", gripper=None))
    unconfigured = services(plain).dispatch("robot.describe", {})["arm"]
    assert unconfigured["safety"] is None
    assert unconfigured["home_joints"] == plain.config.home_joints


# ------------------------------------------------------------------ remote clients


def test_remote_arm_reads_limits_and_gets_safety_violation(services):
    service = services(watchdog_s=30.0)
    robot = RemoteRobot.from_client(LocalClient(service))
    robot.connect()
    try:
        assert robot.arm.safety_limits == service.robot.arm.safety_limits
        assert robot.arm.home_joints == [0.0] * 7
        with pytest.raises(SafetyViolation) as info:
            robot.arm.move_to_joints(np.full(7, 2.0))
        assert "joint_limits" in info.value.message and info.value.hint
        robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
        robot.arm.servo_joints(np.full(7, 2.0))
        np.testing.assert_allclose(service.robot.arm._joint_target, 1.0)
        health = robot.arm.health()
        assert health.safety_clips == 1 and health.safety_rejects == 1
        service.robot.arm.call_log.clear()
        with pytest.raises(ValueError, match="non-finite"):
            robot.arm.servo_joints(np.full(7, np.nan))
        assert service.robot.arm.call_log == []
    finally:
        robot.disconnect()


def test_remote_arm_with_old_describe_has_no_limits(services):
    service = services(watchdog_s=30.0)
    describe = service._methods["robot.describe"]

    def old_describe():
        info = describe()
        info["arm"].pop("safety")
        info["arm"].pop("home_joints")
        return info

    service._methods["robot.describe"] = old_describe
    robot = RemoteRobot.from_client(LocalClient(service))
    robot.connect()
    try:
        assert robot.arm.safety_limits is None
        assert robot.arm.home_joints is None
    finally:
        robot.disconnect()


def test_grpc_round_trip_of_limits_and_safety_violation(services):
    pytest.importorskip("grpc")
    from robot_infra.server.transport import GrpcTransport

    service = services(watchdog_s=30.0)
    transport = GrpcTransport()
    transport.start(service, "127.0.0.1:0")
    robot = Robot.from_dict({"type": "remote", "address": f"127.0.0.1:{transport.port}", "timeout_s": 2.0})
    try:
        robot.connect()
        assert robot.arm.safety_limits == service.robot.arm.safety_limits
        assert robot.arm.home_joints == [0.0] * 7
        with pytest.raises(SafetyViolation, match="safety_box"):
            robot.arm.move_to_pose(np.array([0.4, 0.0, 0.9, *QUAT]))
    finally:
        robot.disconnect()
        transport.stop(grace_s=0.1)


# ------------------------------------------------------------------ config


def test_config_validation_of_limits():
    with pytest.raises(ValueError, match="zmin"):
        FrankaConfig(safety_box=[0.2, -0.3, 0.6, 0.6, 0.3, 0.5])
    with pytest.raises(ValueError, match="7 \\[lo, hi\\] pairs"):
        FrankaConfig(home_joints=None, joint_limits=[[-1.0, 1.0]] * 6)
    with pytest.raises(ValueError, match="pairs"):
        FrankaConfig(home_joints=None, joint_limits=[[-1.0, 0.0, 1.0]] * 7)
    with pytest.raises(ValueError, match=r"joint_limits\[2\] lo"):
        FrankaConfig(home_joints=None, joint_limits=[[-1.0, 1.0]] * 2 + [[1.0, -1.0]] + [[-1.0, 1.0]] * 4)
    with pytest.raises(ValueError, match="unknown max_delta_per_step key 'rpy_rad'"):
        FrankaConfig(max_delta_per_step={"rpy_rad": 0.1})
    for bad in (0.0, -0.1, None, True, "0.1"):
        with pytest.raises(ValueError, match="positive"):
            FrankaConfig(max_delta_per_step={"xyz_m": bad})
    with pytest.raises(ValueError, match=r"home_joints\[3\] = -2.356"):
        FrankaConfig(joint_limits=JOINT_LIMITS)  # default Franka home is outside these limits
    with pytest.raises(ValueError, match="unknown key 'joint_limit'"):
        Robot.from_dict({"type": "franka", "joint_limit": JOINT_LIMITS})


def _uncomment_safety_keys(text: str) -> str:
    return re.sub(r"^# ((safety_box|joint_limits|max_delta_per_step):|  - )", r"\1", text, flags=re.M)


@pytest.mark.parametrize("name, config_cls", [("franka", FrankaConfig), ("piper", PiperConfig)])
def test_example_yaml_loads_with_and_without_limits(tmp_path, name, config_cls):
    source = DOCS / f"robot.{name}.example.yaml"
    assert isinstance(Robot.from_yaml(source).config, config_cls)
    enabled = tmp_path / source.name
    enabled.write_text(_uncomment_safety_keys(source.read_text()))
    config = Robot.from_yaml(enabled).config
    assert config.safety_box is not None
    assert len(config.joint_limits) == config.NUM_JOINTS
    assert set(config.max_delta_per_step) == {"xyz_m", "joint_rad"}


def test_arm_health_accepts_an_old_daemon_dict():
    old = dataclasses.asdict(ArmHealth(ok=True))
    del old["safety_clips"], old["safety_rejects"]
    health = ArmHealth(**old)
    assert health.safety_clips == 0 and health.safety_rejects == 0


def test_failed_move_still_forgets_the_cached_state(monkeypatch):
    arm = connected_arm(ArmSafety(max_delta_joint_rad=0.1), CountingFakeArm)
    arm.read_state()
    assert arm._last_state is not None

    def fail(q, blocking=True):
        raise DeviceFault("move reflected", device=arm.name)

    monkeypatch.setattr(arm, "_move_to_joints", fail)
    with pytest.raises(DeviceFault):
        arm.move_to_joints(np.zeros(7))
    assert arm._last_state is None

    monkeypatch.undo()
    arm.set_control_mode(ControlMode.JOINT_SERVO)
    reads = arm.reads
    arm.servo_joints(np.zeros(7))
    assert arm.reads == reads + 1

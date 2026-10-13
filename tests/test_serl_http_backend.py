"""serl_http backends against a loopback fake of the Flask bridge (no ROS, no network)."""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import robot_infra.devices.backends.arms.serl_http as arm_module
import robot_infra.devices.backends.grippers.serl_http  # noqa: F401  (registers the gripper)
from robot_infra.devices import Arm, ControlMode, DeviceFault, DeviceNotFound, Gripper
from robot_infra.devices.backends.arms.serl_http import SerlHttpArm
from robot_infra.devices.backends.grippers.serl_http import SerlHttpGripper
from robot_infra.devices.robots.franka import FrankaConfig

HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self.fail_with_500 = False
        self.freeze_pose = False
        self.pose = [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
        self.gripper_pos = 0.25
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                payload = json.loads(raw) if raw else None
                bridge.calls.append((self.path, payload))
                if bridge.fail_with_500:
                    self.send_response(500)
                    self.end_headers()
                    return
                if self.path == "/getstate":
                    body = json.dumps(
                        {
                            "pose": bridge.pose,
                            "vel": [0.0] * 6,
                            "force": [1.0, 2.0, 3.0],
                            "torque": [4.0, 5.0, 6.0],
                            "gripper_pos": bridge.gripper_pos,
                        }
                    ).encode()
                    content_type = "application/json"
                else:
                    if self.path == "/pose" and not bridge.freeze_pose:
                        bridge.pose = payload["arr"]
                    body = b"ok"
                    content_type = "text/html"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def bridge():
    fake = FakeBridge()
    yield fake
    fake.close()


@pytest.fixture
def arm(bridge):
    arm = SerlHttpArm(bridge_url=bridge.url, home_joints=HOME)
    arm.connect()
    bridge.calls.clear()
    yield arm
    arm.disconnect()


def closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_registry_and_from_config():
    assert Arm.backend("serl_http") is SerlHttpArm
    assert Gripper.backend("serl_http") is SerlHttpGripper
    config = FrankaConfig(arm_backend="serl_http", bridge_url="http://10.0.0.5:5001")
    assert FrankaConfig().bridge_url == "http://127.0.0.1:5000"
    arm = SerlHttpArm.from_config(config)
    assert arm._bridge_url == "http://10.0.0.5:5001"
    assert np.allclose(arm._home_joints, HOME)
    gripper = SerlHttpGripper.from_config(config)
    assert gripper._bridge_url == "http://10.0.0.5:5001"
    assert gripper.max_width_m == 0.08


def test_connect_validates_bridge_and_reports_feedback(bridge):
    arm = SerlHttpArm(bridge_url=bridge.url)
    arm.connect()
    assert bridge.paths() == ["/getstate"]
    assert arm.health().first_feedback_received is False
    arm.read_state()
    assert arm.health().first_feedback_received is True
    arm.disconnect()


def test_refused_connection_is_device_not_found():
    arm = SerlHttpArm(bridge_url=closed_port_url())
    with pytest.raises(DeviceNotFound) as info:
        arm.connect()
    assert "franka_bridge" in info.value.hint
    assert not arm.is_connected


def test_http_500_is_recoverable_fault(bridge):
    bridge.fail_with_500 = True
    arm = SerlHttpArm(bridge_url=bridge.url)
    with pytest.raises(DeviceFault) as info:
        arm.connect()
    assert info.value.recoverable
    assert "no robot state" in info.value.message


def test_timeout_is_fault_mentioning_timeout_s(monkeypatch):
    def slow(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(arm_module.urllib.request, "urlopen", slow)
    arm = SerlHttpArm(bridge_url="http://127.0.0.1:1", timeout_s=0.5)
    with pytest.raises(DeviceFault) as info:
        arm.connect()
    assert "timeout_s" in info.value.hint


def test_read_state_shapes_and_gripper_cache(arm):
    state = arm.read_state()
    assert state["tcp_pose"].shape == (7,)
    assert state["tcp_vel"].shape == (6,)
    assert state["tcp_force"].shape == (3,)
    assert state["tcp_torque"].shape == (3,)
    assert all(value.dtype == np.float64 for value in state.values())
    assert arm.last_gripper_pos == 0.25
    observation = arm.get_observation()
    assert set(observation) == set(SerlHttpArm.SUPPORTED_STATE_FIELDS)
    assert observation["tcp_pose"].dtype == np.float32


def test_cartesian_servo_lifecycle(arm, bridge):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    pose = np.array([0.4, 0.1, 0.2, 0.0, 1.0, 0.0, 0.0])
    arm.send_action({"tcp_pose": pose})
    arm.set_control_mode(ControlMode.IDLE)
    # fresh connect: impedance state unknown, so entering servo restarts it first
    assert bridge.paths() == ["/stopimp", "/startimp", "/pose", "/stopimp"]
    assert bridge.calls[2][1] == {"arr": pose.tolist()}


def test_joint_control_not_implemented(arm):
    with pytest.raises(NotImplementedError, match="franky"):
        arm.servo_joints(np.zeros(7))
    with pytest.raises(NotImplementedError, match="franky"):
        arm.set_control_mode(ControlMode.JOINT_SERVO)


def test_servo_cartesian_requires_mode(arm):
    with pytest.raises(DeviceFault):
        arm.servo_cartesian(np.zeros(7))


def test_move_to_joints_home_resets_via_bridge(arm, bridge):
    arm.move_to_joints(np.array(HOME))
    assert bridge.paths() == ["/jointreset"]


def test_move_to_joints_non_home_rejected(arm, bridge):
    with pytest.raises(NotImplementedError, match="configured joint target"):
        arm.move_to_joints(np.full(7, 0.3))
    assert bridge.calls == []


def test_move_to_joints_in_servo_mode_is_fault(arm):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    with pytest.raises(DeviceFault):
        arm.move_to_joints(np.array(HOME))


def test_move_to_pose_blocking_converges_and_leaves_controller_running(arm, bridge):
    target = np.array([0.45, 0.05, 0.25, 1.0, 0.0, 0.0, 0.0])
    arm.move_to_joints(np.array(HOME))  # /jointreset leaves impedance known-running
    bridge.calls.clear()
    arm.move_to_pose(target, blocking=True)
    assert bridge.paths()[0] == "/pose"
    assert "/startimp" not in bridge.paths() and "/stopimp" not in bridge.paths()
    assert np.allclose(arm.read_state()["tcp_pose"], target)


def test_move_to_pose_restores_stopped_controller(arm, bridge):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.set_control_mode(ControlMode.IDLE)
    bridge.calls.clear()
    arm.move_to_pose(np.array([0.45, 0.05, 0.25, 1.0, 0.0, 0.0, 0.0]))
    assert bridge.paths()[0] == "/startimp"
    assert bridge.paths()[-1] == "/stopimp"


def test_move_to_pose_times_out_recoverably(arm, bridge, monkeypatch):
    monkeypatch.setattr(arm_module, "_MOVE_TIMEOUT_S", 0.1)
    bridge.freeze_pose = True
    with pytest.raises(DeviceFault) as info:
        arm.move_to_pose(np.array([0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0]))
    assert info.value.recoverable


def test_stop_and_recover_post_nothing(arm, bridge):
    arm.stop()
    arm.recover()
    assert bridge.calls == []


def test_disconnect_posts_nothing(arm, bridge):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    bridge.calls.clear()
    arm.disconnect()
    assert bridge.calls == []


def test_preflight_is_informational(arm):
    (result,) = arm.preflight()
    assert (result.layer, result.name, result.ok) == (5, "mode:serl_http", True)


def test_gripper_open_close_width(bridge):
    gripper = SerlHttpGripper(bridge_url=bridge.url)
    gripper.connect()
    assert gripper.width == 0.25
    assert gripper.get_observation()["gripper_pos"][0] == pytest.approx(0.25)
    bridge.calls.clear()
    gripper.open()
    gripper.close()
    assert bridge.paths() == ["/open_gripper", "/close_gripper"]
    with pytest.raises(NotImplementedError, match="binary"):
        gripper.move(0.5)
    bridge.calls.clear()
    gripper.disconnect()
    assert bridge.calls == []


def test_gripper_refused_connection():
    with pytest.raises(DeviceNotFound):
        SerlHttpGripper(bridge_url=closed_port_url()).connect()


def test_fresh_connect_servo_restarts_impedance_then_poses(arm, bridge):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert bridge.paths() == ["/stopimp", "/startimp"]
    arm.send_action({"tcp_pose": np.array([0.4, 0.1, 0.2, 0.0, 1.0, 0.0, 0.0])})
    assert bridge.paths() == ["/stopimp", "/startimp", "/pose"]


def test_servo_after_idle_only_starts_impedance(arm, bridge):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.set_control_mode(ControlMode.IDLE)
    bridge.calls.clear()
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert bridge.paths() == ["/startimp"]


def test_fresh_connect_idle_stops_impedance_once(arm, bridge):
    # set_control_mode(IDLE) is a no-op while already IDLE, so call the hook directly
    arm._enter_control_mode(ControlMode.IDLE)
    assert bridge.paths() == ["/stopimp"]
    arm._enter_control_mode(ControlMode.IDLE)
    assert bridge.paths() == ["/stopimp"]


def test_joint_reset_leaves_impedance_running(arm, bridge):
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    arm.set_control_mode(ControlMode.IDLE)
    arm.move_to_joints(np.array(HOME))
    bridge.calls.clear()
    arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
    assert bridge.calls == []


def test_frozen_bridge_state_does_not_renew_feedback(arm, bridge):
    arm.read_state()
    first = arm._last_feedback_time
    arm.read_state()
    assert arm._last_feedback_time == first
    bridge.pose = [0.51, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
    arm.read_state()
    assert arm._last_feedback_time > first


def test_gripper_from_config_rejects_borrowed_session(arm):
    config = FrankaConfig(arm_backend="serl_http", bridge_url=arm._bridge_url)
    with pytest.raises(ValueError, match="shares_arm_session: false"):
        SerlHttpGripper.from_config(config, owner=arm)

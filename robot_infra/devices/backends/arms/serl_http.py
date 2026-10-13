"""Legacy Franka backend: HTTP client of the ROS1 ``serl_franka_controllers`` Flask bridge."""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_infra.devices.arm import Arm, ControlMode
from robot_infra.devices.errors import DeviceFault, DeviceNotFound

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig
    from robot_infra.doctor.checks import CheckResult

_BRIDGE_LAUNCH_HINT = (
    "start it on the robot PC: python -m robot_infra.controller.real.franka_bridge "
    "--robot_ip <ip> --port <port>"
)
# /startimp, /stopimp and /jointreset sleep for several seconds inside the bridge.
SLOW_ENDPOINT_TIMEOUT_S = 90.0
_POSE_TOLERANCE_M = 0.005
_MOVE_TIMEOUT_S = 10.0
_POLL_PERIOD_S = 0.02
_NO_JOINT_CONTROL = "serl_http has no joint-space control; use arm_backend: franky"


def _post(
    bridge_url: str,
    path: str,
    payload: dict | None = None,
    timeout_s: float = 5.0,
    *,
    device: str = "",
) -> Any:
    """POST JSON to the bridge; return the decoded JSON body (or text) and map failures to device errors."""
    url = bridge_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload if payload is not None else {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        if error.code == 500:
            raise DeviceFault(
                "bridge has no robot state yet",
                hint="check the ROS controller is running and the robot is reachable",
                device=device,
                recoverable=True,
            ) from None
        raise DeviceFault(
            f"bridge returned HTTP {error.code} for {path}",
            hint="check the franka_bridge log on the robot PC",
            device=device,
            recoverable=True,
        ) from None
    except (TimeoutError, socket.timeout):
        raise _timeout_fault(path, timeout_s, device) from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise _timeout_fault(path, timeout_s, device) from None
        raise DeviceNotFound(
            f"no Franka bridge at {bridge_url}", hint=_BRIDGE_LAUNCH_HINT, device=device
        ) from None
    except ConnectionError:
        raise DeviceNotFound(
            f"no Franka bridge at {bridge_url}", hint=_BRIDGE_LAUNCH_HINT, device=device
        ) from None
    try:
        return json.loads(body)
    except ValueError:
        return body


def _timeout_fault(path: str, timeout_s: float, device: str) -> DeviceFault:
    return DeviceFault(
        f"bridge request {path} timed out after {timeout_s}s",
        hint=f"the bridge is unresponsive; raise timeout_s (now {timeout_s}) or check the robot PC",
        device=device,
        recoverable=True,
    )


class _BridgeHandle:
    def __init__(self, bridge_url: str, timeout_s: float, device: str) -> None:
        self.bridge_url = bridge_url
        self.timeout_s = timeout_s
        self._device = device

    def post(self, path: str, payload: dict | None = None, *, slow: bool = False) -> Any:
        timeout_s = max(self.timeout_s, SLOW_ENDPOINT_TIMEOUT_S) if slow else self.timeout_s
        return _post(self.bridge_url, path, payload, timeout_s, device=self._device)


@Arm.register("serl_http")
class SerlHttpArm(Arm):
    """Legacy backend: forwards to the ROS1 serl_franka_controllers Flask bridge. No joint-space control.

    ``move_to_pose`` is implemented with the bridge's ``/pose`` (an equilibrium-pose
    setpoint), which only works while the impedance controller runs, so it starts it
    if needed, optionally polls ``/getstate`` until the TCP is within 5 mm (10 s limit,
    then a recoverable ``DeviceFault``), and restores a known-stopped controller state.
    Whether the controller runs is tracked in ``_impedance_running`` (``None`` = unknown).
    """

    SDK = None
    SUPPORTED_STATE_FIELDS = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque")

    def __init__(
        self,
        name: str = "arm",
        *,
        bridge_url: str,
        timeout_s: float = 5.0,
        home_joints: list[float] | None = None,
    ) -> None:
        super().__init__(name, num_joints=7)
        self._bridge_url = bridge_url
        self._timeout_s = timeout_s
        self._home_joints = None if home_joints is None else np.asarray(home_joints, dtype=np.float64)
        # None until resolved: the bridge's /getstate does not report whether impedance runs.
        self._impedance_running: bool | None = None
        self._last_state_sample: tuple | None = None
        self.last_gripper_pos: float | None = None

    @classmethod
    def from_config(cls, config: "RobotConfig", *, name: str = "arm") -> "SerlHttpArm":
        return cls(name, bridge_url=config.bridge_url, home_joints=config.home_joints)

    def _open(self) -> _BridgeHandle:
        handle = _BridgeHandle(self._bridge_url, self._timeout_s, self.name)
        handle.post("/getstate")
        self._impedance_running = None
        self._last_state_sample = None
        return handle

    def _release(self, handle: _BridgeHandle) -> None:
        # Stateless HTTP; the bridge keeps running, so never /stopimp here.
        pass

    def read_state(self) -> dict[str, np.ndarray]:
        self._require_connected()
        state = self._handle.post("/getstate")
        self.last_gripper_pos = float(state["gripper_pos"])
        self._note_feedback_if_new(state)
        return {
            "tcp_pose": np.asarray(state["pose"], dtype=np.float64),
            "tcp_vel": np.asarray(state["vel"], dtype=np.float64),
            "tcp_force": np.asarray(state["force"], dtype=np.float64),
            "tcp_torque": np.asarray(state["torque"], dtype=np.float64),
        }

    def _note_feedback_if_new(self, state: dict) -> None:
        """The bridge serves its last cached ROS state with no timestamp (``stamp`` is used if
        a bridge adds one), so a sample counts as fresh feedback only when it differs from the
        previous one. A live arm's wrench estimate is never bit-identical between samples; a
        frozen cache (controller stopped or dead) is, and then the feedback age keeps growing."""
        if "stamp" in state:
            sample: tuple = (state["stamp"],)
        else:
            sample = tuple(
                float(v) for key in ("pose", "vel", "force", "torque") for v in state[key]
            )
        if sample != self._last_state_sample:
            self._last_state_sample = sample
            self._note_feedback()

    def servo_cartesian(self, pose: np.ndarray) -> None:
        self._require_mode(ControlMode.CARTESIAN_SERVO)
        self._handle.post("/pose", {"arr": self._pose7(pose).tolist()})

    def servo_joints(self, q: np.ndarray) -> None:
        raise NotImplementedError(_NO_JOINT_CONTROL)

    def move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self._require_mode(ControlMode.IDLE)
        target = self._pose7(pose)
        was_running = self._impedance_running is True
        self._ensure_impedance_running()
        try:
            self._handle.post("/pose", {"arr": target.tolist()})
            if blocking:
                self._wait_until_at(target)
        finally:
            if not was_running:
                self._stop_impedance()

    def move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self._require_mode(ControlMode.IDLE)
        q = np.asarray(q, dtype=np.float64)
        if self._home_joints is not None and not np.allclose(q, self._home_joints, atol=1e-3):
            raise NotImplementedError(
                "serl_http can only reset to the bridge's configured joint target"
            )
        # The bridge blocks until its own --reset_joint_target is reached, then restarts impedance.
        self._handle.post("/jointreset", slow=True)
        self._impedance_running = True

    def stop(self) -> None:
        # The impedance controller holds the last equilibrium pose; power stays on.
        pass

    def recover(self) -> None:
        # The bridge exposes no error-recovery endpoint (/startimp and /jointreset clear errors themselves).
        self._clear_errors()

    def _enter_control_mode(self, mode: ControlMode) -> None:
        """Drive the bridge's impedance controller to match ``mode``.

        ``/startimp`` and ``/stopimp`` are not idempotent (a second ``/startimp`` launches a
        duplicate roslaunch; ``/stopimp`` with nothing running errors) and ``/getstate`` does
        not report the controller state. After connect the state is unknown (``None``), so the
        only way to reach a known state is ``/stopimp`` then ``/startimp``; this costs one
        extra ~10 s controller restart per session. ``/jointreset`` leaves it running.
        """
        if mode is ControlMode.CARTESIAN_SERVO:
            self._ensure_impedance_running()
        elif mode is ControlMode.IDLE:
            if self._impedance_running is not False:
                self._stop_impedance()
        else:
            raise NotImplementedError(_NO_JOINT_CONTROL)

    def preflight(self) -> list["CheckResult"]:
        from robot_infra.doctor.checks import CheckResult

        return [
            CheckResult(
                layer=5,
                name="mode:serl_http",
                ok=True,
                message="legacy ROS1 bridge backend; joint control unavailable",
            )
        ]

    def _ensure_impedance_running(self) -> None:
        if self._impedance_running is None:
            self._stop_impedance()
        if self._impedance_running is False:
            self._start_impedance()

    def _start_impedance(self) -> None:
        self._handle.post("/startimp", slow=True)
        self._impedance_running = True

    def _stop_impedance(self) -> None:
        self._handle.post("/stopimp", slow=True)
        self._impedance_running = False

    def _require_mode(self, mode: ControlMode) -> None:
        self._require_connected()
        if self._control_mode is not mode:
            raise DeviceFault(
                f"arm '{self.name}' is in {self._control_mode.value} mode, expected {mode.value}",
                hint=f"call set_control_mode(ControlMode.{mode.name}) first",
                device=self.name,
            )

    @staticmethod
    def _pose7(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"expected pose shape (7,), got {pose.shape}")
        return pose

    def _wait_until_at(self, target: np.ndarray) -> None:
        deadline = time.monotonic() + _MOVE_TIMEOUT_S
        while True:
            error = np.linalg.norm(self.read_state()["tcp_pose"][:3] - target[:3])
            if error < _POSE_TOLERANCE_M:
                return
            if time.monotonic() > deadline:
                raise DeviceFault(
                    f"move_to_pose did not reach the target within {_MOVE_TIMEOUT_S}s "
                    f"(position error {error * 1000:.1f} mm)",
                    hint="check the target is reachable and the impedance controller is running",
                    device=self.name,
                    recoverable=True,
                )
            time.sleep(_POLL_PERIOD_S)

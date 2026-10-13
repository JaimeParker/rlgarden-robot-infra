"""Client-side mirror of a robot served by the daemon: ``RemoteRobot`` builds its parts from
the remote ``robot.describe`` and forwards every device call through a ``TransportClient``."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_infra.devices.arm import Arm, ArmHealth, ControlMode
from robot_infra.devices.base import Connection, Features, Health, Part, PartGroup
from robot_infra.devices.camera import CameraConfig, Frame
from robot_infra.devices.errors import DeviceError
from robot_infra.devices.gripper import Gripper, GripperMode
from robot_infra.devices.robot import GripperConfig, Robot, RobotConfig
from robot_infra.server.transport.base import TransportClient

if TYPE_CHECKING:
    from robot_infra.doctor.checks import CheckResult

logger = logging.getLogger(__name__)

_TRANSPORTS = ("grpc", "local")
_DEFAULT_BINARY_THRESHOLD = 0.5


@dataclass
class RemoteRobotConfig(RobotConfig):
    """``arm_backend``/``gripper``/``cameras``/``home_joints`` are ignored: the parts come
    from the daemon's ``robot.describe``."""

    TYPE = "remote"
    NUM_JOINTS = 0
    name: str = "remote"
    gripper: GripperConfig | None = None
    address: str = "127.0.0.1:50051"
    token: str | None = None
    timeout_s: float = 5.0
    reconnect_attempts: int = 5
    reconnect_backoff_s: float = 0.5
    transport: str = "grpc"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transport not in _TRANSPORTS:
            raise ValueError(f"transport '{self.transport}' invalid; choose from {list(_TRANSPORTS)}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")


class _RemotePart:
    """Forwards ``<prefix>.<method>`` calls. Remote parts always borrow the ``RemoteRobot``
    session, so their own ``_open``/``_release`` must never run."""

    _client: TransportClient
    _method_prefix: str

    def _call(self, method: str, **kwargs: Any) -> Any:
        self._require_connected()
        return self._client.call(f"{self._method_prefix}.{method}", **kwargs)

    def _open(self) -> Any:
        raise RuntimeError(f"{type(self).__name__} borrows its RemoteRobot session; connect the robot")

    def _release(self, handle: Any) -> None:
        raise RuntimeError(f"{type(self).__name__} borrows its RemoteRobot session; disconnect the robot")


class RemoteArm(_RemotePart, Arm):
    _method_prefix = "arm"

    def __init__(
        self,
        name: str,
        *,
        num_joints: int,
        supported_state_fields: tuple[str, ...],
        client: TransportClient,
        owner: Connection,
        control_mode: ControlMode = ControlMode.IDLE,
    ) -> None:
        super().__init__(name, num_joints=num_joints, owner=owner)
        # Instance attribute shadows the Arm ClassVar: the field set is only known per remote robot.
        self.SUPPORTED_STATE_FIELDS = tuple(supported_state_fields)
        self._client = client
        self._control_mode = control_mode

    def read_state(self) -> dict[str, np.ndarray]:
        return self._call("read_state")

    def servo_cartesian(self, pose: np.ndarray) -> None:
        self._call("servo_cartesian", pose=np.asarray(pose))

    def servo_joints(self, q: np.ndarray) -> None:
        self._call("servo_joints", q=np.asarray(q))

    def move_to_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        self._call("move_to_pose", pose=np.asarray(pose), blocking=blocking)

    def move_to_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        self._call("move_to_joints", q=np.asarray(q), blocking=blocking)

    def stop(self) -> None:
        self._call("stop")

    def recover(self) -> None:
        self._call("recover")

    def set_control_mode(self, mode: ControlMode) -> None:
        # One round trip: the daemon's arm stops itself before switching. No early return on an
        # unchanged local mode, because the daemon watchdog may have idled the arm meanwhile.
        self._enter_control_mode(mode)
        self._control_mode = mode

    def _enter_control_mode(self, mode: ControlMode) -> None:
        self._call("set_control_mode", mode=mode.value)

    def health(self) -> ArmHealth:
        if not self.is_connected:
            return ArmHealth(ok=False, detail="not connected", control_mode=self._control_mode.value)
        health = ArmHealth(**self._call("health"))
        self._control_mode = ControlMode(health.control_mode)
        return health

    def preflight(self) -> list["CheckResult"]:
        from robot_infra.doctor.checks import CheckResult

        return [
            CheckResult(**{key: value for key, value in entry.items() if key != "has_autofix"})
            for entry in self._call("preflight")
        ]


class RemoteGripper(_RemotePart, Gripper):
    _method_prefix = "gripper"

    def __init__(
        self,
        name: str,
        *,
        max_width_m: float,
        mode: GripperMode,
        binary_threshold: float,
        client: TransportClient,
        owner: Connection,
    ) -> None:
        super().__init__(
            name, max_width_m=max_width_m, mode=mode, binary_threshold=binary_threshold, owner=owner
        )
        self._client = client

    def open(self) -> None:
        self._call("open")

    def close(self) -> None:
        self._call("close")

    def move(self, width_norm: float) -> None:
        self._call("move", width=float(width_norm))

    @property
    def width(self) -> float:
        return float(self._call("width"))


class RemoteCamera(_RemotePart, Part):
    """Camera-shaped ``Part``, not a ``Camera``: ``Camera`` owns a capture thread through
    ``@final`` ``_open``/``_release`` and never borrows a session, while frames here come from
    the daemon's capture thread one ``camera.<name>.read`` call at a time."""

    def __init__(self, config: CameraConfig, *, client: TransportClient, owner: Connection) -> None:
        super().__init__(config.name, owner=owner)
        self._config = config
        self._client = client
        self._method_prefix = f"camera.{config.name}"
        self._latest: Frame | None = None

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def observation_features(self) -> Features:
        c = self._config
        features: Features = {"color": {"shape": (c.height, c.width, 3), "dtype": "uint8"}}
        if c.enable_depth:
            features["depth"] = {"shape": (c.height, c.width), "dtype": "float32"}
        return features

    def get_observation(self) -> dict[str, np.ndarray]:
        frame = self.read()
        observation = {"color": frame.color}
        if self._config.enable_depth:
            observation["depth"] = frame.depth
        return observation

    def read(self, timeout_s: float | None = None) -> Frame:
        reply = self._call("read", timeout_s=timeout_s)
        self._latest = Frame(color=reply["color"], depth=reply["depth"], timestamp=reply["timestamp"])
        return self._latest

    def latest(self) -> Frame | None:
        """Last frame this client read; no round trip."""
        return self._latest

    def is_ready(self, timeout_s: float = 2.0) -> bool:
        try:
            self.read(timeout_s)
        except DeviceError:
            return False
        return True

    def health(self) -> Health:
        if not self.is_connected:
            return Health(ok=False, detail="not connected")
        return Health(**self._client.call("robot.health")["cameras"][self.name])


@Robot.register_type
class RemoteRobot(Robot):
    ROBOT_TYPE = "remote"
    CONFIG = RemoteRobotConfig

    def __init__(self, config: RemoteRobotConfig, *, client: TransportClient | None = None) -> None:
        self._client = client
        super().__init__(config)

    @classmethod
    def from_client(cls, client: TransportClient, *, name: str = "remote") -> "RemoteRobot":
        return cls(RemoteRobotConfig(name=name, address="local", transport="local"), client=client)

    @property
    def client(self) -> TransportClient | None:
        return self._client

    def build(self) -> None:
        """No network and no transport import: the gRPC client is created at connect time."""
        if self._client is None and self.config.transport == "local":
            raise ValueError(
                "transport 'local' needs an in-process client; use RemoteRobot.from_client(LocalClient(service))"
            )

    def _ensure_client(self) -> TransportClient:
        if self._client is None:
            from robot_infra.server.transport.grpc_transport import GrpcClient

            cfg = self.config
            self._client = GrpcClient(
                cfg.address,
                token=cfg.token,
                timeout_s=cfg.timeout_s,
                reconnect_attempts=cfg.reconnect_attempts,
                reconnect_backoff_s=cfg.reconnect_backoff_s,
            )
        return self._client

    def _open(self) -> dict:
        client = self._ensure_client()
        self.children.clear()
        client.open()
        remote_connected = False
        try:
            info = client.call("robot.describe")
            self._build_remote_parts(info, client)
            client.call("robot.connect")
            remote_connected = True
            # Publish the handle before the children borrow it: a borrowing child's connect()
            # calls owner.connect(), which would otherwise re-enter _open.
            self._handle = info
            for child in self.children.values():
                child.connect()
        except BaseException:
            self._handle = None
            self.children.clear()
            try:
                if remote_connected:
                    client.call("robot.disconnect")
            except DeviceError as error:
                logger.warning("remote robot.disconnect after a failed connect failed: %s", error)
            finally:
                client.close()
            raise
        return info

    def _build_remote_parts(self, info: dict, client: TransportClient) -> None:
        arm_info = info["arm"]
        self.add(
            "arm",
            RemoteArm(
                "arm",
                num_joints=arm_info["num_joints"],
                supported_state_fields=tuple(arm_info["supported_state_fields"]),
                client=client,
                owner=self,
                control_mode=ControlMode(arm_info["control_mode"]),
            ),
        )
        gripper_info = info["gripper"]
        if gripper_info:
            threshold = gripper_info["binary_threshold"]
            self.add(
                "gripper",
                RemoteGripper(
                    "gripper",
                    max_width_m=gripper_info["max_width_m"],
                    mode=GripperMode(gripper_info["mode"]),
                    binary_threshold=_DEFAULT_BINARY_THRESHOLD if threshold is None else threshold,
                    client=client,
                    owner=self,
                ),
            )
        if info["cameras"]:
            group = PartGroup("cameras")
            for name, camera_cfg in info["cameras"].items():
                group.add(name, RemoteCamera(CameraConfig(**camera_cfg), client=client, owner=self))
            self.add("cameras", group)

    def _release(self, handle: Any) -> None:
        client = self._client
        try:
            self._disconnect_children_reversed()
            try:
                client.call("robot.disconnect")
            except DeviceError as error:
                logger.warning("remote robot.disconnect at %s failed: %s", self.config.address, error)
        finally:
            client.close()
            self.children.clear()

    @property
    def cameras(self) -> dict[str, RemoteCamera]:
        return super().cameras

    def describe_text(self) -> str:
        if not self.is_connected:
            return f"{self.name} (remote @ {self.config.address}) — not connected"
        return super().describe_text()

    def get_observation(self) -> dict[str, np.ndarray]:
        """One round trip instead of one per part."""
        self._require_connected()
        return self._client.call("robot.get_observation")

    def step(self, action: dict) -> dict:
        """Send ``action`` and return the next observation in one round trip."""
        self._require_connected()
        return self._client.call("robot.step", action=action)

    def remote_health(self) -> dict:
        self._require_connected()
        return self._client.call("robot.health")

    def health(self) -> Health:
        if not self.is_connected:
            return Health(ok=False, detail=f"not connected to {self.config.address}")
        remote = self.remote_health()
        rtt = self._client.last_rtt_ms
        rtt_text = "n/a" if rtt is None else f"{rtt:.1f}"
        return Health(
            ok=bool(remote["connected"] and remote["arm"]["ok"]),
            detail=f"rtt={rtt_text} ms, watchdog_trips={remote['watchdog_trips']}",
        )

"""Robot daemon service layer: wire codec, transport-neutral ``RobotService``, transports and
the client-side ``RemoteRobot`` (importing this package registers robot type "remote")."""
from robot_infra.server.codec import encode_jpeg, pack, unpack
from robot_infra.server.service import METHODS, RobotService
from robot_infra.server.remote import (
    RemoteArm,
    RemoteCamera,
    RemoteGripper,
    RemoteRobot,
    RemoteRobotConfig,
)

__all__ = [
    "METHODS",
    "RemoteArm",
    "RemoteCamera",
    "RemoteGripper",
    "RemoteRobot",
    "RemoteRobotConfig",
    "RobotService",
    "encode_jpeg",
    "pack",
    "unpack",
]

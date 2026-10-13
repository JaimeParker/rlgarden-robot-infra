"""Device layer: typed errors, connection/part abstractions, backend registry, claims."""
from robot_infra.devices.arm import (
    ARM_STATE_FIELDS,
    Arm,
    ArmHealth,
    ControlMode,
    arm_state_shape,
)
from robot_infra.devices.base import (
    Connection,
    ControllablePart,
    Features,
    FeatureSpec,
    Health,
    Part,
    PartGroup,
)
from robot_infra.devices.camera import Camera, CameraConfig, Frame
from robot_infra.devices.claims import DeviceClaim
from robot_infra.devices.errors import (
    DeviceAlreadyConnected,
    DeviceBusy,
    DeviceError,
    DeviceFault,
    DeviceNotConnected,
    DeviceNotFound,
    SdkMissing,
    VersionMismatch,
)
from robot_infra.devices.gripper import Gripper, GripperMode
from robot_infra.devices.registry import BackendRegistry
from robot_infra.devices.robot import GripperConfig, Robot, RobotConfig

__all__ = [
    "ARM_STATE_FIELDS",
    "Arm",
    "ArmHealth",
    "Camera",
    "CameraConfig",
    "ControlMode",
    "Frame",
    "Gripper",
    "GripperConfig",
    "GripperMode",
    "arm_state_shape",
    "BackendRegistry",
    "Connection",
    "ControllablePart",
    "DeviceAlreadyConnected",
    "DeviceBusy",
    "DeviceClaim",
    "DeviceError",
    "DeviceFault",
    "DeviceNotConnected",
    "DeviceNotFound",
    "FeatureSpec",
    "Features",
    "Health",
    "Part",
    "PartGroup",
    "Robot",
    "RobotConfig",
    "SdkMissing",
    "VersionMismatch",
]

from robot_infra.devices import backends as _backends  # noqa: E402,F401
from robot_infra.devices import robots as _robots  # noqa: E402,F401

"""Importing this package registers every SDK-free backend."""
from robot_infra.devices.backends.arms import fake as _fake_arm  # noqa: F401
from robot_infra.devices.backends.cameras import fake as _fake_camera  # noqa: F401
from robot_infra.devices.backends.grippers import fake as _fake_gripper  # noqa: F401
from robot_infra.devices.backends.cameras import realsense as _realsense_camera  # noqa: F401  (SDK import is lazy)

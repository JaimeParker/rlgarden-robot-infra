"""Importing this package registers every SDK-free backend."""
from robot_infra.devices.backends.arms import fake as _fake_arm  # noqa: F401
from robot_infra.devices.backends.cameras import fake as _fake_camera  # noqa: F401
from robot_infra.devices.backends.grippers import fake as _fake_gripper  # noqa: F401
from robot_infra.devices.backends.cameras import realsense as _realsense_camera  # noqa: F401  (SDK import is lazy)
from robot_infra.devices.backends.arms import franky as _franky_arm  # noqa: F401  (SDK import is lazy)
from robot_infra.devices.backends.grippers import franky as _franky_gripper  # noqa: F401
from robot_infra.devices.backends.arms import serl_http as _serl_http_arm  # noqa: F401
from robot_infra.devices.backends.grippers import serl_http as _serl_http_gripper  # noqa: F401
from robot_infra.devices.backends.arms import pyagxarm as _pyagxarm_arm  # noqa: F401  (SDK import is lazy)
from robot_infra.devices.backends.grippers import pyagxarm as _pyagxarm_gripper  # noqa: F401
from robot_infra.devices.backends.arms import piper_sdk as _piper_sdk_arm  # noqa: F401  (SDK import is lazy)
from robot_infra.devices.backends.grippers import piper_sdk as _piper_sdk_gripper  # noqa: F401

"""Importing this package registers every robot type (no vendor SDK is imported)."""
from robot_infra.devices.robots import franka as _franka  # noqa: F401
from robot_infra.devices.robots import piper as _piper  # noqa: F401

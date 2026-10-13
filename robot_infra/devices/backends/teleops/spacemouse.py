"""3Dconnexion SpaceMouse through pyspacemouse 2.x: the puck drives a cartesian twist, button 0
toggles the (sticky) gripper target and button 1 raises ``episode_end``."""
from __future__ import annotations

import glob
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np

from robot_infra.devices.errors import DeviceNotFound, SdkMissing, VersionMismatch
from robot_infra.devices.teleop import RawReading, Teleop, TeleopConfig

PYSPACEMOUSE_REQUIREMENT = "pyspacemouse>=2.0,<3"
HIDRAW_GLOB = "/dev/hidraw*"
# Unverified on hardware: 3Dconnexion is 0x256f, older units enumerate under Logitech 0x046d.
SPACEMOUSE_VENDOR_IDS = (0x256F, 0x046D)
GRIPPER_BUTTON = 0
EPISODE_END_BUTTON = 1
_OPEN_GRIPPER, _CLOSED_GRIPPER = 1.0, 0.0
_NUM_BUTTONS = 2
_OPTION_KEYS = {"device_index", "deadzone", "invert_axes", "poll_interval_s"}
_sleep = time.sleep  # module-level so tests can observe the poll cadence
_SDK_HINT = f"pip install '{PYSPACEMOUSE_REQUIREMENT}' (or: pip install 'rlgarden-robot-infra[spacemouse]')"


@Teleop.register("spacemouse")
class SpaceMouseTeleop(Teleop):
    """Options: ``device_index`` (default 0), ``deadzone`` (default 1e-3, twist norm below which
    the puck counts as idle), ``invert_axes`` (indices 0-5 of the output twist whose sign is
    flipped), ``poll_interval_s`` (default 0.005 = 200 Hz; device poll cadence so a non-blocking
    ``read()`` does not spin a core, may be 0 once ``read()`` is confirmed to block). The twist is the raw puck reading in the robot base frame (axis order
    ``[-y, x, z, -roll, -pitch, -yaw]`` of the pyspacemouse state), as in HIL-SERL."""

    SDK = "pyspacemouse"
    ACTION_SPACE = "cartesian_delta"
    HAS_GRIPPER = True

    @classmethod
    def require_sdk(cls) -> ModuleType:
        try:
            return super().require_sdk()
        except SdkMissing as error:
            raise SdkMissing(error.message, hint=_SDK_HINT, device=error.device) from None

    def __init__(self, config: TeleopConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        options = dict(config.options)
        for key in options:
            if key not in _OPTION_KEYS:
                raise ValueError(f"unknown spacemouse option '{key}'; known: {sorted(_OPTION_KEYS)}")
        self._device_index = int(options.get("device_index", 0))
        self._deadzone = float(options.get("deadzone", 1e-3))
        if self._deadzone < 0:
            raise ValueError(f"deadzone must be >= 0, got {self._deadzone}")
        self._poll_interval_s = float(options.get("poll_interval_s", 0.005))
        if self._poll_interval_s < 0:
            raise ValueError(f"poll_interval_s must be >= 0, got {self._poll_interval_s}")
        invert_axes = [int(axis) for axis in options.get("invert_axes", [])]
        for axis in invert_axes:
            if not 0 <= axis < self.action_dim:
                raise ValueError(f"invert_axes entries must be in [0, {self.action_dim}), got {axis}")
        self._axis_signs = np.ones(self.action_dim, dtype=np.float32)
        self._axis_signs[invert_axes] = -1.0
        self._state_lock = threading.Lock()
        self._gripper = _OPEN_GRIPPER
        self._previous_buttons = [False] * _NUM_BUTTONS

    @classmethod
    def discover(cls) -> set[str]:
        """Device ids: ``vid:pid:path`` via hidapi when installed, else every ``/dev/hidraw*``
        (which also lists non-SpaceMouse HID devices)."""
        try:
            import hid
        except ImportError:
            return set(glob.glob(HIDRAW_GLOB))
        found = set()
        for info in hid.enumerate():
            if info["vendor_id"] in SPACEMOUSE_VENDOR_IDS:
                path = info["path"]
                path = path.decode(errors="replace") if isinstance(path, bytes) else str(path)
                found.add(f"{info['vendor_id']:04x}:{info['product_id']:04x}:{path}")
        return found

    # -- device ----------------------------------------------------------------

    def _open_device(self) -> Any:
        sdk = self.require_sdk()
        mismatch = VersionMismatch(
            f"pyspacemouse {getattr(sdk, '__version__', '?')} lacks the 2.x API "
            "(open(device_index=...) returning a device with read())",
            hint=f"pip install '{PYSPACEMOUSE_REQUIREMENT}'",
            device=self.name,
        )
        if not callable(getattr(sdk, "open", None)):
            raise mismatch
        try:
            device = sdk.open(device_index=self._device_index)
        except Exception as error:
            raise DeviceNotFound(
                f"cannot open SpaceMouse #{self._device_index}: {error}",
                hint="plug it in and check /dev/hidraw* permissions (`python -m robot_infra.doctor`)",
                device=self.name,
            ) from error
        if device is None:
            raise DeviceNotFound(
                f"pyspacemouse found no SpaceMouse #{self._device_index}",
                hint="plug it in and check /dev/hidraw* permissions (`python -m robot_infra.doctor`)",
                device=self.name,
            )
        if not callable(getattr(device, "read", None)):
            raise mismatch
        with self._state_lock:
            self._reset_state()
        return device

    def _close_device(self, handle: Any) -> None:
        # The poll thread is already stopped. The HID handle must be released, otherwise the
        # next open() cannot get it.
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    def _reset_state(self) -> None:
        self._gripper = _OPEN_GRIPPER
        self._previous_buttons = [False] * _NUM_BUTTONS

    def _reset_device(self, handle: Any) -> None:
        with self._state_lock:
            self._reset_state()

    def _poll_device(self, handle: Any) -> RawReading:
        _sleep(self._poll_interval_s)
        state = handle.read()
        twist = self._axis_signs * np.array(
            [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw], dtype=np.float32
        )
        pressed = [bool(b) for b in list(state.buttons)[:_NUM_BUTTONS]]
        pressed += [False] * (_NUM_BUTTONS - len(pressed))
        with self._state_lock:
            rising = [now and not before for now, before in zip(pressed, self._previous_buttons)]
            self._previous_buttons = pressed
            if rising[GRIPPER_BUTTON]:
                self._gripper = _CLOSED_GRIPPER if self._gripper == _OPEN_GRIPPER else _OPEN_GRIPPER
            gripper = self._gripper
        return RawReading(
            action=twist,
            gripper=gripper,
            buttons={f"button_{index}": value for index, value in enumerate(pressed)},
            events_now={"episode_end": rising[EPISODE_END_BUTTON]},
            active=bool(np.linalg.norm(twist) > self._deadzone or any(pressed)),
        )

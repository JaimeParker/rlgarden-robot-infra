"""Intel RealSense camera backend (pyrealsense2, imported lazily)."""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_infra.devices.camera import Camera, Frame
from robot_infra.devices.errors import DeviceFault, DeviceNotFound

_WARMUP_FRAMES = 5


@dataclass
class _Handle:
    pipeline: Any
    align: Any
    depth_scale: float | None
    serial: str


@Camera.register("realsense")
class RealSenseCamera(Camera):
    SDK = "pyrealsense2"

    @classmethod
    def discover(cls) -> set[str]:
        try:
            rs = importlib.import_module(cls.SDK)
        except ImportError:
            return set()
        return {
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        }

    def _open_device(self) -> _Handle:
        rs = self.require_sdk()
        serial = self._resolve_serial(self.discover())
        c = self.config
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(serial)
        rs_config.enable_stream(rs.stream.color, c.width, c.height, rs.format.rgb8, c.fps)
        if c.enable_depth:
            rs_config.enable_stream(rs.stream.depth, c.width, c.height, rs.format.z16, c.fps)
        try:
            profile = pipeline.start(rs_config)
            depth_scale = None
            align = None
            if c.enable_depth:
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                align = rs.align(rs.stream.color)
            for _ in range(_WARMUP_FRAMES):
                pipeline.wait_for_frames(timeout_ms=self._frame_timeout_ms())
        except Exception as error:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise DeviceFault(
                f"RealSense '{serial}' failed to start: {error}",
                hint="check the USB 3 cable/port and udev rules "
                "(librealsense scripts/setup_udev_rules.sh)",
                device=self.name,
                recoverable=True,
            ) from error
        return _Handle(pipeline, align, depth_scale, serial)

    def _resolve_serial(self, found: set[str]) -> str:
        serial = self.config.serial
        if serial:
            if serial not in found:
                raise DeviceNotFound(
                    f"RealSense serial '{serial}' for camera '{self.name}' not attached",
                    hint=f"discovered: {sorted(found)}; run `python -m robot_infra.discovery`",
                    device=self.name,
                )
            return serial
        if len(found) == 1:
            return next(iter(found))
        if not found:
            raise DeviceNotFound(
                f"no RealSense device found for camera '{self.name}'",
                hint="plug in the camera and run `python -m robot_infra.discovery`",
                device=self.name,
            )
        raise DeviceNotFound(
            f"camera '{self.name}' has no serial and {len(found)} RealSense devices are attached",
            hint=f"set serial to one of {sorted(found)}",
            device=self.name,
        )

    def _capture_frame(self, handle: _Handle) -> Frame:
        frames = handle.pipeline.wait_for_frames(timeout_ms=self._frame_timeout_ms())
        if handle.align is not None:
            frames = handle.align.process(frames)
        # Copy: a view would pin the librealsense frame and starve its small frame pool.
        color = np.array(frames.get_color_frame().get_data(), copy=True)
        depth = None
        if handle.depth_scale is not None:
            raw_depth = np.asanyarray(frames.get_depth_frame().get_data())
            depth = raw_depth.astype(np.float32) * handle.depth_scale
        return Frame(color=color, depth=depth, timestamp=time.monotonic())

    def _close_device(self, handle: _Handle) -> None:
        try:
            handle.pipeline.stop()
        except Exception:
            pass

    def _frame_timeout_ms(self) -> int:
        return int(2000 / self.config.fps) + 1000

"""Hardware-free camera producing frames whose pixel [0, 0] encodes a frame counter."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from robot_infra.devices.camera import Camera, CameraConfig, Frame


@Camera.register("fake")
class FakeCamera(Camera):
    def __init__(self, config: CameraConfig, *, fail_after: int | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._fail_after = fail_after
        self._counter = 0

    @classmethod
    def discover(cls) -> set[str]:
        return {"FAKE0", "FAKE1"}

    def _open_device(self) -> Any:
        self._counter = 0
        return {"opened": True}

    def _close_device(self, handle: Any) -> None:
        pass

    def _capture_frame(self, handle: Any) -> Frame:
        time.sleep(1.0 / self._config.fps)
        self._counter += 1
        if self._fail_after is not None and self._counter > self._fail_after:
            raise RuntimeError("fake capture failure")
        c = self._config
        color = np.zeros((c.height, c.width, 3), dtype=np.uint8)
        color[0, 0, 0] = self._counter % 256
        color[0, 0, 1] = self._counter // 256 % 256
        depth = np.ones((c.height, c.width), dtype=np.float32) if c.enable_depth else None
        return Frame(color=color, depth=depth, timestamp=time.monotonic())

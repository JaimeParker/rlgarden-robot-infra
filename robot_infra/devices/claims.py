"""Advisory, node-local exclusive claims on device endpoints."""
from __future__ import annotations

import re
from pathlib import Path
from types import TracebackType

import filelock

from robot_infra.devices.errors import DeviceBusy

_DEFAULT_LOCK_DIR = Path.home() / ".cache" / "robot_infra" / "claims"


class DeviceClaim:
    """Advisory, node-local exclusive claim on a device endpoint (filelock, timeout=0)."""

    def __init__(self, key: str, lock_dir: Path | None = None) -> None:
        self.key = key
        lock_dir = Path(lock_dir) if lock_dir is not None else _DEFAULT_LOCK_DIR
        lock_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        # thread_local=False: a daemon acquires and releases from different gRPC worker threads.
        self._lock = filelock.FileLock(str(lock_dir / f"{slug}.lock"), timeout=0, thread_local=False)

    def acquire(self) -> None:
        try:
            self._lock.acquire()
        except filelock.Timeout:
            raise DeviceBusy(
                f"device '{self.key}' is claimed by another process",
                hint=f"stop the other process using '{self.key}' or run with claim=False",
                device=self.key,
            ) from None

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "DeviceClaim":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

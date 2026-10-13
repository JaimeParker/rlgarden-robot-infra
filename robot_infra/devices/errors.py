"""Typed device errors that carry an actionable hint and survive a dict round trip
(so a remote device server can ship them to the client and re-raise the same type)."""
from __future__ import annotations

from typing import Any, ClassVar, Mapping


class DeviceError(RuntimeError):
    kind: ClassVar[str] = "device_error"
    _KINDS: ClassVar[dict[str, type["DeviceError"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        DeviceError._KINDS[cls.kind] = cls

    def __init__(self, message: str, *, hint: str = "", device: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.device = device

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "message": self.message,
            "hint": self.hint,
            "device": self.device,
        }

    @staticmethod
    def from_dict(d: Mapping[str, str]) -> "DeviceError":
        error_cls = DeviceError._KINDS.get(d.get("kind", ""), DeviceError)
        return error_cls(
            d.get("message", ""),
            hint=d.get("hint", ""),
            device=d.get("device", ""),
            **error_cls._extra_fields_from_dict(d),
        )

    @classmethod
    def _extra_fields_from_dict(cls, d: Mapping[str, str]) -> dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return f"{self.message}\n  hint: {self.hint}" if self.hint else self.message


DeviceError._KINDS[DeviceError.kind] = DeviceError


class SdkMissing(DeviceError):
    kind = "sdk_missing"


class DeviceNotFound(DeviceError):
    kind = "device_not_found"


class DeviceNotConnected(DeviceError):
    kind = "device_not_connected"


class DeviceAlreadyConnected(DeviceError):
    kind = "device_already_connected"


class DeviceBusy(DeviceError):
    kind = "device_busy"


class DeviceFault(DeviceError):
    kind = "device_fault"

    def __init__(
        self, message: str, *, hint: str = "", device: str = "", recoverable: bool = False
    ) -> None:
        super().__init__(message, hint=hint, device=device)
        self.recoverable = recoverable

    def to_dict(self) -> dict[str, str]:
        return {**super().to_dict(), "recoverable": "1" if self.recoverable else "0"}

    @classmethod
    def _extra_fields_from_dict(cls, d: Mapping[str, str]) -> dict[str, Any]:
        return {"recoverable": d.get("recoverable") == "1"}


class VersionMismatch(DeviceError):
    kind = "version_mismatch"

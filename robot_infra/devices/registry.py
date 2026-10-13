"""Name -> class registry mixin shared by device category roots (Arm/Gripper/Camera)."""
from __future__ import annotations

from typing import Callable, ClassVar


class BackendRegistry:
    """Each category root defines its own ``_BACKENDS: ClassVar[dict[str, type]] = {}``;
    subclasses of the root resolve that same dict through the MRO."""

    _BACKENDS: ClassVar[dict[str, type]]

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        key = name.lower()

        def decorator(backend_cls: type) -> type:
            if key in cls._BACKENDS:
                raise ValueError(f"{cls.__name__} backend '{key}' is already registered")
            cls._BACKENDS[key] = backend_cls
            return backend_cls

        return decorator

    @classmethod
    def backend(cls, name: str) -> type:
        try:
            return cls._BACKENDS[name.lower()]
        except KeyError:
            raise ValueError(
                f"unknown {cls.__name__} backend '{name}'; registered: {sorted(cls._BACKENDS)}"
            ) from None

    @classmethod
    def backends(cls) -> dict[str, type]:
        return dict(cls._BACKENDS)

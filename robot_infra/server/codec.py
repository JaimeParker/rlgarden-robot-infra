"""msgpack wire codec shared by every transport: ndarrays, numpy scalars, enums and
typed ``DeviceError``s survive a round trip; nested dicts/lists are walked recursively."""
from __future__ import annotations

import io
from enum import Enum
from typing import Any

import msgpack
import numpy as np

from robot_infra.devices.errors import DeviceError, SdkMissing

_NDARRAY_TAG = "__nd__"
_ERROR_TAG = "__err__"
_PASSTHROUGH = (str, int, float, bool, bytes, type(None))


def pack(obj: Any) -> bytes:
    return msgpack.packb(_encode(obj), use_bin_type=True)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_decode_map)


def encode_jpeg(image: np.ndarray, quality: int = 90) -> dict:
    """Wire form of a uint8 HxWx3 RGB image as JPEG bytes; ``unpack`` restores the ndarray."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"encode_jpeg expects a uint8 (H, W, 3) array, got {image.dtype} {image.shape}")
    return {
        _NDARRAY_TAG: True,
        "jpeg": True,
        "shape": list(image.shape),
        "dtype": "uint8",
        "data": _jpeg_bytes(np.ascontiguousarray(image), quality),
    }


def _encode(obj: Any) -> Any:
    if isinstance(obj, DeviceError):
        return {_ERROR_TAG: obj.to_dict()}
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        return {
            _NDARRAY_TAG: True,
            "shape": list(obj.shape),  # ascontiguousarray promotes a 0-d array to 1-d
            "dtype": str(array.dtype),
            "data": array.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Enum):
        return _encode(obj.value)
    if isinstance(obj, _PASSTHROUGH):
        return obj
    if isinstance(obj, dict):
        return {_encode(key): _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(item) for item in obj]
    raise TypeError(f"cannot encode {type(obj).__name__} for the wire; convert it to a dict/list/ndarray first")


def _decode_map(obj: dict) -> Any:
    if obj.get(_NDARRAY_TAG) is True:
        shape = tuple(obj["shape"])
        if obj.get("jpeg"):
            return _decode_jpeg(obj["data"]).reshape(shape)
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(shape).copy()
    if _ERROR_TAG in obj and len(obj) == 1:
        return DeviceError.from_dict(obj[_ERROR_TAG])
    return obj


def _jpeg_bytes(image: np.ndarray, quality: int) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    cv2 = _import_cv2()
    ok, encoded = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cv2.imencode failed to encode the JPEG frame")
    return encoded.tobytes()


def _decode_jpeg(data: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    cv2 = _import_cv2()
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(bgr[..., ::-1])


def require_jpeg_decoder() -> None:
    """Fail fast with the install hint: a decode error inside a gRPC response deserializer
    surfaces as an opaque INTERNAL TransportError."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        _import_cv2()


def _import_cv2():
    try:
        import cv2
    except ImportError:
        raise SdkMissing(
            "JPEG frames need Pillow or OpenCV",
            hint='pip install "rlgarden-robot-infra[jpeg]"',
            device="codec",
        ) from None
    return cv2

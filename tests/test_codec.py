"""Wire codec round trips: ndarrays, nesting, typed DeviceErrors, enums, optional JPEG."""
from __future__ import annotations

import numpy as np
import pytest

from robot_infra.devices import ControlMode, DeviceError, DeviceFault, DeviceNotFound
from robot_infra.server.codec import encode_jpeg, pack, unpack


def round_trip(obj):
    return unpack(pack(obj))


@pytest.mark.parametrize(
    "array",
    [
        np.arange(7, dtype=np.float32),
        np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3),
        np.array(42, dtype=np.int64),
        np.arange(20, dtype=np.float64).reshape(4, 5)[:, ::2],
    ],
    ids=["float32", "uint8_hwc", "int64_scalar", "non_contiguous"],
)
def test_ndarray_round_trip(array):
    restored = round_trip(array)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == array.dtype and restored.shape == array.shape
    np.testing.assert_array_equal(restored, array)
    assert restored.flags.writeable


def test_nested_containers_and_scalars():
    obj = {
        "arm": {"joint_pos": np.ones(7, dtype=np.float32), "mode": ControlMode.JOINT_SERVO},
        "list": [np.float32(1.5), np.int64(3), np.bool_(True), (1, 2)],
        "none": None,
        "flag": False,
        "raw": b"\x00\x01",
    }
    restored = round_trip(obj)
    np.testing.assert_array_equal(restored["arm"]["joint_pos"], np.ones(7, dtype=np.float32))
    assert restored["arm"]["mode"] == "joint_servo"
    assert restored["list"] == [1.5, 3, True, [1, 2]]
    assert type(restored["list"][1]) is int and type(restored["list"][2]) is bool
    assert restored["none"] is None and restored["flag"] is False
    assert restored["raw"] == b"\x00\x01"


def test_device_errors_round_trip_typed():
    fault = round_trip({"error": DeviceFault("boom", hint="call recover()", device="arm", recoverable=True)})
    error = fault["error"]
    assert type(error) is DeviceFault
    assert (error.message, error.hint, error.device, error.recoverable) == (
        "boom",
        "call recover()",
        "arm",
        True,
    )
    missing = round_trip(DeviceNotFound("no gripper", hint="this robot has no gripper"))
    assert type(missing) is DeviceNotFound and missing.hint == "this robot has no gripper"
    assert type(round_trip(DeviceError("plain"))) is DeviceError


def test_unsupported_type_names_it():
    with pytest.raises(TypeError, match="object"):
        pack({"x": object()})


def test_jpeg_round_trip():
    pytest.importorskip("PIL")
    # Smooth gray ramp: JPEG's default 4:2:0 chroma subsampling would smear per-pixel color noise.
    ramp = np.add.outer(np.linspace(60, 130, 8), np.linspace(0, 70, 8))
    image = np.repeat(ramp[..., None], 3, axis=-1).astype(np.uint8)
    wire = encode_jpeg(image, quality=95)
    assert wire["jpeg"] is True and isinstance(wire["data"], bytes)
    restored = round_trip({"color": wire})["color"]
    assert restored.shape == (8, 8, 3) and restored.dtype == np.uint8
    assert np.abs(restored.astype(int) - image.astype(int)).max() <= 10


def test_jpeg_rejects_non_rgb_uint8():
    with pytest.raises(ValueError, match="uint8"):
        encode_jpeg(np.zeros((8, 8), dtype=np.uint8))

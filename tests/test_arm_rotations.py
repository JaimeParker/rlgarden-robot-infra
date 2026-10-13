"""rpy <-> quaternion helpers: round trip, identity, and agreement with scipy's
extrinsic "xyz" convention when scipy is installed."""
from __future__ import annotations

import numpy as np
import pytest

from robot_infra.devices.backends.arms._rotations import (
    quat_xyzw_to_rpy,
    rpy_to_matrix,
    rpy_to_quat_xyzw,
)


def test_identity():
    assert np.allclose(rpy_to_quat_xyzw(np.zeros(3)), [0, 0, 0, 1])
    assert np.allclose(quat_xyzw_to_rpy(np.array([0, 0, 0, 1.0])), 0)


def test_round_trip_away_from_gimbal_lock():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = rng.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi])
        back = quat_xyzw_to_rpy(rpy_to_quat_xyzw(rpy))
        assert np.allclose(rpy_to_matrix(back), rpy_to_matrix(rpy), atol=1e-9)


def test_pure_yaw_quaternion():
    quat = rpy_to_quat_xyzw(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(quat, [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])


def test_matches_scipy_extrinsic_xyz():
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(1)
    for _ in range(20):
        rpy = rng.uniform(-np.pi, np.pi, size=3)
        expected = scipy_rotation.from_euler("xyz", rpy).as_quat()
        got = rpy_to_quat_xyzw(rpy)
        assert np.allclose(got, expected, atol=1e-9) or np.allclose(got, -expected, atol=1e-9)

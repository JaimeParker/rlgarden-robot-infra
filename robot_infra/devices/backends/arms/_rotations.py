"""Rotation helpers shared by arm backends whose SDKs report Euler angles.

Convention: ``rpy`` is extrinsic x-y-z (roll about fixed X, then pitch about
fixed Y, then yaw about fixed Z), i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``,
matching ``scipy.spatial.transform.Rotation.from_euler("xyz", rpy)``.
Quaternions are ``[x, y, z, w]``.
"""
from __future__ import annotations

import numpy as np


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in np.asarray(rpy, dtype=np.float64))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    pitch = np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: fold roll into yaw
        roll = 0.0
        yaw = np.arctan2(-m[0, 1], m[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    return matrix_to_quat_xyzw(rpy_to_matrix(rpy))


def quat_xyzw_to_rpy(quat: np.ndarray) -> np.ndarray:
    return matrix_to_rpy(quat_xyzw_to_matrix(quat))

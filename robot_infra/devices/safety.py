"""Device-side safety limits for an arm: a base-frame TCP position box, joint limits and
per-step motion caps. Pure numpy, no SDK dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from robot_infra.devices.errors import SafetyViolation

if TYPE_CHECKING:
    from robot_infra.devices.robot import RobotConfig

_AXES = ("x", "y", "z")
_LIMITS_HINT = "change the target, or the limits in the robot yaml on the robot host"


@dataclass(frozen=True)
class ArmSafety:
    """``safety_box`` is ``[xmin, ymin, zmin, xmax, ymax, zmax]`` in the frame of
    ``read_state()["tcp_pose"]``; ``joint_limits`` is ``(num_joints, 2)`` in rad."""

    safety_box: np.ndarray | None = None
    joint_limits: np.ndarray | None = None
    max_delta_xyz_m: float | None = None
    max_delta_joint_rad: float | None = None

    def __post_init__(self) -> None:
        if self.safety_box is not None:
            box = np.asarray(self.safety_box, dtype=np.float64)
            if box.shape != (6,) or np.any(box[:3] > box[3:]):
                raise ValueError(f"safety_box must be [xmin, ymin, zmin, xmax, ymax, zmax], got {box.tolist()}")
            object.__setattr__(self, "safety_box", box)
        if self.joint_limits is not None:
            limits = np.asarray(self.joint_limits, dtype=np.float64)
            if limits.ndim != 2 or limits.shape[1] != 2 or np.any(limits[:, 0] > limits[:, 1]):
                raise ValueError(f"joint_limits must be [[lo, hi], ...] with lo <= hi, got {limits.tolist()}")
            object.__setattr__(self, "joint_limits", limits)

    @classmethod
    def from_config(cls, config: "RobotConfig") -> "ArmSafety | None":
        max_delta = config.max_delta_per_step or {}
        safety = cls(
            safety_box=config.safety_box,
            joint_limits=config.joint_limits,
            max_delta_xyz_m=max_delta.get("xyz_m"),
            max_delta_joint_rad=max_delta.get("joint_rad"),
        )
        return None if safety.is_unlimited else safety

    @property
    def is_unlimited(self) -> bool:
        return (
            self.safety_box is None
            and self.joint_limits is None
            and self.max_delta_xyz_m is None
            and self.max_delta_joint_rad is None
        )

    def to_dict(self) -> dict[str, Any]:
        max_delta = {
            key: float(value)
            for key, value in (("xyz_m", self.max_delta_xyz_m), ("joint_rad", self.max_delta_joint_rad))
            if value is not None
        }
        return {
            "safety_box": None if self.safety_box is None else self.safety_box.tolist(),
            "joint_limits": None if self.joint_limits is None else self.joint_limits.tolist(),
            "max_delta_per_step": max_delta or None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmSafety":
        max_delta = data.get("max_delta_per_step") or {}
        return cls(
            safety_box=data.get("safety_box"),
            joint_limits=data.get("joint_limits"),
            max_delta_xyz_m=max_delta.get("xyz_m"),
            max_delta_joint_rad=max_delta.get("joint_rad"),
        )

    # -- clipping (servo) ----------------------------------------------------------

    def clip_pose(self, pose: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clip the position into the box; the orientation is left untouched."""
        if self.safety_box is None:
            return pose, False
        xyz = np.clip(pose[:3], self.safety_box[:3], self.safety_box[3:])
        if np.array_equal(xyz, pose[:3]):
            return pose, False
        return np.concatenate([xyz, pose[3:]]), True

    def clip_joints(self, q: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.joint_limits is None:
            return q, False
        clipped = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        if np.array_equal(clipped, q):
            return q, False
        return clipped, True

    def limit_step_pose(self, pose: np.ndarray, current_pose: np.ndarray) -> tuple[np.ndarray, bool]:
        """Shorten the position step to ``max_delta_xyz_m``, keeping its direction and the orientation."""
        if self.max_delta_xyz_m is None:
            return pose, False
        current_xyz = np.asarray(current_pose, dtype=np.float64)[:3]
        step = pose[:3] - current_xyz
        distance = float(np.linalg.norm(step))
        if distance <= self.max_delta_xyz_m:
            return pose, False
        xyz = current_xyz + step * (self.max_delta_xyz_m / distance)
        return np.concatenate([xyz, pose[3:]]), True

    def limit_step_joints(self, q: np.ndarray, current_q: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.max_delta_joint_rad is None:
            return q, False
        current_q = np.asarray(current_q, dtype=np.float64)
        limited = np.clip(q, current_q - self.max_delta_joint_rad, current_q + self.max_delta_joint_rad)
        if np.array_equal(limited, q):
            return q, False
        return limited, True

    # -- rejection (move) ----------------------------------------------------------

    def require_pose_inside(self, pose: np.ndarray, *, device: str = "") -> None:
        if self.safety_box is None:
            return
        low, high = self.safety_box[:3], self.safety_box[3:]
        outside = [
            f"{_AXES[i]}={pose[i]:.4f} not in [{low[i]:.4f}, {high[i]:.4f}]"
            for i in range(3)
            if not low[i] <= pose[i] <= high[i]
        ]
        if outside:
            raise SafetyViolation(
                f"target tcp_pose is outside the safety_box: {', '.join(outside)}",
                hint=_LIMITS_HINT,
                device=device,
            )

    def require_joints_inside(self, q: np.ndarray, *, device: str = "") -> None:
        if self.joint_limits is None:
            return
        outside = [
            f"joint {i}={q[i]:.4f} not in [{lo:.4f}, {hi:.4f}]"
            for i, (lo, hi) in enumerate(self.joint_limits)
            if not lo <= q[i] <= hi
        ]
        if outside:
            raise SafetyViolation(
                f"target joint_pos is outside the joint_limits: {', '.join(outside)}",
                hint=_LIMITS_HINT,
                device=device,
            )

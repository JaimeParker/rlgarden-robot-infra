"""Layered hardware diagnostic: ``python -m robot_infra.doctor --config robot.yaml``."""
from robot_infra.doctor.checks import (
    LAYER_NAMES,
    CheckResult,
    exit_code,
    run_doctor,
    summarize,
)

__all__ = ["LAYER_NAMES", "CheckResult", "exit_code", "run_doctor", "summarize"]

from __future__ import annotations

import argparse
import dataclasses
import json

from robot_infra.discovery.autoconfig import load_config_with_env
from robot_infra.doctor.checks import CheckResult, exit_code, run_doctor, summarize

_LAYER_COUNT = 5


def _status(result: CheckResult) -> str:
    if result.skipped:
        return "SKIP"
    return "OK" if result.ok else "FAIL"


def format_text(results: list[CheckResult]) -> str:
    lines = []
    for result in results:
        lines.append(
            f"[{result.layer}/{_LAYER_COUNT}] {result.name} ... {_status(result)}  {result.message}"
        )
        if _status(result) == "FAIL" and result.hint:
            lines.append(f"    hint: {result.hint}")
    ok, failed, skipped = summarize(results)
    lines.append(f"summary: {ok} ok, {failed} failed, {skipped} skipped")
    return "\n".join(lines)


def format_json(results: list[CheckResult]) -> str:
    return json.dumps(
        [
            {f.name: getattr(result, f.name) for f in dataclasses.fields(result) if f.name != "autofix"}
            for result in results
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot_infra.doctor", description="Layered robot diagnostic")
    parser.add_argument("--config", required=True, help="robot yaml")
    parser.add_argument("--mock", action="store_true", help="swap every backend for its fake")
    parser.add_argument("--fix", action="store_true", help="run autofixes for failed checks")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    config = load_config_with_env(args.config)
    results = run_doctor(config, mock=args.mock, fix=args.fix)
    print(format_json(results) if args.json else format_text(results))
    return exit_code(results)

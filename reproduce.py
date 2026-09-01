#!/usr/bin/env python3
"""Single entry point for quick or full FreshSky-Safe reproduction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*parts: str, cwd: Path = ROOT) -> None:
    command = [sys.executable, *parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def quick() -> None:
    run("verify_claims.py")
    run("analysis/replot_iotj_inference.py")
    run(
        "analysis/figure_generation_tmc_advanced.py",
        "--project-root",
        str(ROOT),
        "--extended-results",
        str(ROOT / "experiments" / "results" / "tmc_extended"),
        "--legacy-results",
        str(ROOT / "experiments" / "results"),
        "--output-dir",
        str(ROOT / "figures"),
        "--derived-dir",
        str(ROOT / "experiments" / "results" / "tmc_extended" / "derived"),
    )
    run("experiments/run_tmc_evidence.py", "--manifest-only")


def full() -> None:
    run("experiments/run_tmc_evidence.py")
    run("analysis/iotj_closed_loop_grid.py")
    run("analysis/iotj_regularized_policy_selection.py")
    run("analysis/iotj_inference_evidence.py")
    run("verify_claims.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick",
        action="store_true",
        help="validate committed outputs and rebuild all empirical figures (default)",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="rerun the simulation suites before validation; CPU intensive",
    )
    args = parser.parse_args()
    if args.full:
        full()
    else:
        quick()
    print("FreshSky-Safe reproduction completed successfully")


if __name__ == "__main__":
    main()

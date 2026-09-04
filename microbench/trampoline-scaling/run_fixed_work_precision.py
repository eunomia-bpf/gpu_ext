#!/usr/bin/env python3
"""Run the frozen high-precision fixed-work trampoline follow-up."""

from pathlib import Path

import run_fixed_work as fixed


runner = fixed.runner
PRECISION_BLOCKS = 48
PRECISION_WARMUP = 16
PRECISION_LAUNCHES = 512
PRECISION_HOOK_REPEATS = 16


def configure() -> None:
    """Select the precision profile without changing the compiled kernel."""
    fixed.configure()
    runner.EXPERIMENT_KIND = "RTX 5090 fixed-work trampoline precision"
    runner.SUMMARY_TITLE = "RTX 5090 fixed-work trampoline precision"
    runner.PREFLIGHT_CELL_IDS = (2,)
    runner.FULL_CELL_IDS = tuple(cell["id"] for cell in fixed.FIXED_WORK_CELLS)
    runner.PREFLIGHT_BLOCKS = 1
    runner.FULL_BLOCKS = PRECISION_BLOCKS
    runner.PREFLIGHT_WARMUP = PRECISION_WARMUP
    runner.FULL_WARMUP = PRECISION_WARMUP
    runner.PREFLIGHT_LAUNCHES = PRECISION_LAUNCHES
    runner.FULL_LAUNCHES = PRECISION_LAUNCHES
    runner.PREFLIGHT_HOOK_REPEATS = PRECISION_HOOK_REPEATS
    runner.FULL_HOOK_REPEATS = PRECISION_HOOK_REPEATS
    runner.RANDOMIZE_CELL_ORDER = True
    runner.BALANCE_ARM_ORDER = True
    runner.WRITE_INDEPENDENT_RAW_EVIDENCE = True
    runner.EXTRA_SOURCE_PATHS = (
        Path(fixed.__file__).resolve(),
        Path(__file__).resolve().with_name("analyze_fixed_work.py"),
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("analyze_fixed_work_precision.py"),
        Path(__file__).resolve().with_name("fixed-work-precision-plan.md"),
    )


def main() -> int:
    configure()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())

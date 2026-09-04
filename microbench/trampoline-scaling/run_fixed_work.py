#!/usr/bin/env python3
"""Run the fixed-total-work RTX 5090 block-organization experiment."""

from pathlib import Path

import run_scaling as runner


FIXED_WORK_CELLS = (
    {"id": 0, "blocks": 128, "threads_per_block": 1024,
     "active_threads": 131_072, "counter_key": 0},
    {"id": 1, "blocks": 256, "threads_per_block": 512,
     "active_threads": 131_072, "counter_key": 1},
    {"id": 2, "blocks": 1024, "threads_per_block": 128,
     "active_threads": 131_072, "counter_key": 2},
    {"id": 3, "blocks": 2048, "threads_per_block": 64,
     "active_threads": 131_072, "counter_key": 3},
    {"id": 4, "blocks": 4096, "threads_per_block": 32,
     "active_threads": 131_072, "counter_key": 4},
)


def configure() -> None:
    runner.EXPERIMENT_KIND = "RTX 5090 fixed-work trampoline block organization"
    runner.SUMMARY_TITLE = "RTX 5090 fixed-work trampoline block organization"
    runner.CELLS = FIXED_WORK_CELLS
    runner.PREFLIGHT_CELL_IDS = (2,)
    runner.FULL_CELL_IDS = tuple(cell["id"] for cell in FIXED_WORK_CELLS)
    runner.PREFLIGHT_BLOCKS = 1
    runner.FULL_BLOCKS = 10
    runner.PREFLIGHT_WARMUP = 1
    runner.FULL_WARMUP = 2
    runner.PREFLIGHT_LAUNCHES = 2
    runner.FULL_LAUNCHES = 8
    runner.PREFLIGHT_HOOK_REPEATS = 2
    runner.FULL_HOOK_REPEATS = 16
    runner.RANDOMIZE_CELL_ORDER = True
    runner.BALANCE_ARM_ORDER = True
    runner.WRITE_INDEPENDENT_RAW_EVIDENCE = True
    runner.MAX_THREADS_PER_BLOCK = 1024
    runner.MATRIX_HEADER = runner.HERE / "fixed_work_matrix.h"
    runner.APPLICATION_BINARY = runner.HERE / ".output/fixed-work-scaling"
    runner.COMPILED_PTX = runner.HERE / ".output/fixed-work-scaling.ptx"
    runner.LOADER_BINARY = runner.HERE / ".output/fixed-work-probe"
    runner.BPF_OBJECT_PREFIX = "fixed-work-probe"
    runner.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("analyze_fixed_work.py"),
        Path(__file__).resolve().with_name("fixed-work-plan.md"),
    )


configure()


if __name__ == "__main__":
    raise SystemExit(runner.main())

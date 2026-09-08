#!/usr/bin/env python3
"""Five-block matched-pair transport runner (llama.cpp pp512, transport2 vs 3).

Runs the original per-thread kernelretsnoop probe with
BPFTIME_GPU_AUTO_WARP_EXECUTION=1 on both arms and contrasts the two ringbuf
transport layouts:

  * transport2 -- encoded per-record tail publication (AoS),
  * transport3 -- transposed record words (SoA, mode-2 handshake).

Each requested block runs both arms as a same-block matched pair; the run
order rotates each block so neither transport is always first.
No no-probe baseline and no NVBit replay are produced; the same-block paired
change is the metric. The probe is the prebuilt original kernelretsnoop tool,
so no source preparation, compilation, or tool build happens here. All build
and command plumbing is reused from run_table1_perf.py and
run_revision_rq4.py.

The opt-in loader-exit wait sends SIGINT once after the CUDA client returns
and waits without a kill deadline, so the loader's final ring drain and
post-processing finish before the private shared-memory segment is removed.
Numeric throughput and loader/collector status are retained separately in
each cell record. --dry-run prints the JSON schedule with no GPU work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_table1_perf as table1  # noqa: E402

KIND = "transport_pairs"
ARMS = ("transport2", "transport3")
TRANSPORT_BY_ARM = {
    "transport2": table1.AUTO_WARP_TRANSPORT_ENCODED,
    "transport3": table1.AUTO_WARP_TRANSPORT_TRANSPOSED,
}
SOURCE_CONTRACT = (
    "Same original per-thread probe: encoded AoS versus transposed SoA "
    "transport; both auto-execution env ON; no baseline replay."
)
DEFAULT_PREBUILT_TOOL = (
    HERE / "raw/table1-original-ring-encoded-20260908.iVdbES"
    / "gpubpf_tool_build/kernelretsnoop"
)
DEFAULT_BPFTIME_ROOT = Path("/home/yunwei37/workspace/gpu/bpftime-auto-warp")
DEFAULT_BPFTIME_BUILD_DIR = Path(
    "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575")


def block_modes(block: int) -> tuple[int, int]:
    """Rotate which transport runs first so both share early/late positions."""
    return (2, 3) if block % 2 else (3, 2)


def build_table1_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    """Reuse the existing runner's default pp512/tg0 namespace, then override."""
    argv: list[str] = []
    if cli_args.model is not None:
        argv += ["--model", str(cli_args.model)]
    if cli_args.llama_bench is not None:
        argv += ["--llama-bench", str(cli_args.llama_bench)]
    argv += ["--bpftime-root", str(cli_args.bpftime_root)]
    argv += ["--bpftime-build-dir", str(cli_args.bpftime_build_dir)]
    if cli_args.target_symbol is not None:
        argv += ["--target-symbol", cli_args.target_symbol]
    table1_args = table1.parse_args(argv)
    table1_args.auto_warp_task = "kernelretsnoop"
    table1_args.nvbit_tool = None
    table1_args.wait_for_probe_exit = cli_args.wait_for_probe_exit
    table1_args.timeout_s = cli_args.client_timeout_s
    return table1_args


def dry_run_plan(cli_args: argparse.Namespace) -> dict[str, Any]:
    schedule = {
        str(block): [f"transport{mode}" for mode in block_modes(block)]
        for block in range(1, cli_args.blocks + 1)
    }
    return {
        "dry_run": True,
        "kind": KIND,
        "metric": ("llama.cpp pp512 prefill token/s; same-block "
                   "transport2/transport3 paired change; no no-probe "
                   "baseline, no NVBit replay"),
        "arms": list(ARMS),
        "blocks": cli_args.blocks,
        "cells_per_block": len(ARMS),
        "cell_count": len(ARMS) * cli_args.blocks,
        "tool": "kernelretsnoop (original per-thread object, prebuilt)",
        "prebuilt_tool": str(cli_args.prebuilt_tool),
        "auto_warp_execution": "1 on both arms (helper label auto_warp_on)",
        "transports": {arm: TRANSPORT_BY_ARM[arm] for arm in ARMS},
        "wait_for_probe_exit": cli_args.wait_for_probe_exit,
        "client_timeout_s": cli_args.client_timeout_s,
        "diagnostic_call_count": 0,
        "schedule": schedule,
        "source_contract": SOURCE_CONTRACT,
        "model": (str(cli_args.model) if cli_args.model
                  else "<run_table1_perf default>"),
        "llama_bench": (str(cli_args.llama_bench) if cli_args.llama_bench
                        else "<run_table1_perf default>"),
        "bpftime_root": str(cli_args.bpftime_root),
        "bpftime_build_dir": str(cli_args.bpftime_build_dir),
        "pp": 512,
        "tg": 0,
    }


def run_campaign(cli_args: argparse.Namespace,
                 table1_args: argparse.Namespace) -> int:
    os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (cli_args.output_dir
                  or (HERE / "raw" / f"{KIND}-{timestamp}")).resolve()
    if (output_dir / "cells.json").exists() or any(
            (output_dir / f"pair-{block:02d}" / arm).exists()
            for block in range(1, cli_args.blocks + 1) for arm in ARMS):
        raise SystemExit(f"Refusing to overwrite existing measurements: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table1.SOURCE_CONTRACTS[KIND] = SOURCE_CONTRACT
    tools = {"kernelretsnoop": cli_args.prebuilt_tool}
    cells: list[dict[str, Any]] = []
    for block in range(1, cli_args.blocks + 1):
        for mode in block_modes(block):
            table1_args.auto_warp_transport = mode
            cell_output = output_dir / f"pair-{block:02d}" / f"transport{mode}"
            cell_output.mkdir(parents=True, exist_ok=True)
            print(f"START block={block} transport={mode}", flush=True)
            row: dict[str, Any] = {
                "block": block,
                "arm": f"transport{mode}",
                "cell_root": str(cell_output.relative_to(output_dir)),
            }
            try:
                row.update(table1.run_arm_cell(
                    "auto_warp_on", block, table1_args, cell_output,
                    tools, None))
            except table1.runner.OwnedCleanupError:
                raise
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            cells.append(row)
            table1.write_records(output_dir, cells, ARMS, KIND)
            print(f"DONE block={block} transport={mode} "
                  f"returncode={row.get('returncode')} "
                  f"tok_s={row.get('throughput_tok_s')}", flush=True)
    print(f"wrote {len(cells)} cells under {output_dir}", flush=True)
    return int(any(row.get("returncode") != 0 or row.get("error")
                   or row.get("probe_teardown_error") for row in cells))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="cell root (default: raw/transport-pairs-<ts>)")
    parser.add_argument("--blocks", type=int, default=5,
                        help="number of rotating transport2/transport3 "
                             "matched pairs")
    parser.add_argument("--prebuilt-tool", type=Path,
                        default=DEFAULT_PREBUILT_TOOL,
                        help="directory of the prebuilt original "
                             "kernelretsnoop tool")
    parser.add_argument("--model", type=Path, default=None,
                        help="llama model (default: run_table1_perf default)")
    parser.add_argument("--llama-bench", type=Path, default=None,
                        help="llama-bench binary "
                             "(default: run_table1_perf default)")
    parser.add_argument("--bpftime-root", type=Path,
                        default=DEFAULT_BPFTIME_ROOT)
    parser.add_argument("--bpftime-build-dir", type=Path,
                        default=DEFAULT_BPFTIME_BUILD_DIR)
    parser.add_argument("--target-symbol", default=None,
                        help="target symbol (default: run_table1_perf default)")
    parser.add_argument("--wait-for-probe-exit",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="SIGINT once and wait without a kill deadline "
                             "after the CUDA client returns, so the loader's "
                             "final drain finishes (default on; disable with "
                             "--no-wait-for-probe-exit)")
    parser.add_argument("--client-timeout-s", type=int, default=None,
                        help="llama-bench client timeout seconds; default None "
                             "waits without a deadline")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the JSON schedule and exit without GPU work")
    args = parser.parse_args(argv)
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    for field in ("output_dir", "prebuilt_tool", "model", "llama_bench",
                  "bpftime_root", "bpftime_build_dir"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tool_binary = args.prebuilt_tool / "kernelretsnoop"
    if not args.dry_run and not tool_binary.is_file():
        raise SystemExit(f"prebuilt tool binary not found: {tool_binary}")
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), indent=2), flush=True)
        return 0
    return run_campaign(args, build_table1_args(args))


if __name__ == "__main__":
    raise SystemExit(main())

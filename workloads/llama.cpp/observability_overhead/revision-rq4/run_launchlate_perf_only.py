#!/usr/bin/env python3
"""RTX 5090 Table 1 launchlate performance-only runner (pp512, 10 blocks).

A small companion to run_revision_rq4.py that measures only the Table 1
launchlate performance row: llama.cpp prefill token/s and percent overhead
for baseline, gpubpf launchlate (bpftime-table1-575, branch
revision/table1-575), and the matched NVBit launchlate adapter across ten
randomized complete blocks. It reuses the existing private_probe mechanics,
tool sources, and launch paths from the gated runner, but runs none of the
correctness, engagement, verifier, precision, or source-schema gates and
attempts each cell exactly once. A cell is valid when and only when
llama-bench completes the configured prefill with finite positive token/s;
raw stdout/stderr, the return code, and numeric throughput are always
retained. Probe counter parsing is optional metadata and zero or missing
counters never invalidate, reject, or retry a cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OBS_ROOT = HERE.parent
sys.path.insert(0, str(OBS_ROOT))
import run_observability_overhead as core  # noqa: E402
sys.path.insert(0, str(HERE))
import run_revision_rq4 as gated  # noqa: E402


KIND = "launchlate_perf_only"
CONFIGS = ("baseline", "gpubpf_launchlate", "nvbit_launchlate")
DEFAULT_BPFTIME_ROOT = Path("/home/yunwei37/workspace/gpu/bpftime-table1-575")
DEFAULT_BPFTIME_BUILD_DIR = Path(
    "/home/yunwei37/workspace/gpu/bpftime-table1-575/build-launchlate-575"
)


def benchmark_valid(result: dict[str, Any], pp: int) -> bool:
    """Throughput-only validity; hook counters never reject or retry a cell."""
    if result.get("returncode") != 0:
        return False
    metrics = result.get("metrics") or {}
    if metrics.get("pp_tokens") != pp:
        return False
    try:
        value = float(metrics.get("pp_tok_s", 0.0))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def defining_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kind": KIND,
        "model": str(args.model),
        "llama_bench": str(args.llama_bench),
        "llama_cli": str(args.llama_cli),
        "bpftime_root": str(args.bpftime_root),
        "bpftime_build_dir": str(args.bpftime_build_dir),
        "target_symbol": args.target_symbol,
        "runs": args.runs,
        "pp": args.pp,
        "tg": args.tg,
        "n_gpu_layers": args.n_gpu_layers,
        "timeout_s": args.timeout_s,
        "probe_startup_s": args.probe_startup_s,
        "gpu_thread_count": args.gpu_thread_count,
        "uprobe_binary": str(args.uprobe_binary),
        "uprobe_symbol_hint": args.uprobe_symbol_hint,
        "nvbit_tool": str(args.nvbit_tool) if args.nvbit_tool is not None else None,
        "uvm": args.uvm,
        "no_warmup": args.no_warmup,
        "cuda_graphs_disabled": core.CUDA_GRAPHS_DISABLED,
        "schedule_seed": gated.SCHEDULE_SEED,
        "worker_cpus": gated.CLIENT_CPUS,
        "attempts_per_cell": 1,
    }


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    configs = gated.selected_configs(args)
    if configs != CONFIGS:
        raise ValueError("the performance-only matrix is fixed to the three launchlate arms")
    return {
        "dry_run": True,
        "kind": KIND,
        "metric": "llama.cpp pp{pp} prefill token/s and percent overhead "
                  "versus the same-block baseline".format(pp=args.pp),
        "configs": list(configs),
        "runs": args.runs,
        "pp": args.pp,
        "schedule_seed": gated.SCHEDULE_SEED,
        "timing_schedule": gated.fixed_schedule(args),
        "timing_cell_count": len(configs) * args.runs,
        "completion_rule": {
            "attempts_per_cell": 1,
            "valid_cell": (
                "llama-bench returncode 0 with pp_tokens == pp and finite "
                "positive pp_tok_s"
            ),
            "hook_accounting": (
                "hook accounting is recorded as optional metadata and never "
                "rejects or retries a cell"
            ),
        },
        "bypassed_gates": [
            "RM/PTIMER launch-clock calibration controls (endpoint precision and globaltimer identity)",
            "1.5 us bracket precision check",
            "gpubpf and NVBit launchlate source-schema checks",
            "correctness cells and the exact-47-byte output engagement gate",
            "verifier evidence gate",
            "probe-validity, retry, and rejection gates",
            "GPU safety/telemetry cell gate, read-only lease, ambient-injection and exact-driver admission",
        ],
        "preserved": [
            "per-cell raw stdout/stderr in each llama_bench.log plus the return code",
            "numeric throughput is recorded even when probe counters are zero or missing",
            "probe counters parsed and stored advisory-only; zero counters never invalidate a cell",
            "CPU affinity pinning to CPUs 8-15 and the private probe segment mechanics",
        ],
    }


def new_state(args: argparse.Namespace, timestamp: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    configs = gated.selected_configs(args)
    return {
        "kind": KIND,
        "timestamp": timestamp,
        "params": defining_params(args),
        "provenance": {
            "nvidia_smi": snapshot,
            "driver": gated.parse_driver(snapshot),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "model_file": gated.file_metadata(args.model),
            "llama_bench_file": gated.file_metadata(args.llama_bench),
            "llama_cli_file": gated.file_metadata(args.llama_cli),
            "libggml_cuda_file": gated.file_metadata(args.llama_bench.parent / "libggml-cuda.so"),
            "bpftime_agent_file": gated.file_metadata(
                args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"
            ),
            "bpftime_syscall_server_file": gated.file_metadata(
                args.bpftime_build_dir
                / "runtime/syscall-server/libbpftime-syscall-server.so"
            ),
        },
        "artifacts": {},
        "schedule": gated.fixed_schedule(args),
        "configs": {config: {"runs": []} for config in configs},
    }


def verify_resume(
    state: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Path], Path]:
    """Rebind the recorded artifact paths; never rebuild or relabel them."""
    if state.get("kind") != KIND:
        raise RuntimeError("resume state is not a launchlate performance-only campaign")
    if dict(state.get("params", {})) != defining_params(args):
        raise RuntimeError("resume parameters differ from the recorded campaign")
    if set(state.get("configs", {})) != set(CONFIGS):
        raise RuntimeError("resume timing matrix differs from the three launchlate arms")
    if state.get("schedule") != gated.fixed_schedule(args):
        raise RuntimeError("resume schedule differs from the fixed randomized matrix")
    artifacts = state.get("artifacts", {})
    if set(artifacts) != {"gpubpf_launchlate", "nvbit_tool"}:
        raise RuntimeError("resume artifact manifest is incomplete")
    paths: dict[str, Path] = {}
    for name in ("gpubpf_launchlate", "nvbit_tool"):
        path = Path(artifacts[name]["path"])
        if not path.is_file():
            raise RuntimeError(f"resume artifact {name} is missing: {path}")
        paths[name] = path
    return {"launchlate": paths["gpubpf_launchlate"].parent}, paths["nvbit_tool"]


def run_perf_cell(
    config: str,
    block: int,
    attempt: int,
    args: argparse.Namespace,
    output_dir: Path,
    tool_dirs: dict[str, Path],
    nvbit_tool: Path,
) -> dict[str, Any]:
    """One benchmark cell; probe counters are recorded but never gate it."""
    run_id = block * 100 + attempt
    run_dir = output_dir / f"{config}_run_{run_id:02d}"
    gated.idle_gpu_or_error(core.nvidia_smi_snapshot())
    if config == "baseline":
        result = gated.run_bench(config, run_id, args, output_dir)
    elif config == "gpubpf_launchlate":
        with gated.private_probe("launchlate", args, tool_dirs["launchlate"], run_dir) as env:
            result = gated.run_bench(config, run_id, args, output_dir, env_extra=env)
        probe_log = run_dir / "probe.log"
        result["probe"] = gated.parse_gpubpf(
            "launchlate",
            probe_log.read_text(errors="replace") if probe_log.exists() else "",
        )
        result["probe_log"] = str(probe_log.relative_to(output_dir))
    else:
        result = gated.run_bench(config, run_id, args, output_dir, env_extra={
            "LD_PRELOAD": str(nvbit_tool),
            "NOBANNER": "1",
            "OBS_MODE": "launchlate",
            "OBS_TARGET_SYMBOL": args.target_symbol,
            "OBS_GPU_THREAD_COUNT": str(args.gpu_thread_count),
        })
        result["probe"] = gated.parse_nvbit(
            "launchlate", (output_dir / result["log"]).read_text(errors="replace")
        )
    result["valid"] = benchmark_valid(result, args.pp)
    return result


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    blocks_total = int(state["params"]["runs"])
    configs = list(state["configs"])
    valid_by_block: dict[int, dict[str, dict[str, Any]]] = {}
    for config in configs:
        for run in state["configs"][config]["runs"]:
            if run.get("valid"):
                valid_by_block.setdefault(int(run["block"]), {})[config] = run
    block_rows: list[dict[str, Any]] = []
    for block in range(1, blocks_total + 1):
        cells = valid_by_block.get(block, {})
        row: dict[str, Any] = {"block": block, "complete": len(cells) == len(configs)}
        for config in configs:
            run = cells.get(config)
            row[config] = gated.pp_throughput(run) if run is not None else None
        baseline = cells.get("baseline")
        if baseline is not None:
            base = gated.pp_throughput(baseline)
            for config in ("gpubpf_launchlate", "nvbit_launchlate"):
                run = cells.get(config)
                row[f"{config}_overhead_pct"] = (
                    (base - gated.pp_throughput(run)) / base * 100.0
                    if run is not None else None
                )
        block_rows.append(row)
    config_rows: list[dict[str, Any]] = []
    for config in configs:
        runs = state["configs"][config]["runs"]
        values = [gated.pp_throughput(run) for run in runs if run.get("valid")]
        overheads = [
            row[f"{config}_overhead_pct"]
            for row in block_rows
            if row.get(f"{config}_overhead_pct") is not None
        ]
        config_rows.append({
            "config": config,
            "valid_blocks": len(values),
            "attempts": len(runs),
            "pp_tok_s_geomean": core.geomean(values),
            "mean_overhead_pct_vs_baseline": (
                sum(overheads) / len(overheads) if overheads else None
            ),
        })
    return {"blocks": block_rows, "configs": config_rows}


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["summary"] = summarize(state)
    (output_dir / "result.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "config", "valid_blocks", "attempts",
                "pp_tok_s_geomean", "mean_overhead_pct_vs_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(state["summary"]["configs"])

    lines = [
        "# RQ4 Table 1 launchlate performance-only campaign",
        "",
        f"- Kind: `{state['kind']}`",
        f"- Timestamp: `{state['timestamp']}`",
        f"- Driver: `{state['provenance']['driver']}`",
        f"- Target: `{state['params']['target_symbol']}`",
        f"- Blocks requested: `{state['params']['runs']}`",
        f"- pp: `{state['params']['pp']}`",
        "",
        "Gates bypassed: RM/PTIMER calibration, 1.5 us bracket precision check, "
        "source-schema checks, and engagement/correctness/verifier gates. "
        "No cell is rejected or retried on hook accounting.",
        "",
        "| Config | Valid blocks | Attempts | Prefill tok/s geomean | Mean overhead vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in state["summary"]["configs"]:
        gm = row["pp_tok_s_geomean"]
        overhead = row["mean_overhead_pct_vs_baseline"]
        lines.append(
            f"| {row['config']} | {row['valid_blocks']} | {row['attempts']} | "
            + (f"{gm:.2f}" if gm is not None else "n/a")
            + " | "
            + (f"{overhead:.2f}%" if overhead is not None else "-")
            + " |"
        )
    lines.extend([
        "",
        "## Blocks",
        "",
        "| Block | baseline tok/s | gpubpf tok/s | gpubpf overhead | NVBit tok/s | NVBit overhead |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in state["summary"]["blocks"]:
        def fmt(value: float | None, suffix: str = "") -> str:
            return f"{value:.2f}{suffix}" if value is not None else "n/a"
        lines.append(
            f"| {row['block']} | {fmt(row['baseline'])} | "
            f"{fmt(row['gpubpf_launchlate'])} | "
            f"{fmt(row.get('gpubpf_launchlate_overhead_pct'), '%')} | "
            f"{fmt(row['nvbit_launchlate'])} | "
            f"{fmt(row.get('nvbit_launchlate_overhead_pct'), '%')} |"
        )
    lines.extend([
        "",
        "Positive overhead means token/s degradation relative to the same-block "
        "no-probe baseline.",
        "Probe counters are recorded per cell for audit only and never gate a cell.",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(args: argparse.Namespace) -> None:
    if gated.selected_configs(args) != CONFIGS:
        raise ValueError("the performance-only matrix is fixed to baseline plus the two launchlate arms")
    if args.runs < 1 or args.pp <= 0:
        raise ValueError("--runs must be at least 1 and --pp must be positive")
    for path in (args.model, args.llama_bench, args.llama_cli,
                 args.uprobe_binary, args.bpftime_root, args.bpftime_build_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    for rel in ("runtime/agent/libbpftime-agent.so",
                "runtime/syscall-server/libbpftime-syscall-server.so"):
        if not (args.bpftime_build_dir / rel).exists():
            raise FileNotFoundError(args.bpftime_build_dir / rel)
    if args.nvbit_tool is not None and not args.nvbit_tool.is_file():
        raise FileNotFoundError(args.nvbit_tool)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=Path, default=core.DEFAULT_MODEL)
    parser.add_argument("--llama-bench", type=Path, default=core.DEFAULT_LLAMA_BENCH)
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument("--bpftime-root", type=Path, default=DEFAULT_BPFTIME_ROOT)
    parser.add_argument("--bpftime-build-dir", type=Path, default=DEFAULT_BPFTIME_BUILD_DIR)
    parser.add_argument("--target-symbol", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pp", type=int, default=512)
    parser.add_argument("--tg", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--probe-startup-s", type=float, default=3.0)
    parser.add_argument("--gpu-thread-count", type=int, default=22528)
    parser.add_argument("--uprobe-binary", type=Path, default=core.DEFAULT_LAUNCH_STUB_LIBRARY)
    parser.add_argument("--uprobe-symbol-hint", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument(
        "--nvbit-tool", type=Path, default=None,
        help="prebuilt matched observability.so; built from "
             "nvbit_adapters/observability when omitted",
    )
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the fixed randomized cell matrix and bypassed gates "
             "without touching build or GPU state",
    )
    args = parser.parse_args(argv)

    args.tools = ["launchlate"]
    args.threadhist_gpu_thread_count = 1048576
    args.llama_cli = args.llama_cli or (args.llama_bench.parent / "llama-cli")
    for field in ("model", "llama_bench", "llama_cli", "bpftime_root",
                  "bpftime_build_dir", "uprobe_binary"):
        setattr(args, field, getattr(args, field).resolve())
    if args.nvbit_tool is not None:
        args.nvbit_tool = args.nvbit_tool.resolve()
    return args


def run_campaign(args: argparse.Namespace) -> int:
    configs = gated.selected_configs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or HERE / "raw" / f"{KIND}-{timestamp}").resolve()
    if args.resume and not (output_dir / "result.json").exists():
        raise RuntimeError("--resume requires an existing result.json")
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to reuse a nonempty output directory without --resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = core.nvidia_smi_snapshot()
    driver = gated.parse_driver(snapshot)
    if not gated.nvbit_driver_supported(driver):
        raise RuntimeError(
            f"the matched NVBit 1.8 arm requires a 575.x or older driver; found {driver}"
        )
    gated.idle_gpu_or_error(snapshot)
    admission = {
        "kind": KIND,
        "timestamp": timestamp,
        "nvidia_smi": snapshot,
        "driver": driver,
        "nvbit_driver_supported": True,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (output_dir / "admission.json").write_text(
        json.dumps(admission, indent=2) + "\n", encoding="utf-8"
    )

    if args.resume:
        state = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
        tool_dirs, nvbit_tool = verify_resume(state, args)
    else:
        state = new_state(args, timestamp, snapshot)
        build_root = output_dir / "gpubpf_tool_build"
        build_root.mkdir(exist_ok=True)
        tool_dir = gated.prepare_tool_source(
            core.TOOLS["launchlate"],
            bpftime_root=args.bpftime_root,
            build_root=build_root,
            target_symbol=args.target_symbol,
        )
        core.build_tool(core.TOOLS["launchlate"], tool_dir)
        tool_dirs = {"launchlate": tool_dir}
        if args.nvbit_tool is not None:
            nvbit_tool = args.nvbit_tool
        else:
            nvbit_build_dir = output_dir / "nvbit_tool_build"
            shutil.copytree(
                gated.NVBIT_SOURCE_DIR,
                nvbit_build_dir,
                ignore=shutil.ignore_patterns("*.o", "*.so", "*.fatbin", "flush_channel.c"),
            )
            nvbit_tool = gated.build_nvbit(nvbit_build_dir, output_dir)
        state["artifacts"] = {
            "gpubpf_launchlate": {
                "path": str(tool_dirs["launchlate"] / "launchlate")
            },
            "nvbit_tool": {"path": str(nvbit_tool)},
        }
        write_state(output_dir, state)

    for block in range(1, args.runs + 1):
        for config in state["schedule"][str(block)]:
            # One attempt per cell: an attempted cell is kept as recorded,
            # valid or not, and is never retried.
            attempts = [
                run for run in state["configs"][config]["runs"]
                if run.get("block") == block
            ]
            if attempts:
                continue
            attempt = len(attempts) + 1
            run_id = block * 100 + attempt
            print(f"block={block} config={config} attempt={attempt}", flush=True)
            try:
                run = run_perf_cell(config, block, attempt, args, output_dir,
                                    tool_dirs, nvbit_tool)
            except gated.OwnedCleanupError as exc:
                run = {"returncode": -1, "valid": False, "error": str(exc),
                       "fatal_cleanup": exc.details}
                run.update(block=block, attempt=attempt, run=run_id)
                state["configs"][config]["runs"].append(run)
                state["fatal_cleanup"] = exc.details
                write_state(output_dir, state)
                return 3
            except Exception as exc:  # noqa: BLE001
                run = {"returncode": -1, "valid": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            run.update(block=block, attempt=attempt, run=run_id)
            state["configs"][config]["runs"].append(run)
            write_state(output_dir, state)

    write_state(output_dir, state)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    complete = all(
        any(run.get("valid") and run.get("block") == block
            for run in state["configs"][config]["runs"])
        for block in range(1, args.runs + 1)
        for config in configs
    )
    return 0 if complete else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), indent=2), flush=True)
        return 0
    validate(args)
    previous_run_cmd = core.run_cmd
    try:
        # Only this process uses owned CPU/build helpers, matching the gated runner.
        core.run_cmd = gated.run_cmd_owned
        return run_campaign(args)
    finally:
        core.run_cmd = previous_run_cmd


if __name__ == "__main__":
    raise SystemExit(main())

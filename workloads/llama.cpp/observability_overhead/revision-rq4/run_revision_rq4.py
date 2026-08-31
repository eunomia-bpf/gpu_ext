#!/usr/bin/env python3
"""Run the matched RTX 5090 gpubpf/NVBit observability experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OBS_ROOT = HERE.parent
sys.path.insert(0, str(OBS_ROOT))
import run_observability_overhead as core  # noqa: E402


NVBIT_ROOT = HERE / "deps/nvbit_release_x86_64"
NVBIT_SOURCE_DIR = HERE / "nvbit_adapters/observability"
CONFIGS = [
    "baseline",
    "gpubpf_kernelretsnoop",
    "nvbit_kernelretsnoop",
    "gpubpf_threadhist",
    "nvbit_threadhist",
    "gpubpf_launchlate",
    "nvbit_launchlate",
]
TASKS = ("kernelretsnoop", "threadhist", "launchlate")
SCHEDULE_SEED = 1797
BOOTSTRAP_SAMPLES = 10000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(args: argparse.Namespace) -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        OBS_ROOT / "run_observability_overhead.py",
        NVBIT_SOURCE_DIR / "Makefile",
        NVBIT_SOURCE_DIR / "common.h",
        NVBIT_SOURCE_DIR / "inject_funcs.cu",
        NVBIT_SOURCE_DIR / "observability.cu",
        NVBIT_SOURCE_DIR / "tool_func/flush_channel.cu",
    ]
    for tool in TASKS:
        spec = core.TOOLS[tool]
        paths.extend(
            [
                args.bpftime_root / spec.example_dir / "Makefile",
                args.bpftime_root / spec.example_dir / spec.bpf_file,
                args.bpftime_root / spec.example_dir / spec.user_file,
            ]
        )
    return {str(path): sha256(path) for path in paths}


def defining_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": args.phase,
        "model": str(args.model),
        "llama_bench": str(args.llama_bench),
        "llama_cli": str(args.llama_cli),
        "bpftime_root": str(args.bpftime_root),
        "bpftime_build_dir": str(args.bpftime_build_dir),
        "nvbit_root": str(NVBIT_ROOT),
        "target_symbol": args.target_symbol,
        "runs": args.runs,
        "pp": args.pp,
        "tg": args.tg,
        "n_gpu_layers": args.n_gpu_layers,
        "timeout_s": args.timeout_s,
        "probe_startup_s": args.probe_startup_s,
        "gpu_thread_count": args.gpu_thread_count,
        "threadhist_gpu_thread_count": args.threadhist_gpu_thread_count,
        "uprobe_binary": str(args.uprobe_binary),
        "uprobe_symbol_hint": args.uprobe_symbol_hint,
        "uvm": args.uvm,
        "no_warmup": args.no_warmup,
        "cuda_graphs_disabled": core.CUDA_GRAPHS_DISABLED,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def parse_driver(snapshot: dict[str, Any]) -> str:
    gpu = str(snapshot.get("gpu", ""))
    fields = [field.strip() for field in gpu.split(",")]
    return fields[1] if len(fields) > 1 else "unknown"


def nvbit_driver_supported(driver: str) -> bool:
    match = re.match(r"(\d+)", driver)
    return bool(match and int(match.group(1)) <= 575)


def idle_gpu_or_error(snapshot: dict[str, Any]) -> None:
    applications = str(snapshot.get("compute_apps", "")).strip()
    if applications:
        raise RuntimeError(
            "GPU is not idle; refusing to terminate or overlap external CUDA "
            f"processes:\n{applications}"
        )


def build_nvbit(source_dir: Path, log_dir: Path) -> Path:
    core.run_cmd(
        [
            "make",
            "CXX=g++",
            f"NVBIT_ROOT={NVBIT_ROOT}",
            "ARCH=sm_120",
        ],
        cwd=source_dir,
        log_path=log_dir / "build_nvbit.log",
    )
    tool = source_dir / "observability.so"
    if not tool.exists():
        raise FileNotFoundError(tool)
    return tool


def parse_nvbit(tool: str, text: str) -> dict[str, Any]:
    def last(pattern: str) -> int:
        values = [int(value) for value in re.findall(pattern, text)]
        return values[-1] if values else 0

    selected = last(r"NVBIT selected_launches=(\d+)")
    if tool == "kernelretsnoop":
        events = last(r"NVBIT kernelretsnoop events=(\d+)")
        nonzero = last(r"NVBIT kernelretsnoop events=\d+ nonzero_timestamps=(\d+)")
        return {
            "sample_count": events,
            "nonzero_timestamps": nonzero,
            "selected_launches": selected,
        }
    if tool == "threadhist":
        nonzero = last(r"NVBIT threadhist nonzero_threads=(\d+)")
        total = last(r"NVBIT threadhist nonzero_threads=\d+ total_exit_probes=(\d+)")
        return {
            "sample_count": total,
            "nonzero_threads": nonzero,
            "selected_launches": selected,
        }
    samples = last(r"NVBIT launchlate samples=(\d+)")
    errors = last(r"NVBIT launchlate samples=\d+ clock_errors=(\d+)")
    bins = [last(rf"NVBIT launchlate bin_{index}=(\d+)") for index in range(10)]
    return {
        "sample_count": samples,
        "clock_errors": errors,
        "histogram": bins,
        "histogram_sum": sum(bins),
        "selected_launches": selected,
    }


def nvbit_probe_valid(tool: str, probe: dict[str, Any]) -> bool:
    samples = int(probe.get("sample_count", 0))
    selected = int(probe.get("selected_launches", 0))
    if samples <= 0 or selected <= 0:
        return False
    if tool == "kernelretsnoop":
        return int(probe.get("nonzero_timestamps", 0)) == samples
    if tool == "threadhist":
        return int(probe.get("nonzero_threads", 0)) > 0
    return (
        int(probe.get("clock_errors", -1)) == 0
        and int(probe.get("histogram_sum", -1)) == samples
        and selected == samples
    )


def run_nvbit_once(
    tool: str,
    run_id: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    label = f"nvbit_{tool}"
    result = core.run_llama_once(
        label,
        run_id,
        args,
        output_dir,
        env_extra={
            "LD_PRELOAD": str(args.nvbit_tool),
            "NOBANNER": "1",
            "OBS_MODE": tool,
            "OBS_TARGET_SYMBOL": args.target_symbol,
            "OBS_GPU_THREAD_COUNT": str(
                args.threadhist_gpu_thread_count
                if tool == "threadhist"
                else args.gpu_thread_count
            ),
        },
    )
    log_path = output_dir / result["log"]
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    result["probe"] = parse_nvbit(tool, text)
    result["valid"] = bool(result.get("valid")) and nvbit_probe_valid(
        tool, result["probe"]
    )
    return result


def gpubpf_probe_valid(tool: str, probe: dict[str, Any]) -> bool:
    samples = int(probe.get("sample_count", 0))
    if samples <= 0:
        return False
    if tool == "kernelretsnoop":
        return int(probe.get("nonzero_timestamps", 0)) == samples
    if tool == "threadhist":
        return int(probe.get("nonzero_threads", 0)) > 0
    return (
        int(probe.get("queue_underflows", -1)) == 0
        and int(probe.get("queue_overflows", -1)) == 0
        and int(probe.get("host_launches", -1))
        == int(probe.get("device_entries", -2))
        == samples
    )


def normalized_output(stdout: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def llama_cli_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(args.llama_cli),
        "-m",
        str(args.model),
        "-p",
        "Write one sentence explaining why deterministic tests matter.",
        "-n",
        "8",
        "-c",
        "512",
        "-ngl",
        str(args.n_gpu_layers),
        "--seed",
        str(SCHEDULE_SEED),
        "--temp",
        "0",
        "--no-display-prompt",
        "--simple-io",
        "--log-disable",
    ]


def run_cli_separate(
    cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: int, log_path: Path
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, stderr = process.communicate()
        returncode = -1
    else:
        returncode = process.returncode
    log_path.write_text(
        f"$ {' '.join(cmd)}\n# cwd: {cwd}\n\n## stdout\n{stdout}"
        f"\n## stderr\n{stderr}\n# exit: {returncode}\n",
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def correctness_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if args.uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    return env


def run_correctness_cell(
    config: str,
    attempt: int,
    args: argparse.Namespace,
    output_dir: Path,
    tool_dirs: dict[str, Path],
) -> dict[str, Any]:
    run_dir = output_dir / "correctness" / config / f"attempt_{attempt:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    core.cleanup_gpu(run_dir)
    env = correctness_env(args)
    probe_process = None
    tool = None
    if config != "baseline":
        system, tool = config.split("_", 1)
        if system == "gpubpf":
            probe_process = core.start_probe(core.TOOLS[tool], tool_dirs[tool], args, run_dir)
            env.update(core.agent_env(args, run_dir, tool))
        else:
            env.update(
                {
                    "LD_PRELOAD": str(args.nvbit_tool),
                    "NOBANNER": "1",
                    "OBS_MODE": tool,
                    "OBS_TARGET_SYMBOL": args.target_symbol,
                    "OBS_GPU_THREAD_COUNT": str(
                        args.threadhist_gpu_thread_count
                        if tool == "threadhist"
                        else args.gpu_thread_count
                    ),
                }
            )
    try:
        completed = run_cli_separate(
            llama_cli_cmd(args),
            cwd=core.WORKLOAD_DIR,
            env=env,
            timeout=args.timeout_s,
            log_path=run_dir / "llama_cli.log",
        )
    finally:
        if probe_process is not None:
            core.stop_probe(probe_process)

    output = normalized_output(completed.stdout)
    result: dict[str, Any] = {
        "attempt": attempt,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "stdout_bytes": len(output.encode()),
        "log": str((run_dir / "llama_cli.log").relative_to(output_dir)),
        "valid": completed.returncode == 0 and bool(output),
    }
    if tool is not None:
        if config.startswith("gpubpf_"):
            probe_log = run_dir / "probe.log"
            probe_text = probe_log.read_text(errors="replace") if probe_log.exists() else ""
            result["probe"] = core.parse_probe_samples(tool, probe_text)
            result["valid"] = bool(result["valid"]) and gpubpf_probe_valid(
                tool, result["probe"]
            )
        else:
            result["probe"] = parse_nvbit(tool, completed.stderr)
            result["valid"] = bool(result["valid"]) and nvbit_probe_valid(
                tool, result["probe"]
            )
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    if low == high:
        return ordered[low]
    fraction = location - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_mean_ci(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(SCHEDULE_SEED)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(sample) / len(sample))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": quantile(boot, 0.025),
        "ci95_high": quantile(boot, 0.975),
    }


def valid_run_for_block(state: dict[str, Any], config: str, block: int) -> dict[str, Any] | None:
    for run in reversed(state["configs"][config]["runs"]):
        if run.get("block") == block and run.get("valid"):
            return run
    return None


def valid_correctness(state: dict[str, Any], config: str) -> dict[str, Any] | None:
    baseline_attempts = state["correctness"]["baseline"]["attempts"]
    baseline = next(
        (attempt for attempt in reversed(baseline_attempts) if attempt.get("valid")),
        None,
    )
    if baseline is None:
        return None
    expected = baseline["stdout_sha256"]
    for attempt in reversed(state["correctness"][config]["attempts"]):
        if attempt.get("valid") and attempt.get("stdout_sha256") == expected:
            return attempt
    return None


def pp_throughput(run: dict[str, Any]) -> float:
    return float(run["metrics"]["pp_tok_s"])


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    config_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        valid = [run for run in state["configs"][config]["runs"] if run.get("valid")]
        by_block = {int(run["block"]): run for run in valid}
        values = [pp_throughput(by_block[block]) for block in sorted(by_block)]
        config_rows.append(
            {
                "config": config,
                "valid_blocks": len(values),
                "attempts": len(state["configs"][config]["runs"]),
                "pp_tok_s_geomean": core.geomean(values),
            }
        )

    comparisons = []
    for task in TASKS:
        effects = []
        paired_rows = []
        for block in range(1, int(state["params"]["runs"]) + 1):
            baseline = valid_run_for_block(state, "baseline", block)
            gpubpf = valid_run_for_block(state, f"gpubpf_{task}", block)
            nvbit = valid_run_for_block(state, f"nvbit_{task}", block)
            if not (baseline and gpubpf and nvbit):
                continue
            base_t = pp_throughput(baseline)
            gpubpf_overhead = (base_t - pp_throughput(gpubpf)) / base_t * 100.0
            nvbit_overhead = (base_t - pp_throughput(nvbit)) / base_t * 100.0
            effect = nvbit_overhead - gpubpf_overhead
            effects.append(effect)
            paired_rows.append(
                {
                    "block": block,
                    "baseline_pp_tok_s": base_t,
                    "gpubpf_overhead_pct": gpubpf_overhead,
                    "nvbit_overhead_pct": nvbit_overhead,
                    "effect_pct_points": effect,
                }
            )
        comparisons.append(
            {
                "task": task,
                "paired_blocks": len(effects),
                "effect_definition": "NVBit overhead - gpubpf overhead (percentage points)",
                "paired": paired_rows,
                "bootstrap": bootstrap_mean_ci(effects),
            }
        )
    return {"configs": config_rows, "comparisons": comparisons}


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["summary"] = summarize(state)
    (output_dir / "result.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["config", "valid_blocks", "attempts", "pp_tok_s_geomean"],
        )
        writer.writeheader()
        writer.writerows(state["summary"]["configs"])

    lines = [
        "# RQ4 matched observability experiment",
        "",
        f"- Phase: `{state['phase']}`",
        f"- Driver: `{state['provenance']['driver']}`",
        f"- Target: `{state['params']['target_symbol']}`",
        f"- Blocks requested: `{state['params']['runs']}`",
        "",
        "| Config | Valid blocks | Attempts | Prefill tok/s geomean |",
        "|---|---:|---:|---:|",
    ]
    for row in state["summary"]["configs"]:
        gm = row["pp_tok_s_geomean"]
        lines.append(
            f"| {row['config']} | {row['valid_blocks']} | {row['attempts']} | "
            f"{gm:.2f} |" if gm is not None else
            f"| {row['config']} | {row['valid_blocks']} | {row['attempts']} | n/a |"
        )
    lines.extend(["", "## Paired effects", ""])
    for comparison in state["summary"]["comparisons"]:
        ci = comparison["bootstrap"]
        if ci:
            result = (
                f"mean {ci['mean']:.2f} pp, 95% CI "
                f"[{ci['ci95_low']:.2f}, {ci['ci95_high']:.2f}]"
            )
        else:
            result = "incomplete"
        lines.append(
            f"- {comparison['task']}: {comparison['paired_blocks']} paired blocks; {result}."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def new_state(args: argparse.Namespace, timestamp: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    schedules = {}
    for block in range(1, args.runs + 1):
        order = list(CONFIGS)
        random.Random(SCHEDULE_SEED + block).shuffle(order)
        schedules[str(block)] = order
    artifact = HERE / "deps/nvbit-Linux-x86_64-1.8.tar.bz2"
    return {
        "timestamp": timestamp,
        "phase": args.phase,
        "params": defining_params(args),
        "provenance": {
            "gpu_ext_git": core.git_rev(core.GPU_EXT_ROOT),
            "bpftime_git": core.git_rev(args.bpftime_root),
            "nvidia_smi": snapshot,
            "driver": parse_driver(snapshot),
            "nvbit_driver_supported": nvbit_driver_supported(parse_driver(snapshot)),
            "cuda_ptx": core.cuda_ptx_snapshot(args.llama_bench),
            "model_sha256": sha256(args.model),
            "llama_bench_sha256": sha256(args.llama_bench),
            "llama_cli_sha256": sha256(args.llama_cli),
            "libggml_cuda_sha256": sha256(args.llama_bench.parent / "libggml-cuda.so"),
            "bpftime_agent_sha256": sha256(
                args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"
            ),
            "bpftime_syscall_server_sha256": sha256(
                args.bpftime_build_dir
                / "runtime/syscall-server/libbpftime-syscall-server.so"
            ),
            "nvbit_artifact_sha256": sha256(artifact),
            "source_manifest": source_manifest(args),
        },
        "schedule": schedules,
        "correctness": {config: {"attempts": []} for config in CONFIGS},
        "artifacts": {},
        "configs": {config: {"runs": []} for config in CONFIGS},
    }


def record_artifacts(
    state: dict[str, Any], args: argparse.Namespace, tool_dirs: dict[str, Path]
) -> None:
    paths = {"nvbit_tool": args.nvbit_tool}
    for tool, directory in tool_dirs.items():
        paths[f"gpubpf_{tool}"] = directory / tool
    state["artifacts"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }


def verify_resume(
    state: dict[str, Any], args: argparse.Namespace, snapshot: dict[str, Any]
) -> dict[str, Path]:
    if state.get("params") != defining_params(args):
        raise RuntimeError("resume parameters differ from the recorded experiment")
    if state.get("provenance", {}).get("driver") != parse_driver(snapshot):
        raise RuntimeError("resume driver differs from the recorded experiment")
    checks = {
        "bpftime_git": core.git_rev(args.bpftime_root),
        "model_sha256": sha256(args.model),
        "llama_bench_sha256": sha256(args.llama_bench),
        "llama_cli_sha256": sha256(args.llama_cli),
        "libggml_cuda_sha256": sha256(args.llama_bench.parent / "libggml-cuda.so"),
        "bpftime_agent_sha256": sha256(
            args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"
        ),
        "bpftime_syscall_server_sha256": sha256(
            args.bpftime_build_dir
            / "runtime/syscall-server/libbpftime-syscall-server.so"
        ),
        "source_manifest": source_manifest(args),
    }
    provenance = state.get("provenance", {})
    for key, actual in checks.items():
        if provenance.get(key) != actual:
            raise RuntimeError(f"resume provenance mismatch: {key}")

    tool_dirs: dict[str, Path] = {}
    for name, artifact in state.get("artifacts", {}).items():
        path = Path(artifact["path"])
        if not path.exists() or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"resume artifact mismatch: {name}")
        if name == "nvbit_tool":
            args.nvbit_tool = path
        elif name.startswith("gpubpf_"):
            tool_dirs[name.removeprefix("gpubpf_")] = path.parent
    if set(tool_dirs) != set(TASKS) or not hasattr(args, "nvbit_tool"):
        raise RuntimeError("resume artifact manifest is incomplete")
    return tool_dirs


def run_cell(
    config: str,
    run_id: int,
    args: argparse.Namespace,
    output_dir: Path,
    tool_dirs: dict[str, Path],
) -> dict[str, Any]:
    if config == "baseline":
        return core.run_llama_once("baseline", run_id, args, output_dir)
    system, tool = config.split("_", 1)
    if system == "gpubpf":
        return core.run_tool_once(core.TOOLS[tool], tool_dirs[tool], run_id, args, output_dir)
    return run_nvbit_once(tool, run_id, args, output_dir)


def validate(args: argparse.Namespace) -> None:
    core.validate(args)
    if not args.llama_cli.exists():
        raise FileNotFoundError(args.llama_cli)
    if not NVBIT_ROOT.exists():
        raise FileNotFoundError(NVBIT_ROOT)
    if args.phase == "preflight" and (args.runs != 1 or args.pp != 32):
        raise ValueError("preflight is fixed to --runs 1 --pp 32")
    if args.phase == "full" and (args.runs != 10 or args.pp != 512):
        raise ValueError("paper-facing full run is fixed to --runs 10 --pp 512")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=Path, default=core.DEFAULT_MODEL)
    parser.add_argument("--llama-bench", type=Path, default=core.DEFAULT_LLAMA_BENCH)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--bpftime-root", type=Path, default=core.DEFAULT_BPFTIME_ROOT)
    parser.add_argument(
        "--bpftime-build-dir",
        type=Path,
        default=Path("/home/yunwei37/workspace/gpu/bpftime/build-cuda-pr503"),
    )
    parser.add_argument("--target-symbol", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--tg", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--probe-startup-s", type=float, default=3.0)
    parser.add_argument("--gpu-thread-count", type=int, default=8192)
    parser.add_argument("--threadhist-gpu-thread-count", type=int, default=1048576)
    parser.add_argument("--uprobe-binary", type=Path, default=core.DEFAULT_LAUNCH_STUB_LIBRARY)
    parser.add_argument("--uprobe-symbol-hint", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    args.runs = args.runs if args.runs is not None else (1 if args.phase == "preflight" else 10)
    args.pp = args.pp if args.pp is not None else (32 if args.phase == "preflight" else 512)
    args.llama_cli = args.llama_cli or (args.llama_bench.parent / "llama-cli")
    for field in ("model", "llama_bench", "llama_cli", "bpftime_root", "bpftime_build_dir", "uprobe_binary"):
        setattr(args, field, getattr(args, field).resolve())
    args.tools = list(TASKS)
    validate(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or HERE / "raw" / f"{args.phase}-{timestamp}").resolve()
    if args.resume:
        if not (output_dir / "result.json").exists():
            raise RuntimeError("--resume requires an existing result.json")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to reuse a nonempty output directory without --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = core.nvidia_smi_snapshot()

    admission = {
        "timestamp": timestamp,
        "phase": args.phase,
        "nvidia_smi": snapshot,
        "driver": parse_driver(snapshot),
        "nvbit_driver_supported": nvbit_driver_supported(parse_driver(snapshot)),
    }
    (output_dir / f"admission-{timestamp}.json").write_text(
        json.dumps(admission, indent=2) + "\n", encoding="utf-8"
    )
    if not admission["nvbit_driver_supported"]:
        raise RuntimeError(
            f"NVBit v1.8 documents driver <=575.xx; found {admission['driver']}. "
            "Official preflight and full execution both require a supported stack."
        )
    idle_gpu_or_error(snapshot)

    result_path = output_dir / "result.json"
    if args.resume:
        state = json.loads(result_path.read_text(encoding="utf-8"))
        tool_dirs = verify_resume(state, args, snapshot)
    else:
        nvbit_build_dir = output_dir / "nvbit_tool_build"
        shutil.copytree(
            NVBIT_SOURCE_DIR,
            nvbit_build_dir,
            ignore=shutil.ignore_patterns("*.o", "*.so", "*.fatbin", "flush_channel.c"),
        )
        args.nvbit_tool = build_nvbit(nvbit_build_dir, output_dir)

        build_root = output_dir / "gpubpf_tool_build"
        build_root.mkdir(exist_ok=True)
        tool_dirs = {}
        for tool in TASKS:
            directory = core.prepare_tool_source(
                core.TOOLS[tool],
                bpftime_root=args.bpftime_root,
                build_root=build_root,
                target_symbol=args.target_symbol,
            )
            core.build_tool(core.TOOLS[tool], directory)
            tool_dirs[tool] = directory

        state = new_state(args, timestamp, snapshot)
        record_artifacts(state, args, tool_dirs)
        write_state(output_dir, state)

    correctness_order = ["baseline"] + [
        config for config in state["schedule"]["1"] if config != "baseline"
    ]
    for config in correctness_order:
        if valid_correctness(state, config):
            continue
        attempt = len(state["correctness"][config]["attempts"]) + 1
        print(f"correctness config={config} attempt={attempt}", flush=True)
        try:
            check = run_correctness_cell(config, attempt, args, output_dir, tool_dirs)
        except Exception as exc:  # noqa: BLE001
            check = {"attempt": attempt, "returncode": -1, "valid": False, "error": str(exc)}
        if config != "baseline":
            baseline = valid_correctness(state, "baseline")
            check["matches_baseline"] = bool(
                baseline
                and check.get("stdout_sha256") == baseline.get("stdout_sha256")
            )
            check["valid"] = bool(check.get("valid")) and check["matches_baseline"]
        state["correctness"][config]["attempts"].append(check)
        write_state(output_dir, state)
        if config == "baseline" and valid_correctness(state, "baseline") is None:
            break

    if any(valid_correctness(state, config) is None for config in CONFIGS):
        print("Correctness gate incomplete; performance cells were not started.", flush=True)
        return 2

    for block in range(1, args.runs + 1):
        for config in state["schedule"][str(block)]:
            if valid_run_for_block(state, config, block):
                continue
            attempts = [
                run for run in state["configs"][config]["runs"]
                if run.get("block") == block
            ]
            attempt = len(attempts) + 1
            run_id = block * 100 + attempt
            print(f"block={block} config={config} attempt={attempt}", flush=True)
            try:
                run = run_cell(config, run_id, args, output_dir, tool_dirs)
            except Exception as exc:  # noqa: BLE001
                run = {"returncode": -1, "valid": False, "error": str(exc)}
            run["block"] = block
            run["attempt"] = attempt
            state["configs"][config]["runs"].append(run)
            write_state(output_dir, state)

    write_state(output_dir, state)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    if any(
        valid_run_for_block(state, config, block) is None
        for block in range(1, args.runs + 1)
        for config in CONFIGS
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

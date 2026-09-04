#!/usr/bin/env python3
"""Measure llama.cpp prefill overhead for bpftime GPU observability tools.

The runner keeps all experiment-specific code under gpu_ext while reusing the
bpftime examples as source templates. For each tool, it copies the example into
the result directory, rewrites the SEC() target to a llama.cpp CUDA kernel, and
builds that copy against the local bpftime tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


WORKLOAD_DIR = Path(__file__).resolve().parents[1]
WORKLOADS_DIR = WORKLOAD_DIR.parent
GPU_EXT_ROOT = WORKLOADS_DIR.parent
GPU_WORKSPACE = GPU_EXT_ROOT.parent
DEFAULT_BPFTIME_ROOT = GPU_WORKSPACE / "bpftime"
DEFAULT_BPFTIME_BUILD_DIR = Path(
    os.environ.get("BPFTIME_BUILD_DIR", str(DEFAULT_BPFTIME_ROOT / "build"))
)
DEFAULT_MODEL = WORKLOAD_DIR / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
PTX_LLAMA_BENCH = WORKLOAD_DIR / "build-ptx-1b" / "bin" / "llama-bench"
STANDARD_LLAMA_BENCH = WORKLOAD_DIR / "build" / "bin" / "llama-bench"
DEFAULT_LLAMA_BENCH = PTX_LLAMA_BENCH if PTX_LLAMA_BENCH.exists() else STANDARD_LLAMA_BENCH
DEFAULT_LAUNCH_STUB_LIBRARY = DEFAULT_LLAMA_BENCH.parent / "libggml-cuda.so"
DEFAULT_GPU_THREAD_COUNT = 8192
DEFAULT_THREADHIST_GPU_THREAD_COUNT = 1048576
CUDA_GRAPHS_DISABLED = True

DEFAULT_TARGET_SYMBOL = (
    "_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli"
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    example_dir: str
    bpf_file: str
    user_file: str
    probe_kind: str


TOOLS = {
    "kernelretsnoop": ToolSpec(
        name="kernelretsnoop",
        example_dir="example/gpu/kernelretsnoop",
        bpf_file="kernelretsnoop.bpf.c",
        user_file="kernelretsnoop.c",
        probe_kind="kretprobe",
    ),
    "threadhist": ToolSpec(
        name="threadhist",
        example_dir="example/gpu/threadhist",
        bpf_file="threadhist.bpf.c",
        user_file="threadhist.c",
        probe_kind="kretprobe",
    ),
    "launchlate": ToolSpec(
        name="launchlate",
        example_dir="example/gpu/launchlate",
        bpf_file="launchlate.bpf.c",
        user_file="launchlate.c",
        probe_kind="kprobe",
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    log_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    started = datetime.now().isoformat(timespec="seconds")
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"$ {' '.join(cmd)}\n# cwd: {cwd or Path.cwd()}\n# started: {started}\n\n## output\n")
        log_file.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    output: list[str] = []
    lock = threading.Lock()

    def read_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with lock:
                output.append(line)
                if log_file:
                    log_file.write(line)
                    log_file.flush()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        returncode = proc.wait()
    reader.join(timeout=5)

    if log_file:
        log_file.write(f"\n# exit: {returncode}\n")
        if timed_out:
            log_file.write(f"# timeout_s: {timeout}\n")
        log_file.close()

    completed = subprocess.CompletedProcess(cmd, returncode, "".join(output), "")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output=completed.stdout)
    if check and returncode != 0:
        raise RuntimeError(f"command failed ({returncode}): {' '.join(cmd)}")
    return completed


def cleanup_gpu(output_dir: Path) -> None:
    # Do not invoke workloads/cleanup_gpu.py here: it terminates every CUDA
    # compute process on the host, including unrelated user services.  The
    # revision driver performs a read-only idle-GPU admission check instead.
    cleanup_bpftime_shm(output_dir)


def cleanup_bpftime_shm(output_dir: Path) -> None:
    removed: list[str] = []
    for path in Path("/dev/shm").glob("bpftime*"):
        try:
            if path.is_file():
                path.unlink()
                removed.append(str(path))
        except FileNotFoundError:
            pass
        except PermissionError as exc:
            log(f"warning: unable to remove {path}: {exc}")
    if removed:
        cleanup_log = output_dir / "cleanup_bpftime_shm.log"
        cleanup_log.parent.mkdir(parents=True, exist_ok=True)
        cleanup_log.write_text("\n".join(removed) + "\n", encoding="utf-8")


def nvidia_smi_snapshot() -> dict[str, Any]:
    gpu = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    ).stdout.strip()
    apps = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    ).stdout.strip()
    return {"gpu": gpu, "compute_apps": apps}


def cuda_ptx_snapshot(llama_bench: Path) -> dict[str, Any]:
    lib = llama_bench.parent / "libggml-cuda.so"
    if not lib.exists():
        return {"libggml_cuda": str(lib), "has_ptx": False, "ptx_count": 0, "error": "not found"}
    proc = run_cmd(["cuobjdump", "--list-ptx", str(lib)], check=False)
    ptx_count = len(re.findall(r"^PTX file\s+\d+:", proc.stdout, flags=re.MULTILINE))
    return {
        "libggml_cuda": str(lib.resolve()),
        "has_ptx": ptx_count > 0,
        "ptx_count": ptx_count,
        "cuobjdump_returncode": proc.returncode,
    }


def git_rev(path: Path) -> str:
    proc = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=path, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def patch_makefile(text: str, bpftime_root: Path) -> str:
    text = text.replace("../../../third_party", str(bpftime_root / "third_party"))
    text = text.replace("all: $(APPS) vec_add", "all: $(APPS)")
    return text


def patch_bpf_source(text: str, spec: ToolSpec, target_symbol: str) -> str:
    old = f'SEC("{spec.probe_kind}/_Z9vectorAddPKfS0_Pf")'
    new = f'SEC("{spec.probe_kind}/{target_symbol}")'
    if old not in text:
        raise RuntimeError(f"unable to find vectorAdd SEC target in {spec.bpf_file}")
    return text.replace(old, new)


def patch_launchlate_user_source(text: str) -> str:
    if "find_defined_symbol_matching" not in text:
        raise RuntimeError("launchlate source lacks exact-symbol matching")
    return text


def prepare_tool_source(
    spec: ToolSpec,
    *,
    bpftime_root: Path,
    build_root: Path,
    target_symbol: str,
) -> Path:
    source_dir = bpftime_root / spec.example_dir
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    dest = build_root / spec.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_dir, dest, ignore=shutil.ignore_patterns(".output"))

    makefile = dest / "Makefile"
    makefile.write_text(patch_makefile(makefile.read_text(), bpftime_root))

    bpf_path = dest / spec.bpf_file
    bpf_path.write_text(patch_bpf_source(bpf_path.read_text(), spec, target_symbol))
    if spec.name == "launchlate":
        user_path = dest / spec.user_file
        user_path.write_text(patch_launchlate_user_source(user_path.read_text()))

    return dest


def build_tool(spec: ToolSpec, tool_dir: Path) -> None:
    run_cmd(["make"], cwd=tool_dir, log_path=tool_dir / "build.log")
    binary = tool_dir / spec.name
    if not binary.exists():
        raise FileNotFoundError(binary)


def extract_json_array(output: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\s*\[", output):
        try:
            parsed, _ = decoder.raw_decode(output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(entry, dict) for entry in parsed):
            return parsed
    for candidate in recover_json_arrays_from_mixed_log(output):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(entry, dict) for entry in parsed):
            return parsed
    raise ValueError("llama-bench JSON array not found in output")


def recover_json_arrays_from_mixed_log(output: str) -> list[str]:
    candidates: list[str] = []
    collecting = False
    bracket_depth = 0
    lines: list[str] = []
    log_prefix = re.compile(r"^\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not collecting:
            if line.strip() == "[":
                collecting = True
                bracket_depth = 1
                lines = ["["]
            continue

        if line.startswith("# exit:"):
            collecting = False
            bracket_depth = 0
            lines = []
            continue

        # bpftime logs can be written while llama-bench is emitting JSON, e.g.
        # `  }[2026-07-06 ...]`. Keep the JSON prefix and drop the log suffix.
        split = re.split(r"(?=\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", line, maxsplit=1)
        json_part = split[0].rstrip()
        if log_prefix.match(json_part.lstrip()):
            json_part = ""
        if not json_part:
            continue

        stripped = json_part.lstrip()
        if stripped.startswith("[") and stripped != "[":
            # A spdlog line that was not matched above is not part of the JSON.
            continue
        lines.append(json_part)
        bracket_depth += json_part.count("[") - json_part.count("]")
        if bracket_depth == 0:
            candidates.append("\n".join(lines))
            collecting = False
            lines = []

    return candidates


def parse_llama_bench(output: str) -> dict[str, Any]:
    raw = extract_json_array(output)
    metrics: dict[str, Any] = {}
    for entry in raw:
        if entry.get("n_prompt", 0) > 0:
            metrics["pp_tok_s"] = float(entry["avg_ts"])
            metrics["pp_stddev"] = float(entry.get("stddev_ts", 0.0))
            metrics["pp_tokens"] = int(entry["n_prompt"])
            metrics["pp_samples_tok_s"] = entry.get("samples_ts", [])
        if entry.get("n_gen", 0) > 0:
            metrics["tg_tok_s"] = float(entry["avg_ts"])
            metrics["tg_stddev"] = float(entry.get("stddev_ts", 0.0))
            metrics["tg_tokens"] = int(entry["n_gen"])
            metrics["tg_samples_tok_s"] = entry.get("samples_ts", [])
    return {"metrics": metrics, "raw": raw}


def geomean(values: list[float]) -> float | None:
    clean = [v for v in values if v > 0]
    if not clean:
        return None
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def make_llama_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.llama_bench),
        "-m",
        str(args.model),
        "-r",
        "1",
        "-o",
        "json",
        "-p",
        str(args.pp),
        "-n",
        str(args.tg),
        "-ngl",
        str(args.n_gpu_layers),
    ]
    if args.no_warmup:
        cmd.append("--no-warmup")
    return cmd


def run_llama_once(
    label: str,
    run_idx: int,
    args: argparse.Namespace,
    output_dir: Path,
    env_extra: dict[str, str] | None = None,
    do_cleanup: bool = True,
) -> dict[str, Any]:
    if do_cleanup:
        cleanup_gpu(output_dir / f"{label}_run_{run_idx:02d}")
    env = os.environ.copy()
    if CUDA_GRAPHS_DISABLED:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if args.uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    if env_extra:
        env.update(env_extra)

    log_path = output_dir / f"{label}_run_{run_idx:02d}" / "llama_bench.log"
    parsed: dict[str, Any] = {
        "run": run_idx,
        "log": str(log_path.relative_to(output_dir)),
    }
    try:
        proc = run_cmd(
            make_llama_cmd(args),
            cwd=WORKLOAD_DIR,
            env=env,
            timeout=args.timeout_s,
            log_path=log_path,
            check=False,
        )
    except subprocess.TimeoutExpired:
        parsed["returncode"] = -1
        parsed["error"] = f"llama-bench timed out after {args.timeout_s}s"
        return parsed

    parsed["returncode"] = proc.returncode
    if proc.returncode == 0:
        try:
            parsed.update(parse_llama_bench(proc.stdout + "\n" + proc.stderr))
            metrics = parsed.get("metrics", {})
            parsed["valid"] = (
                metrics.get("pp_tokens") == args.pp
                and math.isfinite(float(metrics.get("pp_tok_s", 0)))
                and float(metrics.get("pp_tok_s", 0)) > 0
            )
        except Exception as exc:  # noqa: BLE001
            parsed["error"] = f"parse failed: {exc}"
            parsed["valid"] = False
    else:
        parsed["error"] = f"llama-bench exited {proc.returncode}"
        parsed["valid"] = False
    return parsed


def gpu_thread_count_for_tool(args: argparse.Namespace, tool: str) -> int:
    if tool == "threadhist":
        return args.threadhist_gpu_thread_count
    return args.gpu_thread_count


def probe_env(args: argparse.Namespace, tool: str) -> dict[str, str]:
    env = os.environ.copy()
    env["BPFTIME_LOG_OUTPUT"] = "console"
    env["SPDLOG_LEVEL"] = "warn"
    env["LD_PRELOAD"] = str(
        args.bpftime_build_dir / "runtime/syscall-server/libbpftime-syscall-server.so"
    )
    env.setdefault("BPFTIME_MAP_GPU_THREAD_COUNT", str(gpu_thread_count_for_tool(args, tool)))
    if tool == "kernelretsnoop":
        env.setdefault("BPFTIME_SHM_MEMORY_MB", "1000")
    if tool == "threadhist":
        env.setdefault("BPFTIME_SHM_MEMORY_MB", "200")
    return env


def agent_env(args: argparse.Namespace, run_dir: Path, tool: str) -> dict[str, str]:
    env = {
        "BPFTIME_LOG_OUTPUT": str(run_dir / "agent.log"),
        "SPDLOG_LEVEL": "warn",
        "BPFTIME_CUDA_DEFER_PTX_EXTRACTION": "1",
        "BPFTIME_CUDA_TARGETED_LATE_BOOTSTRAP": "1",
        "BPFTIME_MAP_GPU_THREAD_COUNT": str(gpu_thread_count_for_tool(args, tool)),
        "GGML_NO_BACKTRACE": "1",
        "LD_PRELOAD": str(args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"),
    }
    return env


def start_probe(
    spec: ToolSpec,
    tool_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
) -> subprocess.Popen[str]:
    cmd = [str(tool_dir / spec.name)]
    if spec.name == "launchlate":
        cmd.append(str(args.uprobe_binary))
        cmd.append(args.uprobe_symbol_hint)

    log_file = open(run_dir / "probe.log", "w", encoding="utf-8")
    log_file.write(f"$ {' '.join(cmd)}\n# cwd: {WORKLOAD_DIR}\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKLOAD_DIR),
        env=probe_env(args, spec.name),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._llama_obs_log_file = log_file  # type: ignore[attr-defined]
    time.sleep(args.probe_startup_s)
    if proc.poll() is not None:
        log_file.close()
        raise RuntimeError(f"{spec.name} probe exited early; see {run_dir / 'probe.log'}")
    return proc


def stop_probe(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    log_file = getattr(proc, "_llama_obs_log_file", None)
    if log_file:
        log_file.close()


def parse_probe_samples(tool: str, text: str) -> dict[str, Any]:
    if tool == "kernelretsnoop":
        counts = [int(x) for x in re.findall(r"Total events collected:\s*(\d+)", text)]
        nonzero = [int(x) for x in re.findall(r"Nonzero timestamps:\s*(\d+)", text)]
        return {
            "sample_count": max(counts) if counts else 0,
            "nonzero_timestamps": max(nonzero) if nonzero else 0,
        }
    if tool == "launchlate":
        counts = [int(x) for x in re.findall(r"Total samples:\s*(\d+)", text)]
        def last_int(label: str) -> int:
            values = [int(x) for x in re.findall(rf"{label}:\s*(\d+)", text)]
            return values[-1] if values else 0

        return {
            "sample_count": max(counts) if counts else 0,
            "host_launches": last_int("Host launches"),
            "device_entries": last_int("Device entries"),
            "queue_underflows": last_int("Queue underflows"),
            "queue_overflows": last_int("Queue overflows"),
        }
    if tool == "threadhist":
        totals = [int(x) for x in re.findall(r"Total exit probes:\s*(\d+)", text)]
        nonzero = [int(x) for x in re.findall(r"Nonzero threads:\s*(\d+)", text)]
        if totals:
            return {
                "sample_count": totals[-1],
                "nonzero_threads": nonzero[-1] if nonzero else 0,
            }
        snapshots: list[int] = []
        current = 0
        seen = False
        for line in text.splitlines():
            if re.match(r"\d{2}:\d{2}:\d{2}", line):
                if seen:
                    snapshots.append(current)
                current = 0
                seen = True
                continue
            match = re.match(r"Thread \d+:\s*(\d+)", line)
            if match:
                current += int(match.group(1))
        if seen:
            snapshots.append(current)
        return {"sample_count": max(snapshots) if snapshots else 0}
    return {"sample_count": 0}


def run_tool_once(
    spec: ToolSpec,
    tool_dir: Path,
    run_idx: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_dir / f"{spec.name}_run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cleanup_gpu(run_dir)
    probe = start_probe(spec, tool_dir, args, run_dir)
    try:
        result = run_llama_once(
            spec.name,
        run_idx,
        args,
        output_dir,
        env_extra=agent_env(args, run_dir, spec.name),
        do_cleanup=False,
    )
    finally:
        stop_probe(probe)

    probe_log = run_dir / "probe.log"
    if probe_log.exists():
        result["probe"] = parse_probe_samples(spec.name, probe_log.read_text(errors="replace"))
        result["probe_log"] = str(probe_log.relative_to(output_dir))
    agent_log = run_dir / "agent.log"
    if agent_log.exists():
        result["agent_log"] = str(agent_log.relative_to(output_dir))
    probe = result.get("probe", {})
    probe_valid = probe.get("sample_count", 0) > 0
    if spec.name == "kernelretsnoop":
        probe_valid = probe_valid and probe.get("nonzero_timestamps") == probe.get("sample_count")
    elif spec.name == "threadhist":
        probe_valid = probe_valid and probe.get("nonzero_threads", 0) > 0
    elif spec.name == "launchlate":
        probe_valid = (
            probe_valid
            and probe.get("queue_underflows") == 0
            and probe.get("queue_overflows") == 0
            and probe.get("host_launches") == probe.get("device_entries")
            and probe.get("device_entries") == probe.get("sample_count")
        )
    result["valid"] = bool(result.get("valid")) and probe_valid
    return result


def summarize(results: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_runs = results["configs"]["baseline"]["runs"]
    baseline_pp = [
        r.get("metrics", {}).get("pp_tok_s")
        for r in baseline_runs
        if r.get("returncode") == 0 and r.get("valid") and r.get("metrics", {}).get("pp_tok_s")
    ]
    baseline_gm = geomean([float(x) for x in baseline_pp])

    rows: list[dict[str, Any]] = []
    for name, config in results["configs"].items():
        values = [
            float(r.get("metrics", {}).get("pp_tok_s"))
            for r in config["runs"]
            if r.get("returncode") == 0 and r.get("valid") and r.get("metrics", {}).get("pp_tok_s")
        ]
        gm = geomean(values)
        sample_counts = [r.get("probe", {}).get("sample_count", 0) for r in config["runs"]]
        overhead = None
        if name != "baseline" and gm and baseline_gm:
            overhead = (baseline_gm - gm) / baseline_gm * 100.0
        rows.append(
            {
                "config": name,
                "successful_runs": len(values),
                "pp_tok_s_geomean": gm,
                "pp_tok_s_runs": values,
                "overhead_pct_vs_baseline": overhead,
                "max_probe_samples": max(sample_counts) if sample_counts else 0,
            }
        )
    return rows


def write_summary(output_dir: Path, results: dict[str, Any]) -> None:
    rows = summarize(results)
    results["summary"] = rows

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "successful_runs",
                "pp_tok_s_geomean",
                "overhead_pct_vs_baseline",
                "max_probe_samples",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    lines = [
        "# llama.cpp observability overhead",
        "",
        f"- Timestamp: `{results['timestamp']}`",
        f"- Model: `{results['params']['model']}`",
        f"- llama-bench: `{results['params']['llama_bench']}`",
        f"- Target kernel: `{results['params']['target_symbol']}`",
        f"- Workload: `llama-bench -p {results['params']['pp']} -n {results['params']['tg']}`",
        f"- Runs per config: `{results['params']['runs']}`",
        f"- PTX files in libggml-cuda: `{results['provenance']['cuda_ptx']['ptx_count']}`",
        f"- CUDA graphs disabled: `{results['params']['cuda_graphs_disabled']}`",
        "",
        "| Config | Runs | Prefill tok/s geomean | Overhead vs baseline | Max probe samples |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        gm = row["pp_tok_s_geomean"]
        overhead = row["overhead_pct_vs_baseline"]
        lines.append(
            "| {config} | {runs} | {gm} | {overhead} | {samples} |".format(
                config=row["config"],
                runs=row["successful_runs"],
                gm=f"{gm:.2f}" if gm is not None else "n/a",
                overhead=f"{overhead:.2f}%" if overhead is not None else "-",
                samples=row["max_probe_samples"],
            )
        )
    lines.extend(
        [
            "",
            "Positive overhead means token/s degradation relative to the no-probe baseline.",
            "A zero probe sample count means the selected CUDA kernel was not observed for that tool run.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def validate(args: argparse.Namespace) -> None:
    if not args.llama_bench.exists():
        raise FileNotFoundError(f"llama-bench not found: {args.llama_bench}")
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")
    if not args.bpftime_root.exists():
        raise FileNotFoundError(f"bpftime root not found: {args.bpftime_root}")
    if not args.bpftime_build_dir.exists():
        raise FileNotFoundError(f"bpftime build directory not found: {args.bpftime_build_dir}")
    for rel in [
        "runtime/syscall-server/libbpftime-syscall-server.so",
        "runtime/agent/libbpftime-agent.so",
    ]:
        path = args.bpftime_build_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"bpftime runtime library not found: {path}")
    if "launchlate" in args.tools and not args.uprobe_binary.exists():
        raise FileNotFoundError(f"uprobe binary not found: {args.uprobe_binary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    parser.add_argument("--bpftime-root", type=Path, default=DEFAULT_BPFTIME_ROOT)
    parser.add_argument(
        "--bpftime-build-dir",
        type=Path,
        default=DEFAULT_BPFTIME_BUILD_DIR,
        help="CUDA-enabled bpftime CMake build directory",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-symbol", default=DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--tools", nargs="+", default=list(TOOLS.keys()), choices=list(TOOLS.keys()))
    parser.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "10")))
    parser.add_argument("--pp", type=int, default=int(os.environ.get("PP", "512")))
    parser.add_argument("--tg", type=int, default=int(os.environ.get("TG", "0")))
    parser.add_argument("--n-gpu-layers", type=int, default=int(os.environ.get("N_GPU_LAYERS", "99")))
    parser.add_argument("--timeout-s", type=int, default=int(os.environ.get("TIMEOUT_S", "300")))
    parser.add_argument("--probe-startup-s", type=float, default=float(os.environ.get("PROBE_STARTUP_S", "2")))
    parser.add_argument("--gpu-thread-count", type=int, default=int(os.environ.get("BPFTIME_MAP_GPU_THREAD_COUNT", str(DEFAULT_GPU_THREAD_COUNT))))
    parser.add_argument("--threadhist-gpu-thread-count", type=int, default=int(os.environ.get("BPFTIME_THREADHIST_GPU_THREAD_COUNT", str(DEFAULT_THREADHIST_GPU_THREAD_COUNT))))
    parser.add_argument("--uprobe-binary", type=Path, default=Path(os.environ.get("LAUNCHLATE_UPROBE_BINARY", str(DEFAULT_LAUNCH_STUB_LIBRARY))))
    parser.add_argument("--uprobe-symbol-hint", default=os.environ.get("LAUNCHLATE_UPROBE_SYMBOL_HINT", DEFAULT_TARGET_SYMBOL))
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    args.model = args.model.resolve()
    args.llama_bench = args.llama_bench.resolve()
    args.bpftime_root = args.bpftime_root.resolve()
    args.bpftime_build_dir = args.bpftime_build_dir.resolve()
    args.uprobe_binary = args.uprobe_binary.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (WORKLOAD_DIR / "results" / "exp_observability_overhead" / timestamp)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate(args)

    results: dict[str, Any] = {
        "timestamp": timestamp,
        "params": {
            "model": str(args.model),
            "llama_bench": str(args.llama_bench),
            "bpftime_root": str(args.bpftime_root),
            "bpftime_build_dir": str(args.bpftime_build_dir),
            "target_symbol": args.target_symbol,
            "tools": args.tools,
            "runs": args.runs,
            "pp": args.pp,
            "tg": args.tg,
            "n_gpu_layers": args.n_gpu_layers,
            "uvm": args.uvm,
            "no_warmup": args.no_warmup,
            "cuda_graphs_disabled": CUDA_GRAPHS_DISABLED,
            "gpu_thread_count": args.gpu_thread_count,
            "threadhist_gpu_thread_count": args.threadhist_gpu_thread_count,
            "uprobe_binary": str(args.uprobe_binary),
            "uprobe_symbol_hint": args.uprobe_symbol_hint,
        },
        "provenance": {
            "gpu_ext_git": git_rev(GPU_EXT_ROOT),
            "bpftime_git": git_rev(args.bpftime_root),
            "nvidia_smi": nvidia_smi_snapshot(),
            "cuda_ptx": cuda_ptx_snapshot(args.llama_bench),
        },
        "configs": {"baseline": {"runs": []}},
    }

    log(f"Results directory: {output_dir}")
    log(f"Target kernel: {args.target_symbol}")
    log("Running baseline...")
    for i in range(1, args.runs + 1):
        results["configs"]["baseline"]["runs"].append(run_llama_once("baseline", i, args, output_dir))
        write_summary(output_dir, results)

    build_root = output_dir / "tool_build"
    build_root.mkdir(exist_ok=True)
    for tool_name in args.tools:
        spec = TOOLS[tool_name]
        log(f"Preparing {tool_name}...")
        tool_dir = prepare_tool_source(
            spec,
            bpftime_root=args.bpftime_root,
            build_root=build_root,
            target_symbol=args.target_symbol,
        )
        build_tool(spec, tool_dir)
        results["configs"][tool_name] = {"tool_dir": str(tool_dir), "runs": []}
        log(f"Running {tool_name}...")
        for i in range(1, args.runs + 1):
            try:
                run = run_tool_once(spec, tool_dir, i, args, output_dir)
            except Exception as exc:  # noqa: BLE001
                run = {"run": i, "returncode": -1, "error": str(exc)}
            results["configs"][tool_name]["runs"].append(run)
            write_summary(output_dir, results)

    write_summary(output_dir, results)
    log((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

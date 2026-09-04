#!/usr/bin/env python3
"""Untimed stdout/probe diagnosis using existing correctness cells and binaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

import run_revision_rq4 as runner


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_inputs(source: Path, output: Path, task: str, build: Path | None,
                nvbit_tool: Path | None = None):
    if output == source or source in output.parents:
        raise ValueError("diagnostic output must be outside the retained source campaign")
    saved = json.loads((source / "result.json").read_text())
    args = argparse.Namespace(**saved["params"])
    if args.phase != "preflight":
        raise ValueError("reuse the closed preflight probe artifacts, not a timing campaign")
    for field in ("model", "llama_bench", "llama_cli", "bpftime_root", "bpftime_build_dir", "uprobe_binary"):
        setattr(args, field, Path(getattr(args, field)).resolve())
    if build is not None:
        args.bpftime_build_dir = build
    args.tools = [task]
    args.output_dir, args.resume = output, False
    tool = Path(saved["artifacts"][f"gpubpf_{task}"]["path"]).resolve()
    if tool.name != task or not tool.is_file():
        raise FileNotFoundError(f"prepared {task} binary missing or misnamed: {tool}")
    tools = {task: tool.parent}
    if task == "threadhist":
        args.nvbit_tool = nvbit_tool or Path(saved["artifacts"]["nvbit_tool"]["path"]).resolve()
        if not args.nvbit_tool.is_file():
            raise FileNotFoundError(args.nvbit_tool)
    runner.validate(args)  # Existing path/protocol checks; this never builds.
    return args, tools


def current_inventory(args, tools):
    paths = [Path(__file__), Path(runner.__file__), args.model, args.llama_cli,
             args.llama_cli.parent / "libggml-cuda.so", args.uprobe_binary,
             args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so",
             args.bpftime_build_dir / "runtime/syscall-server/libbpftime-syscall-server.so",
             args.bpftime_build_dir / "runtime/agent/CMakeFiles/bpftime-agent.dir/agent.cpp.o",
             runner.HERE / "runtime-575/runtime-575.patch",
             runner.LATE_BOOTSTRAP_TARGET_FILTER_PATCH,
             runner.HERE / "bootstrap-output-repair.md"]
    for task, directory in tools.items():
        spec = runner.core.TOOLS[task]
        paths += [directory / name for name in (task, spec.user_file, spec.bpf_file, "Makefile")]
    if hasattr(args, "nvbit_tool"):
        paths.append(args.nvbit_tool)
        paths += [runner.NVBIT_SOURCE_DIR / name for name in
                  ("observability.cu", "inject_funcs.cu", "common.h")]
        paths.append(runner.HERE / "nvbit-exit-predicate-repair.md")
    # Fresh stat calls, never the previous campaign's runtime metadata.
    return [runner.file_metadata(path) for path in paths]


def run_diagnostic(args, tools, task: str, source: Path) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)  # No resume, retries or overwrite.
    configs = (["baseline", "nvbit_threadhist", "gpubpf_threadhist"] if task == "threadhist"
               else ["baseline", "gpubpf_launchlate"])
    record = {"purpose": "untimed targeted diagnostic; not full preflight or performance",
              "started_utc": now(), "source_campaign": str(source),
              "runtime_repair_reference": "8f7d2d5; bootstrap-output-repair.md",
              "parameters": runner.defining_params(args),
              "current_files": current_inventory(args, tools), "configs": configs,
              "cells": [], "diagnostic_passed": False, "timing_cells_started": 0}

    def save():
        (output / "diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")

    try:
        save()
        snapshot = runner.core.nvidia_smi_snapshot()
        record["admission"] = snapshot
        if runner.parse_driver(snapshot) != runner.EXPECTED_DRIVER:
            raise RuntimeError(f"diagnostic requires driver {runner.EXPECTED_DRIVER}")
        runner.idle_gpu_or_error(snapshot)
        expected = None
        for config in configs:
            print(f"UNTIMED diagnostic config={config} attempt=1", flush=True)
            entry = {"config": config, "started_utc": now()}
            record["cells"].append(entry)
            save()
            try:
                check = runner.run_correctness_cell(
                    config, 1, args, output, tools,
                    diagnostic_log_level="info" if config == "gpubpf_launchlate" else None)
                entry.update(check)
                if config == "baseline":
                    expected = check.get("normalized_stdout") if check.get("valid") else None
                entry["matches_baseline"] = bool(expected) and check.get("normalized_stdout") == expected
                # Preserve the original validity result, including launchlate clock errors.
                entry["diagnostic_valid"] = bool(check.get("valid")) and entry["matches_baseline"]
            except BaseException as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                if isinstance(error, runner.OwnedCleanupError):
                    entry["fatal_cleanup"] = error.details
                raise
            finally:
                entry["ended_utc"] = now()
                save()
            if not entry["diagnostic_valid"]:
                print("Targeted correctness/engagement failed; no further cells started.", flush=True)
                return 2
        if task == "threadhist":
            nvbit, bpf = (entry["probe"] for entry in record["cells"][1:])
            expected_count, observed = nvbit["sample_count"], bpf["sample_count"]
            record["histogram_comparison"] = {
                "nvbit_samples": expected_count, "gpubpf_samples": observed,
                "gpubpf_over_nvbit": observed / expected_count,
                "difference_percent": 100.0 * (observed - expected_count) / expected_count,
                "samples_equal": observed == expected_count,
                "nvbit_nonzero_threads": nvbit["nonzero_threads"],
                "gpubpf_nonzero_threads": bpf["nonzero_threads"],
                "nonzero_threads_equal": nvbit["nonzero_threads"] == bpf["nonzero_threads"],
                "scope": "fresh matched-input aggregate counts, not per-launch coverage or timing"}
            comparison = record["histogram_comparison"]
            record["diagnostic_passed"] = comparison["samples_equal"] and comparison["nonzero_threads_equal"]
            print(json.dumps(comparison), flush=True)
        else:
            record["diagnostic_passed"] = True
        return 0 if record["diagnostic_passed"] else 2
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["ended_utc"] = now()
        save()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, default=runner.HERE / "raw/preflight-575-03")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("threadhist", "launchlate"), default="threadhist")
    parser.add_argument("--bpftime-build-dir", type=Path)
    parser.add_argument("--nvbit-tool", type=Path,
                        default=runner.NVBIT_SOURCE_DIR / "observability.so",
                        help="already-built current NVBit library; never build or copy the old runtime")
    options = parser.parse_args(argv)
    source, output = options.source_campaign.resolve(), options.output_dir.resolve()
    runner.reject_ambient_injection()
    args, tools = load_inputs(source, output, options.task,
                              options.bpftime_build_dir.resolve() if options.bpftime_build_dir else None,
                              options.nvbit_tool.resolve() if options.nvbit_tool else None)
    lease = runner.shared.Leases()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    previous = signal.signal(signal.SIGTERM, interrupted)
    previous_run_cmd = runner.core.run_cmd
    try:
        runner.core.run_cmd = runner.run_cmd_owned
        return run_diagnostic(args, tools, options.task, source)
    finally:
        runner.core.run_cmd = previous_run_cmd
        signal.signal(signal.SIGTERM, previous)
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay raw logs from the frozen STRICT-admitted uniform-map run."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


GPU_NAME = "NVIDIA GeForce RTX 5090"
DRIVER = "575.57.08"
SEED = 1797
ARMS = (
    "native", "noop", "device_update", "host_update",
    "device_lookup", "host_lookup",
)
PROGRAMS = {
    "noop": "cuda__noop",
    "device_update": "cuda__dev_up",
    "host_update": "cuda__host_up",
    "device_lookup": "cuda__dev_look",
    "host_lookup": "cuda__host_look",
}
UPDATE_MAGIC = 0x51A7CAFE00000001
LOOKUP_MAGIC = 0x10C4CAFE00000001


class AnalysisError(RuntimeError):
    pass


def phase_parameters(phase: str) -> tuple[int, int, int]:
    if phase == "preflight":
        return 1, 1, 2
    if phase == "full":
        return 12, 8, 64
    raise ValueError(phase)


def arm_base_order() -> tuple[str, ...]:
    values = list(ARMS)
    random.Random(SEED).shuffle(values)
    return tuple(values)


def expected_schedule(phase: str) -> list[dict[str, int | str]]:
    blocks, _warmup, _launches = phase_parameters(phase)
    base = arm_base_order()
    result: list[dict[str, int | str]] = []
    for block in range(blocks):
        cycle = base if block < len(ARMS) else tuple(reversed(base))
        offset = block % len(ARMS)
        order = cycle[offset:] + cycle[:offset]
        for position, arm in enumerate(order):
            result.append({
                "block": block + 1, "order": position + 1, "arm": arm,
                "run_id": block + 1,
            })
    return result


def read_schedule(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["block", "order", "arm", "run_id"]:
            raise AnalysisError("schedule columns changed")
        rows = []
        for row in reader:
            try:
                rows.append({
                    "block": int(row["block"]), "order": int(row["order"]),
                    "arm": row["arm"], "run_id": int(row["run_id"]),
                })
            except (TypeError, ValueError) as error:
                raise AnalysisError("malformed schedule row") from error
    return rows


def infer_phase(rows: list[dict[str, int | str]]) -> str:
    if len(rows) == len(ARMS):
        return "preflight"
    if len(rows) == 12 * len(ARMS):
        return "full"
    raise AnalysisError(f"unexpected schedule length: {len(rows)}")


def validate_environment(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        f"gpu\t{GPU_NAME}\n", f"driver\t{DRIVER}\n",
        "BPFTIME_ENABLE_CUDA_ATTACH\tON\n", "BPFTIME_LLVM_JIT\tON\n",
        "ENABLE_EBPF_VERIFIER\tON\n", "strict_binary_markers\tpresent\n",
        "bpftime_revision\t", "agent_bytes\t", "server_bytes\t",
        "nvcc_begin\n", "nvcc_end\n",
    )
    if any(marker not in text for marker in required):
        raise AnalysisError("environment record is incomplete or off-plan")


def parse_application(path: Path, warmup: int, launches: int) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    device = re.findall(r"^FIG15_DEVICE\t(.+)\t12\t0\t32$", text, re.MULTILINE)
    measurement = re.findall(
        r"^FIG15_MEASUREMENT\t(\d+)\t(\d+)\t([0-9.eE+-]+)$", text, re.MULTILINE,
    )
    correctness = re.findall(r"^FIG15_CORRECT\t(\d+)\t(\d+)$", text, re.MULTILINE)
    if device != [GPU_NAME] or len(measurement) != 1:
        raise AnalysisError(f"application identity/timing missing: {path}")
    measured_warmup, measured_launches, elapsed = measurement[0]
    if (int(measured_warmup), int(measured_launches)) != (warmup, launches):
        raise AnalysisError(f"timing parameters changed: {path}")
    elapsed_ms = float(elapsed)
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise AnalysisError(f"invalid elapsed time: {path}")
    if correctness != [("32", "0")]:
        raise AnalysisError(f"CUDA correctness failed: {path}")
    return elapsed_ms * 1000.0 / launches


def expected_map_rows(arm: str) -> dict[tuple[str, int], int]:
    if arm == "noop":
        return {}
    source = "device_values" if arm.startswith("device_") else "host_values"
    if arm.endswith("_update"):
        return {(source, 0): UPDATE_MAGIC}
    return {(source, 0): LOOKUP_MAGIC, ("observed_values", 0): LOOKUP_MAGIC}


def parse_loader(path: Path, arm: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    prime = list(re.finditer(
        r"^FIG15_UNIFORM_SERVER_PRIMED\t1$", text, re.MULTILINE
    ))
    object_load = list(re.finditer(r"^libbpf: loading object from .+$", text, re.MULTILINE))
    if len(prime) != 1 or len(object_load) != 1 or prime[0].start() >= object_load[0].start():
        raise AnalysisError(f"loader prime failed: {path}")
    if re.findall(r"^FIG15_UNIFORM_READY\t([^\t]+)\t1$", text, re.MULTILINE) != [arm]:
        raise AnalysisError(f"loader readiness failed: {path}")
    if len(re.findall(r"^FIG15_UNIFORM_DETACHED\t1$", text, re.MULTILINE)) != 1:
        raise AnalysisError(f"loader detach failed: {path}")
    raw = re.findall(
        r"^FIG15_UNIFORM_MAP\t([^\t]+)\t(\d+)\t(\d+)$", text, re.MULTILINE
    )
    parsed = {(name, int(key)): int(value) for name, key, value in raw}
    if len(parsed) != len(raw) or parsed != expected_map_rows(arm):
        raise AnalysisError(f"uniform map-effect oracle failed: {path}")


def read_execution(path: Path) -> int:
    rows = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(rows) != 2 or rows[0] != "target_pid\treturncode\tverifier_level":
        raise AnalysisError(f"execution record malformed: {path}")
    fields = rows[1].split("\t")
    if len(fields) != 3 or fields[1:] != ["0", "STRICT"]:
        raise AnalysisError(f"execution did not complete under STRICT: {path}")
    try:
        target_pid = int(fields[0])
    except ValueError as error:
        raise AnalysisError(f"invalid target PID: {path}") from error
    if target_pid <= 0:
        raise AnalysisError(f"non-positive target PID: {path}")
    return target_pid


def parse_strict(application_path: Path, target_pid: int, arm: str) -> None:
    text = application_path.read_text(encoding="utf-8", errors="replace")
    prefix = rf"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[{target_pid}\] "
    program = re.escape(PROGRAMS[arm])
    accepted = re.findall(
        prefix + rf"GPU eBPF verification accepted: mode=STRICT program={program} "
        + r"attach=kprobe/fig15_map_kernel instructions=([1-9][0-9]*)$",
        text, re.MULTILINE,
    )
    timing = re.findall(
        prefix + rf"GPU eBPF verification timing: program={program} "
        + r"verification_elapsed_ns=([1-9][0-9]*)$",
        text, re.MULTILINE,
    )
    maps = re.findall(
        prefix + rf"GPU eBPF verified map: program={program} fd=([0-9]+) "
        + r"type=([0-9]+) key_size=4 value_size=8 max_entries=1$",
        text, re.MULTILINE,
    )
    fragments = (
        "GPU eBPF verification accepted:", "GPU eBPF verification timing:",
        "GPU eBPF verified map:", "GPU eBPF verification failed",
        "Skipping GPU eBPF verification", "verifier unavailable",
    )
    target_records = [line for line in text.splitlines()
                      if f"][{target_pid}] " in line
                      and any(fragment in line for fragment in fragments)]
    if len(accepted) != 1 or len(timing) != 1:
        raise AnalysisError(f"STRICT acceptance/timing failed: {application_path}")
    if len(maps) != 3 or sorted(int(map_type) for _fd, map_type in maps) != [1503, 1503, 1513]:
        raise AnalysisError(f"STRICT map descriptors failed: {application_path}")
    if len(target_records) != 5:
        raise AnalysisError(f"unexpected verifier records: {application_path}")
    if any(fragment in text for fragment in (
        "GPU eBPF verification failed", "Skipping GPU eBPF verification",
        "verifier unavailable",
    )):
        raise AnalysisError(f"strict rejection/skip/unavailable: {application_path}")


def parse_engagement(application_path: Path, agent_path: Path, arm: str,
                     target_pid: int) -> None:
    application = application_path.read_text(encoding="utf-8", errors="replace")
    agent = agent_path.read_text(encoding="utf-8", errors="replace")
    required = {
        "target_transform": r"^\[ptxpass\] kprobe_entry_stub: matched=1, in=\d+, out=\d+$",
        "module_load": r"Loaded module: patched\.map_bench\.sm_120\.ptx",
        "attach": r"Attach successfully",
    }
    counts = {name: len(re.findall(pattern, application, re.MULTILINE))
              for name, pattern in required.items()}
    programs = re.findall(
        r"corresponding program ([A-Za-z0-9_]+) is cuda program", application
    )
    if any(value != 1 for value in counts.values()) or not programs or set(programs) != {PROGRAMS[arm]}:
        raise AnalysisError(f"target engagement failed: {application_path}")
    bootstrap = {
        "verifier_mode": r"Verifier mode: STRICT",
        "cuda_shm": r"Registered shared memory with CUDA:",
        "global_shm": r"Global shm constructed\. shm_open_type 1 for fig15_uniform_",
        "global_shm_ready": r"Global shm initialized",
    }
    if any(len(re.findall(pattern, agent)) != 1 for pattern in bootstrap.values()):
        raise AnalysisError(f"strict agent bootstrap failed: {agent_path}")
    parse_strict(application_path, target_pid, arm)
    if re.search(r"\[(?:error|critical)\]", application + "\n" + agent, re.IGNORECASE):
        raise AnalysisError(f"runtime error/critical record: {application_path}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median(values: list[float], confidence: float,
                     seed: int) -> tuple[float, float, float]:
    generator = random.Random(seed)
    samples = [
        statistics.median(generator.choice(values) for _ in values)
        for _ in range(10_000)
    ]
    alpha = (1.0 - confidence) / 2.0
    return (
        statistics.median(values), quantile(samples, alpha),
        quantile(samples, 1 - alpha),
    )


def effect(records: dict[tuple[int, str], float], left: str, right: str,
           confidence: float, seed: int) -> dict[str, float | int]:
    blocks = sorted({block for block, arm in records if arm == left})
    log_ratios = [math.log(records[(block, left)] / records[(block, right)])
                  for block in blocks]
    differences = [records[(block, left)] - records[(block, right)]
                   for block in blocks]
    log_mid, log_low, log_high = bootstrap_median(log_ratios, confidence, seed)
    delta_mid, delta_low, delta_high = bootstrap_median(
        differences, confidence, seed + 10_000
    )
    return {
        "pairs": len(blocks), "ratio": math.exp(log_mid),
        "ratio_low": math.exp(log_low), "ratio_high": math.exp(log_high),
        "delta_us": delta_mid, "delta_low_us": delta_low,
        "delta_high_us": delta_high, "confidence": confidence,
    }


def analyze_campaign(campaign: Path) -> dict[str, Any]:
    schedule = read_schedule(campaign / "schedule.tsv")
    phase = infer_phase(schedule)
    if schedule != expected_schedule(phase):
        raise AnalysisError("schedule differs from frozen seed-1797 design")
    validate_environment(campaign / "environment.txt")
    _blocks, warmup, launches = phase_parameters(phase)
    records: dict[tuple[int, str], float] = {}
    strict_cells = 0
    for item in schedule:
        arm = str(item["arm"])
        directory = campaign / (
            f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-{arm}"
        )
        key = (int(item["block"]), arm)
        if not directory.is_dir() or key in records:
            raise AnalysisError(f"missing or duplicate arm directory: {directory}")
        application = directory / "application.log"
        records[key] = parse_application(application, warmup, launches)
        if arm != "native":
            target_pid = read_execution(directory / "execution.tsv")
            parse_loader(directory / "loader.log", arm)
            parse_engagement(application, directory / "agent.log", arm, target_pid)
            strict_cells += 1
    if len(records) != len(schedule):
        raise AnalysisError("raw record count differs from schedule")

    arm_medians = {
        arm: statistics.median(
            value for (_block, record_arm), value in records.items()
            if record_arm == arm
        ) for arm in ARMS
    }
    if phase == "preflight":
        return {
            "run_status": "valid_preflight", "tested_hypothesis": "not_tested",
            "phase": phase, "raw_arm_processes": len(records),
            "strict_accepted_cells": strict_cells, "arm_median_us": arm_medians,
            "effects": {}, "descriptive": {},
        }

    effects = {
        "host_vs_device_update": effect(
            records, "host_update", "device_update", 0.975, SEED
        ),
        "host_vs_device_lookup": effect(
            records, "host_lookup", "device_lookup", 0.975, SEED + 1
        ),
    }
    descriptive = {
        "noop_vs_native": effect(records, "noop", "native", 0.95, SEED + 2),
        "device_update_vs_noop": effect(
            records, "device_update", "noop", 0.95, SEED + 3
        ),
        "device_lookup_vs_noop": effect(
            records, "device_lookup", "noop", 0.95, SEED + 4
        ),
    }
    primary = tuple(effects.values())
    if all(float(item["ratio_low"]) > 1.0 for item in primary):
        verdict = "supported"
    elif any(float(item["ratio_high"]) <= 1.0 for item in primary):
        verdict = "contradicted"
    else:
        verdict = "inconclusive"
    return {
        "run_status": "valid", "tested_hypothesis": verdict, "phase": phase,
        "raw_arm_processes": len(records), "strict_accepted_cells": strict_cells,
        "arm_median_us": arm_medians, "effects": effects,
        "descriptive": descriptive,
        "scope": (
            "one 32-thread block; constant key/value; scalar per-thread callbacks; "
            "STRICT verifier; RTX 5090; lookup and update only"
        ),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# RTX 5090 STRICT uniform-map analysis", "",
        f"- Run status: {analysis['run_status']}",
        f"- Tested hypothesis: {analysis['tested_hypothesis']}",
        f"- Raw arm processes replayed: {analysis['raw_arm_processes']}",
        f"- Attached cells with exact STRICT acceptance: {analysis['strict_accepted_cells']}",
        "", "## Arm latency", "", "| Arm | median us/launch |", "|---|---:|",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {analysis['arm_median_us'][arm]:.6f} |")
    if not analysis["effects"]:
        lines.extend(["", "Preflight establishes execution only; it is not a paper result."])
        return "\n".join(lines) + "\n"
    lines.extend([
        "", "## Co-primary paired effects", "",
        "| Comparison | pairs | ratio (97.5% interval) | delta us (97.5% interval) |",
        "|---|---:|---:|---:|",
    ])
    for name, item in analysis["effects"].items():
        lines.append(
            f"| {name} | {item['pairs']} | {item['ratio']:.4f} "
            f"[{item['ratio_low']:.4f}, {item['ratio_high']:.4f}] | "
            f"{item['delta_us']:.6f} [{item['delta_low_us']:.6f}, "
            f"{item['delta_high_us']:.6f}] |"
        )
    lines.extend([
        "", "## Descriptive controls", "",
        "| Comparison | pairs | ratio (95% interval) | delta us (95% interval) |",
        "|---|---:|---:|---:|",
    ])
    for name, item in analysis["descriptive"].items():
        lines.append(
            f"| {name} | {item['pairs']} | {item['ratio']:.4f} "
            f"[{item['ratio_low']:.4f}, {item['ratio_high']:.4f}] | "
            f"{item['delta_us']:.6f} [{item['delta_low_us']:.6f}, "
            f"{item['delta_high_us']:.6f}] |"
        )
    lines.extend([
        "", "The two placement intervals are Bonferroni-adjusted co-primary comparisons.",
        "The controls are descriptive and do not isolate a causal component cost.",
        "Map-effect readback is idempotent: it proves the final nonzero effect, not "
        "callback invocation cardinality or verifier soundness.", "",
        f"Scope: {analysis['scope']}",
    ])
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, analysis: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "comparison", "role", "pairs", "ratio", "ratio_low", "ratio_high",
            "delta_us", "delta_low_us", "delta_high_us", "confidence",
        ))
        for role, group in (("primary", analysis["effects"]),
                            ("descriptive", analysis["descriptive"])):
            for name, item in group.items():
                writer.writerow((
                    name, role, item["pairs"], item["ratio"], item["ratio_low"],
                    item["ratio_high"], item["delta_us"], item["delta_low_us"],
                    item["delta_high_us"], item["confidence"],
                ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    analysis = analyze_campaign(args.campaign)
    markdown = render_markdown(analysis)
    (args.campaign / "analysis.md").write_text(markdown, encoding="utf-8")
    write_tsv(args.campaign / "analysis.tsv", analysis)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

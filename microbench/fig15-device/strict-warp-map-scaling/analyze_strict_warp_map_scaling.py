#!/usr/bin/env python3
"""Replay raw logs from the frozen STRICT-admitted warp-key map-sharding run."""

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
SHAPES = (32, 128, 256, 512, 1024)
ARMS = ("native", "noop", "shared_update", "warp_update")
PROGRAMS = {
    "noop": "cuda__noop",
    "shared_update": "cuda__shared",
    "warp_update": "cuda__warp",
}
WARP_MAGIC = 0x57504d4150000000


class AnalysisError(RuntimeError):
    pass


def phase_parameters(phase: str) -> tuple[int, int, int]:
    if phase == "preflight":
        return 1, 1, 4
    if phase == "full":
        return 8, 8, 128
    raise ValueError(phase)


def arm_order(shape: int) -> tuple[str, ...]:
    ordered = list(ARMS)
    random.Random(SEED + shape).shuffle(ordered)
    return tuple(ordered)


def expected_schedule(phase: str) -> list[dict[str, int | str]]:
    blocks, _warmup, _launches = phase_parameters(phase)
    schedule: list[dict[str, int | str]] = []
    for shape in SHAPES:
        base = arm_order(shape)
        for block in range(blocks):
            cycle = base if block < len(ARMS) else tuple(reversed(base))
            offset = block % len(ARMS)
            order = cycle[offset:] + cycle[:offset]
            run_id = shape * 100 + block + 1
            for position, arm in enumerate(order):
                schedule.append({
                    "shape": shape,
                    "block": block + 1,
                    "order": position + 1,
                    "arm": arm,
                    "run_id": run_id,
                })
    return schedule


def read_schedule(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["shape", "block", "order", "arm", "run_id"]:
            raise AnalysisError("schedule columns changed")
        rows = []
        for row in reader:
            try:
                rows.append({
                    "shape": int(row["shape"]), "block": int(row["block"]),
                    "order": int(row["order"]), "arm": row["arm"],
                    "run_id": int(row["run_id"]),
                })
            except (TypeError, ValueError) as error:
                raise AnalysisError("malformed schedule row") from error
    return rows


def infer_phase(rows: list[dict[str, int | str]]) -> str:
    if len(rows) == len(SHAPES) * len(ARMS):
        return "preflight"
    if len(rows) == 8 * len(SHAPES) * len(ARMS):
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


def parse_application(path: Path, warmup: int, launches: int, shape: int) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    device = re.findall(r"^FIG15_DEVICE\t(.+)\t12\t0\t32$", text, re.MULTILINE)
    measurement = re.findall(
        r"^FIG15_MEASUREMENT\t(\d+)\t(\d+)\t([0-9.eE+-]+)$", text, re.MULTILINE,
    )
    correct = re.findall(rf"^FIG15_CORRECT\t{shape}\t0$", text, re.MULTILINE)
    if device != [GPU_NAME] or len(measurement) != 1:
        raise AnalysisError(f"application identity/timing missing: {path}")
    measured_warmup, measured_launches, elapsed = measurement[0]
    if (int(measured_warmup), int(measured_launches)) != (warmup, launches):
        raise AnalysisError(f"timing parameters changed: {path}")
    elapsed_ms = float(elapsed)
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise AnalysisError(f"invalid elapsed time: {path}")
    if not correct:
        raise AnalysisError(f"CUDA correctness failed: {path}")
    return elapsed_ms * 1000.0 / launches


def parse_loader(path: Path, shape: int, arm: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    prime = list(re.finditer(r"^FIG15_WARP_SERVER_PRIMED\t1$", text, re.MULTILINE))
    object_load = list(re.finditer(r"^libbpf: loading object from .+$", text, re.MULTILINE))
    if len(prime) != 1 or len(object_load) != 1 or prime[0].start() >= object_load[0].start():
        raise AnalysisError(f"loader prime failed: {path}")
    if not re.findall(rf"^FIG15_WARP_READY\t{re.escape(str(arm))}\t1$", text, re.MULTILINE):
        raise AnalysisError(f"loader readiness failed: {path}")
    if len(re.findall(r"^FIG15_WARP_DETACHED\t1$", text, re.MULTILINE)) != 1:
        raise AnalysisError(f"loader detach failed: {path}")
    rows = re.findall(r"^FIG15_WARP_MAP\t(\d+)\t(\d+)$", text, re.MULTILINE)
    parsed = {(int(key), int(value)) for key, value in rows}
    if arm == "noop":
        if parsed:
            raise AnalysisError(f"noop map-effect must remain empty: {path}")
        return
    if arm == "shared_update":
        if parsed != {(0, WARP_MAGIC)}:
            raise AnalysisError(f"shared-update map-effect failed: {path}")
        return
    if arm == "warp_update":
        active = max(1, shape // 32)
        if len(parsed) == 0:
            raise AnalysisError(f"warp-effect missing: {path}")
        if any(key >= 64 or value != (WARP_MAGIC ^ key) for key, value in parsed):
            raise AnalysisError(f"invalid warp key/value readback: {path}")
        if len(parsed) < active:
            raise AnalysisError(f"warp coverage below requested active warps: {path}")
        return
    raise AnalysisError(f"unexpected arm in loader log: {path}")


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
        + r"attach=kprobe/fig15_warp_map_kernel instructions=([1-9][0-9]*)$",
        text, re.MULTILINE,
    )
    timing = re.findall(
        prefix + rf"GPU eBPF verification timing: program={program} "
        + r"verification_elapsed_ns=([1-9][0-9]*)$",
        text, re.MULTILINE,
    )
    maps = re.findall(
        prefix + rf"GPU eBPF verified map: program={program} fd=([0-9]+) "
        + r"type=([0-9]+) key_size=4 value_size=8 max_entries=64$",
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
        raise AnalysisError("STRICT acceptance/timing failed")
    if len(maps) != 1 or int(maps[0][1]) != 1503:
        raise AnalysisError("STRICT map descriptor mismatch")
    if len(target_records) != 3:
        raise AnalysisError(f"unexpected target verifier record count: {application_path}")
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
        "module_load": r"Loaded module: patched\.warp_map_bench\.sm_120\.ptx",
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
        "global_shm": r"Global shm constructed\. shm_open_type 1 for fig15_warp_",
        "global_shm_ready": r"Global shm initialized",
    }
    if any(len(re.findall(pattern, agent)) != 1 for pattern in bootstrap.values()):
        raise AnalysisError(f"strict agent bootstrap failed: {agent_path}")
    parse_strict(application_path, target_pid, arm)
    if re.search(r"\[(?:error|critical)\]", application + "\n" + agent, re.IGNORECASE):
        raise AnalysisError(f"runtime error/critical record: {application_path}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("bootstrap quantile on empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median(values: list[float], confidence: float, seed: int) -> tuple[float, float, float]:
    if confidence <= 0.0 or confidence >= 1.0:
        raise AnalysisError(f"invalid confidence: {confidence}")
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


def paired_effect(records: dict[tuple[int, int, str], float], shape: int,
                 left: str, right: str, confidence: float, seed: int) -> dict[str, float | int]:
    pairs = [records[(shape, block, left)] / records[(shape, block, right)]
             for block in range(1, 9)]
    deltas = [records[(shape, block, left)] - records[(shape, block, right)]
              for block in range(1, 9)]
    log_pairs = [math.log(pair) for pair in pairs]
    log_mid, log_low, log_high = bootstrap_median(log_pairs, confidence, seed)
    delta_mid, delta_low, delta_high = bootstrap_median(
        deltas, confidence, seed + 1000
    )
    sign_count = sum(1 for value in deltas if value > 0.0)
    return {
        "shape": shape, "pairs": len(pairs), "ratio": math.exp(log_mid),
        "ratio_low": math.exp(log_low), "ratio_high": math.exp(log_high),
        "delta_us": delta_mid, "delta_low_us": delta_low,
        "delta_high_us": delta_high, "confidence": confidence,
        "sign_count": sign_count,
    }


def cross_scale_effect(records: dict[tuple[int, int, str], float], metric_left: str,
                      metric_right: str, seed: int) -> dict[str, float]:
    def shape_effect(shape: int) -> list[float]:
        return [records[(shape, block, metric_left)] / records[(shape, block, metric_right)]
                for block in range(1, 9)]
    low_shape, high_shape = min(SHAPES), max(SHAPES)
    low = [math.log(r) for r in shape_effect(low_shape)]
    high = [math.log(r) for r in shape_effect(high_shape)]
    deltas = [b - a for b, a in zip(high, low)]
    # compare against no change in log space
    mid, low_ci, high_ci = bootstrap_median(deltas, 0.95, seed)
    return {
        "low_shape": low_shape, "high_shape": high_shape,
        "factor": math.exp(mid), "factor_low": math.exp(low_ci),
        "factor_high": math.exp(high_ci), "pairs": len(deltas),
        "confidence": 0.95,
    }


def analyze_campaign(campaign: Path) -> dict[str, Any]:
    schedule = read_schedule(campaign / "schedule.tsv")
    phase = infer_phase(schedule)
    expected = expected_schedule(phase)
    if schedule != expected:
        raise AnalysisError("schedule differs from frozen seed-1797 design")
    validate_environment(campaign / "environment.txt")

    _blocks, warmup, launches = phase_parameters(phase)
    records: dict[tuple[int, int, str], float] = {}
    strict_cells = 0
    for item in schedule:
        shape = int(item["shape"])
        block = int(item["block"])
        arm = str(item["arm"])
        directory = campaign / (
            f"shape-{shape}-block-{block:02d}-order-{int(item['order']):02d}-{arm}"
        )
        key = (shape, block, arm)
        if not directory.is_dir() or key in records:
            raise AnalysisError(f"missing or duplicate arm directory: {directory}")
        application = directory / "application.log"
        records[key] = parse_application(application, warmup, launches, shape)
        if arm != "native":
            target_pid = read_execution(directory / "execution.tsv")
            parse_loader(directory / "loader.log", shape, arm)
            parse_engagement(application, directory / "agent.log", arm, target_pid)
            strict_cells += 1
    if len(records) != len(schedule):
        raise AnalysisError("raw record count differs from schedule")

    shape_arms = {
        shape: {arm: statistics.median(
            records[(shape, block, arm)] for block in range(1, _blocks + 1)
        ) for arm in ARMS} for shape in SHAPES
    }

    if phase == "preflight":
        return {
            "run_status": "valid_preflight",
            "tested_hypothesis": "not_tested",
            "phase": phase,
            "raw_arm_processes": len(records),
            "strict_accepted_cells": strict_cells,
            "shape_median_us": shape_arms,
            "effects": {}, "secondary": {}, "scope": "not_tested",
        }

    shape_effects: dict[str, dict[str, Any]] = {}
    for shape in SHAPES:
        for left, right, name in (
            ("shared_update", "noop", "shared_vs_noop"),
            ("warp_update", "noop", "warp_vs_noop"),
            ("warp_update", "shared_update", "warp_vs_shared"),
        ):
            key = f"shape-{shape}-{name}"
            shape_effects[key] = paired_effect(
                records, shape, left, right, 0.95, hash((shape, left, right)) & 0xFFFFFFFF
            )

    secondary = {
        "shared_scaling_1_to_32": cross_scale_effect(
            records, "shared_update", "noop", 1000 + SEED
        ),
        "warp_scaling_1_to_32": cross_scale_effect(
            records, "warp_update", "noop", 2000 + SEED
        ),
        "warp_vs_shared_scaling_1_to_32": cross_scale_effect(
            records, "warp_update", "shared_update", 3000 + SEED
        ),
    }

    # Inverted-direction count for warp-vs-shared per-shape
    sign_summary: dict[str, int] = {}
    for shape in SHAPES:
        key = f"shape-{shape}-warp_vs_shared"
        sign_summary[key] = int(shape_effects[key]["sign_count"])

    primary = tuple(shape_effects.values())
    if all(item["ratio_low"] > 1.0 for item in primary):
        verdict = "supported"
    elif any(item["ratio_high"] <= 1.0 for item in primary):
        verdict = "contradicted"
    else:
        verdict = "inconclusive"

    return {
        "run_status": "valid",
        "tested_hypothesis": verdict,
        "phase": phase,
        "raw_arm_processes": len(records),
        "strict_accepted_cells": strict_cells,
        "shape_median_us": shape_arms,
        "effects": shape_effects,
        "secondary": secondary,
        "shape_sign_summary": sign_summary,
        "scope": (
            "five shapes (32/128/256/512/1024 threads), STRICT admission, "
            "single-block launches, deterministic frozen order, per-lane callbacks, "
            "map-key-shard readback idempotent oracle, not per-warp aggregation"
        ),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# RTX 5090 STRICT warp-map scaling analysis",
        "",
        f"- Run status: {analysis['run_status']}",
        f"- Tested hypothesis: {analysis['tested_hypothesis']}",
        f"- Raw arm processes replayed: {analysis['raw_arm_processes']}",
        f"- Attached cells with exact STRICT acceptance: {analysis['strict_accepted_cells']}",
        "",
        "## Arm median latency (µs/launch)",
        "",
        "| Shape | native | noop | shared_update | warp_update |",
        "|---|---:|---:|---:|---:|",
    ]
    for shape in SHAPES:
        arm_values = analysis["shape_median_us"][shape]
        lines.append(
            f"| {shape} | {arm_values['native']:.6f} | {arm_values['noop']:.6f} | "
            f"{arm_values['shared_update']:.6f} | {arm_values['warp_update']:.6f} |"
        )

    if not analysis["effects"]:
        lines.extend([
            "",
            "Preflight establishes only execution; it is not a paper result.",
        ])
        return "\n".join(lines) + "\n"

    lines.extend([
        "",
        "## Per-shape paired primary effects",
        "",
        "| Shape | Comparison | pairs | ratio (95% CI) | delta us (95% CI) | sign(positive) |",
        "|---|---|---:|---:|---:|",
    ])
    for shape in SHAPES:
        for label in ("shared_vs_noop", "warp_vs_noop", "warp_vs_shared"):
            key = f"shape-{shape}-{label}"
            item = analysis["effects"][key]
            name = label.replace("_", " / ")
            lines.append(
                f"| {shape} | {name} | {item['pairs']} | "
                f"{item['ratio']:.4f} [{item['ratio_low']:.4f}, {item['ratio_high']:.4f}] | "
                f"{item['delta_us']:.6f} [{item['delta_low_us']:.6f}, {item['delta_high_us']:.6f}] | "
                f"{item['sign_count']}/8 |"
            )

    lines.extend([
        "",
        "## Cross-shape trend (from 32 to 1024 warps)",
        "",
        "| Comparison | pairs | factor (95% CI) |",
        "|---|---:|---:|",
    ])
    for key, item in analysis["secondary"].items():
        pretty = key.replace("shared_scaling_1_to_32", "shared_update/noop")
        pretty = pretty.replace("warp_scaling_1_to_32", "warp_update/noop")
        pretty = pretty.replace("warp_vs_shared_scaling_1_to_32", "warp_update/shared_update")
        lines.append(
            f"| {pretty} | {item['pairs']} | "
            f"{item['factor']:.4f} [{item['factor_low']:.4f}, {item['factor_high']:.4f}] |"
        )

    lines.extend([
        "",
        f"Scope: {analysis['scope']}",
    ])
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, analysis: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "scope", "shape", "comparison", "pairs", "ratio",
            "ratio_low", "ratio_high", "delta_us", "delta_low_us",
            "delta_high_us", "sign_count", "confidence",
        ))
        for shape in SHAPES:
            for left, right, name in (
                ("shared_update", "noop", "shared_vs_noop"),
                ("warp_update", "noop", "warp_vs_noop"),
                ("warp_update", "shared_update", "warp_vs_shared"),
            ):
                item = analysis["effects"][f"shape-{shape}-{name}"]
                writer.writerow((
                    "paired_shape", shape, f"{left}/{right}",
                    item["pairs"], item["ratio"], item["ratio_low"], item["ratio_high"],
                    item["delta_us"], item["delta_low_us"], item["delta_high_us"],
                    item["sign_count"], item["confidence"],
                ))
        for name, item in analysis["secondary"].items():
            writer.writerow((
                "cross_shape", "32->1024", name,
                item["pairs"], item["factor"], item["factor_low"], item["factor_high"],
                "", "", "", "", item["confidence"],
            ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    analysis = analyze_campaign(args.campaign)
    (args.campaign / "analysis.md").write_text(render_markdown(analysis), encoding="utf-8")
    write_tsv(args.campaign / "analysis.tsv", analysis)
    print(render_markdown(analysis), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

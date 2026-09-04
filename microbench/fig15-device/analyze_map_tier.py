#!/usr/bin/env python3
"""Independently replay raw logs from the frozen device-map placement run."""

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
    "native",
    "noop",
    "device_update",
    "host_update",
    "rpc_update",
    "device_lookup",
    "host_lookup",
    "rpc_lookup",
)
PROGRAM_PREFIXES = {
    "noop": "cuda__noop",
    "device_update": "cuda__device_up",
    "host_update": "cuda__host_upda",
    "rpc_update": "cuda__rpc_updat",
    "device_lookup": "cuda__device_lo",
    "host_lookup": "cuda__host_look",
    "rpc_lookup": "cuda__rpc_looku",
}
UPDATE_MAGIC = 0x51A7000000000000
LOOKUP_MAGIC = 0x10C4000000000000


class AnalysisError(RuntimeError):
    pass


def phase_parameters(phase: str) -> tuple[int, int, int]:
    if phase == "preflight":
        return 1, 1, 2
    if phase == "full":
        return 16, 8, 64
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
            result.append(
                {"block": block + 1, "order": position + 1, "arm": arm,
                 "run_id": block + 1}
            )
    return result


def read_schedule(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["block", "order", "arm", "run_id"]:
            raise AnalysisError("schedule columns changed")
        rows = []
        for row in reader:
            try:
                rows.append(
                    {"block": int(row["block"]), "order": int(row["order"]),
                     "arm": row["arm"], "run_id": int(row["run_id"])}
                )
            except (TypeError, ValueError) as error:
                raise AnalysisError("malformed schedule row") from error
    return rows


def infer_phase(rows: list[dict[str, int | str]]) -> str:
    if len(rows) == len(ARMS):
        return "preflight"
    if len(rows) == 16 * len(ARMS):
        return "full"
    raise AnalysisError(f"unexpected schedule length: {len(rows)}")


def validate_environment(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        f"gpu\t{GPU_NAME}\n",
        f"driver\t{DRIVER}\n",
        "BPFTIME_ENABLE_CUDA_ATTACH\tON\n",
        "BPFTIME_LLVM_JIT\tON\n",
        "ENABLE_EBPF_VERIFIER\tOFF\n",
        "bpftime_revision\t",
        "nvcc_begin\n",
        "nvcc_end\n",
    )
    if any(marker not in text for marker in required):
        raise AnalysisError("environment record is incomplete or off-plan")


def parse_application(path: Path, warmup: int, launches: int) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    device = re.findall(r"^FIG15_DEVICE\t(.+)\t12\t0\t32$", text, re.MULTILINE)
    measurement = re.findall(
        r"^FIG15_MEASUREMENT\t(\d+)\t(\d+)\t([0-9.eE+-]+)$",
        text, re.MULTILINE,
    )
    correctness = re.findall(r"^FIG15_CORRECT\t(\d+)\t(\d+)$", text,
                             re.MULTILINE)
    if device != [GPU_NAME]:
        raise AnalysisError(f"wrong device record: {path}")
    if len(measurement) != 1:
        raise AnalysisError(f"wrong measurement count: {path}")
    measured_warmup, measured_launches, elapsed = measurement[0]
    if (int(measured_warmup), int(measured_launches)) != (warmup, launches):
        raise AnalysisError(f"timing parameters changed: {path}")
    elapsed_ms = float(elapsed)
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise AnalysisError(f"invalid elapsed time: {path}")
    if correctness != [("32", "0")]:
        raise AnalysisError(f"CUDA correctness failed: {path}")
    if re.search(r"\[(?:error|critical)\]", text, re.IGNORECASE):
        raise AnalysisError(f"runtime error/critical record: {path}")
    return elapsed_ms * 1000.0 / launches


def parse_engagement(application_path: Path, agent_path: Path, arm: str) -> None:
    """Independently bind the selected arm to the transformed application."""
    application = application_path.read_text(encoding="utf-8", errors="replace")
    agent = agent_path.read_text(encoding="utf-8", errors="replace")
    combined = application + "\n" + agent
    required = {
        "target_transform": (
            r"^\[ptxpass\] kprobe_entry_stub: matched=1, "
            r"in=\d+, out=\d+$"
        ),
        "module_load": r"Loaded module: patched\.map_bench\.sm_120\.ptx",
        "attach": r"Attach successfully",
    }
    counts = {name: len(re.findall(pattern, application, re.MULTILINE))
              for name, pattern in required.items()}
    programs = re.findall(
        r"corresponding program ([A-Za-z0-9_]+) is cuda program", application,
    )
    expected_program = PROGRAM_PREFIXES[arm]
    if any(value != 1 for value in counts.values()) or not programs or \
            set(programs) != {expected_program}:
        raise AnalysisError(
            f"target engagement failed in {application_path.parent}: "
            f"arm={arm}, expected_program={expected_program}, "
            f"programs={programs}, counts={counts}"
        )
    bootstrap = {
        "verifier_mode": r"Verifier mode: WARNING",
        "cuda_shm": r"Registered shared memory with CUDA:",
        "global_shm": r"Global shm constructed\. shm_open_type 1 for fig15_map_",
        "global_shm_ready": r"Global shm initialized",
    }
    bootstrap_counts = {
        name: len(re.findall(pattern, agent))
        for name, pattern in bootstrap.items()
    }
    if any(value != 1 for value in bootstrap_counts.values()):
        raise AnalysisError(
            f"agent bootstrap failed in {agent_path}: {bootstrap_counts}"
        )
    if re.search(r"\[(?:error|critical)\]", combined, re.IGNORECASE):
        raise AnalysisError(
            f"runtime error/critical record: {application_path.parent}"
        )


def expected_map(arm: str) -> tuple[str, dict[int, int]] | None:
    if arm == "noop":
        return None
    if arm.endswith("_lookup"):
        return "observed_values", {key: LOOKUP_MAGIC ^ key for key in range(32)}
    name = {
        "device_update": "device_values",
        "host_update": "host_values",
        "rpc_update": "rpc_values",
    }[arm]
    return name, {key: UPDATE_MAGIC ^ key for key in range(32)}


def parse_loader(path: Path, arm: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    prime = list(re.finditer(r"^FIG15_SERVER_PRIMED\t1$", text, re.MULTILINE))
    object_load = list(re.finditer(
        r"^libbpf: loading object from .+$", text, re.MULTILINE,
    ))
    if len(prime) != 1 or len(object_load) != 1 or \
            prime[0].start() >= object_load[0].start():
        raise AnalysisError(f"loader syscall-server prime failed: {path}")
    if re.findall(r"^FIG15_READY\t([^\t]+)\t1$", text, re.MULTILINE) != [arm]:
        raise AnalysisError(f"loader readiness failed: {path}")
    if re.findall(r"^FIG15_DETACHED\t1$", text, re.MULTILINE) != ["FIG15_DETACHED\t1"]:
        # re.findall returns the whole match only when the expression has no group.
        raise AnalysisError(f"loader detach failed: {path}")
    raw = re.findall(r"^FIG15_MAP\t([^\t]+)\t(\d+)\t(\d+)$", text,
                     re.MULTILINE)
    parsed = {(name, int(key)): int(value) for name, key, value in raw}
    if len(parsed) != len(raw):
        raise AnalysisError(f"duplicate map rows: {path}")
    expectation = expected_map(arm)
    if expectation is None:
        if parsed:
            raise AnalysisError(f"no-op emitted map data: {path}")
    else:
        name, values = expectation
        expected = {(name, key): value for key, value in values.items()}
        if parsed != expected:
            raise AnalysisError(f"map oracle failed: {path}")


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
    return statistics.median(values), quantile(samples, alpha), quantile(samples, 1 - alpha)


def effect(records: dict[tuple[int, str], float], left: str, right: str,
           confidence: float, seed: int) -> dict[str, float | int]:
    blocks = sorted({block for block, arm in records if arm == left})
    log_ratios = [math.log(records[(block, left)] / records[(block, right)])
                  for block in blocks]
    differences = [records[(block, left)] - records[(block, right)]
                   for block in blocks]
    log_mid, log_low, log_high = bootstrap_median(
        log_ratios, confidence, seed,
    )
    delta_mid, delta_low, delta_high = bootstrap_median(
        differences, confidence, seed + 10_000,
    )
    return {
        "pairs": len(blocks),
        "ratio": math.exp(log_mid),
        "ratio_low": math.exp(log_low),
        "ratio_high": math.exp(log_high),
        "delta_us": delta_mid,
        "delta_low_us": delta_low,
        "delta_high_us": delta_high,
        "confidence": confidence,
    }


def paired_delta(records: dict[tuple[int, str], float], left: str, right: str,
                 confidence: float, seed: int) -> dict[str, float | int]:
    blocks = sorted({block for block, arm in records if arm == left})
    differences = [records[(block, left)] - records[(block, right)]
                   for block in blocks]
    middle, low, high = bootstrap_median(differences, confidence, seed)
    return {
        "pairs": len(blocks),
        "delta_us": middle,
        "delta_low_us": low,
        "delta_high_us": high,
        "confidence": confidence,
    }


def analyze_campaign(campaign: Path) -> dict[str, Any]:
    schedule = read_schedule(campaign / "schedule.tsv")
    phase = infer_phase(schedule)
    if schedule != expected_schedule(phase):
        raise AnalysisError("schedule differs from the frozen seed-1797 design")
    validate_environment(campaign / "environment.txt")
    _blocks, warmup, launches = phase_parameters(phase)
    records: dict[tuple[int, str], float] = {}
    for item in schedule:
        arm = str(item["arm"])
        directory = campaign / (
            f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-{arm}"
        )
        if not directory.is_dir():
            raise AnalysisError(f"missing arm directory: {directory}")
        key = (int(item["block"]), arm)
        if key in records:
            raise AnalysisError(f"duplicate block/arm: {key}")
        application_path = directory / "application.log"
        records[key] = parse_application(application_path, warmup, launches)
        if arm != "native":
            parse_loader(directory / "loader.log", arm)
            agent = directory / "agent.log"
            if not agent.is_file() or not agent.read_text(
                encoding="utf-8", errors="replace"
            ).strip():
                raise AnalysisError(f"missing agent bootstrap log: {agent}")
            parse_engagement(application_path, agent, arm)
    if len(records) != len(schedule):
        raise AnalysisError("raw record count differs from schedule")

    arm_medians = {
        arm: statistics.median(
            value for (block, record_arm), value in records.items()
            if record_arm == arm
        )
        for arm in ARMS
    }
    if phase == "preflight":
        return {
            "run_status": "valid_preflight",
            "tested_hypothesis": "not_tested",
            "phase": phase,
            "raw_arm_processes": len(records),
            "arm_median_us": arm_medians,
            "effects": {},
            "descriptive_deltas": {},
        }

    effects = {
        "host_vs_device_update": effect(
            records, "host_update", "device_update", 0.975, SEED,
        ),
        "host_vs_device_lookup": effect(
            records, "host_lookup", "device_lookup", 0.975, SEED + 1,
        ),
        "rpc_vs_device_update": effect(
            records, "rpc_update", "device_update", 0.95, SEED + 2,
        ),
        "rpc_vs_device_lookup": effect(
            records, "rpc_lookup", "device_lookup", 0.95, SEED + 3,
        ),
        "noop_vs_native": effect(records, "noop", "native", 0.95, SEED + 4),
    }
    descriptive_deltas = {
        "device_update_minus_noop": paired_delta(
            records, "device_update", "noop", 0.95, SEED + 5,
        ),
        "device_lookup_minus_noop": paired_delta(
            records, "device_lookup", "noop", 0.95, SEED + 6,
        ),
    }
    primary = (
        effects["host_vs_device_update"],
        effects["host_vs_device_lookup"],
    )
    if all(float(item["ratio_low"]) > 1.0 for item in primary):
        verdict = "supported"
    elif any(float(item["ratio_high"]) <= 1.0 for item in primary):
        verdict = "contradicted"
    else:
        verdict = "inconclusive"
    return {
        "run_status": "valid",
        "tested_hypothesis": verdict,
        "phase": phase,
        "raw_arm_processes": len(records),
        "arm_median_us": arm_medians,
        "effects": effects,
        "descriptive_deltas": descriptive_deltas,
        "scope": (
            "one 32-thread block; current verification-disabled scalar per-thread "
            "runtime; RTX 5090; lookup and update only"
        ),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# RTX 5090 device-map placement analysis",
        "",
        f"- Run status: {analysis['run_status']}",
        f"- Tested hypothesis: {analysis['tested_hypothesis']}",
        f"- Raw arm processes replayed: {analysis['raw_arm_processes']}",
        "",
        "## Arm latency",
        "",
        "| Arm | median us/launch |",
        "|---|---:|",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {analysis['arm_median_us'][arm]:.6f} |")
    if analysis["effects"]:
        lines.extend(
            [
                "",
                "## Paired effects",
                "",
                "| Comparison | pairs | ratio (interval) | delta us (interval) | confidence |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, item in analysis["effects"].items():
            lines.append(
                f"| {name} | {item['pairs']} | {item['ratio']:.4f} "
                f"[{item['ratio_low']:.4f}, {item['ratio_high']:.4f}] | "
                f"{item['delta_us']:.6f} [{item['delta_low_us']:.6f}, "
                f"{item['delta_high_us']:.6f}] | {100 * item['confidence']:.1f}% |"
            )
        lines.extend(
            [
                "",
                "## Descriptive device/no-op deltas",
                "",
                "| Comparison | pairs | median delta us (interval) | confidence |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, item in analysis["descriptive_deltas"].items():
            lines.append(
                f"| {name} | {item['pairs']} | {item['delta_us']:.6f} "
                f"[{item['delta_low_us']:.6f}, {item['delta_high_us']:.6f}] | "
                f"{100 * item['confidence']:.1f}% |"
            )
        lines.extend(
            [
                "",
                "The two host-mapped/device-resident intervals are the Bonferroni-adjusted "
                "co-primary comparisons. RPC and no-op comparisons are descriptive.",
                "The device-minus-no-op values are within-block arithmetic contrasts only. "
                "A negative value does not estimate a negative mechanism cost or a causal "
                "benefit from adding the map operation.",
                "",
                f"Scope: {analysis['scope']}",
            ]
        )
    else:
        lines.extend(["", "Preflight establishes execution only; it is not a paper result."])
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, analysis: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("comparison", "pairs", "ratio", "ratio_low", "ratio_high",
                         "delta_us", "delta_low_us", "delta_high_us", "confidence"))
        for name, item in analysis["effects"].items():
            writer.writerow(
                (name, item["pairs"], item["ratio"], item["ratio_low"],
                 item["ratio_high"], item["delta_us"], item["delta_low_us"],
                 item["delta_high_us"], item["confidence"])
            )
        for name, item in analysis["descriptive_deltas"].items():
            writer.writerow(
                (name, item["pairs"], "", "", "", item["delta_us"],
                 item["delta_low_us"], item["delta_high_us"],
                 item["confidence"])
            )


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

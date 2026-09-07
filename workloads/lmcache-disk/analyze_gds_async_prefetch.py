#!/usr/bin/env python3
"""CPU-only analyzer for a gds-async-prefetch-575 raw.jsonl (complete or partial).

Reads one record per cell and reports, on stdout:

- arm median throughput (median across blocks of
  ``warm_phase.output_tokens_per_s``) and arm TTFT median (median across
  blocks of ``warm_phase.warm_ttft_median_ms``);
- mean E2E per cell over ``requests[phase=warm].e2e_ms``;
- paired change by block against the reference arm (``--reference``,
  default the demand arm), as absolute deltas (candidate minus reference)
  plus paired percentage deltas ``100*(candidate/reference - 1)``, each with
  median/range and the count of improving blocks.

Paired aggregation uses same-block candidate/reference pairs only: an incomplete
block pair (either side missing a required value) is rejected from pairing,
while every individual observation remains in the arm medians and the
per-cell table.  A short-circuited or incomplete raw file is reported with a
PARTIAL status line, never dropped or gated.  Markdown and/or JSON are
emitted to stdout for root redirection.  No GPU, no network, stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RAW_NAME = "raw.jsonl"
DEFAULT_REFERENCE = "gds_demand_fifo"
KNOWN_ARMS = ("gds_demand_fifo", "gds_eager_async", "gds_async_native",
              "gds_async_bpf")


def _num(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_cells(raw_path: Path) -> list[dict[str, object]]:
    """Parse raw.jsonl into per-cell rows using only the required fields."""
    cells: list[dict[str, object]] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{raw_path} line {line_no}: {error}") from error
            block = record.get("block")
            config = record.get("config")
            if not isinstance(block, int) or not isinstance(config, str):
                raise ValueError(
                    f"{raw_path} line {line_no}: missing integer block or string config")
            warm_phase = record.get("warm_phase")
            warm_phase = warm_phase if isinstance(warm_phase, dict) else {}
            e2e: list[float] = []
            for entry in record.get("requests") or []:
                if not isinstance(entry, dict) or entry.get("phase") != "warm":
                    continue
                e2e.append(_num(entry.get("e2e_ms")))
                e2e = [v for v in e2e if v is not None]
            cells.append({
                "block": block,
                "config": config,
                "tps": _num(warm_phase.get("output_tokens_per_s")),
                "ttft_median_ms": _num(warm_phase.get("warm_ttft_median_ms")),
                "warm_e2e_ms": e2e,
            })
    return cells


def merge_cells(cells: list[dict[str, object]]) -> dict[tuple[int, str], dict[str, object]]:
    """Collapse duplicate (block, config) records.

    Warm E2E observations are concatenated; the scalar tps/TTFT values keep
    the first non-null record (later duplicates never replace them).
    Duplicates are reported by the caller, not claimed retained.
    """
    merged: dict[tuple[int, str], dict[str, object]] = {}
    for cell in cells:
        key = (cell["block"], cell["config"])
        row = merged.setdefault(key, {
            "block": cell["block"], "config": cell["config"],
            "tps": None, "ttft_median_ms": None, "warm_e2e_ms": []})
        if row["tps"] is None and cell["tps"] is not None:
            row["tps"] = cell["tps"]
        if row["ttft_median_ms"] is None and cell["ttft_median_ms"] is not None:
            row["ttft_median_ms"] = cell["ttft_median_ms"]
        row["warm_e2e_ms"].extend(cell["warm_e2e_ms"])
    return merged


def duplicate_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[tuple[int, str], int] = {}
    for cell in cells:
        key = (cell["block"], cell["config"])
        seen[key] = seen.get(key, 0) + 1
    return sorted(
        {"block": block, "config": config, "records": count}
        for (block, config), count in seen.items() if count > 1)


def arm_rows(merged: dict[tuple[int, str], dict[str, object]],
             arm: str, blocks: list[int]) -> list[dict[str, object]]:
    rows = []
    for block in blocks:
        row = merged.get((block, arm))
        e2e = list(row["warm_e2e_ms"]) if row else []
        rows.append({
            "block": block,
            "present": row is not None,
            "tps": row["tps"] if row else None,
            "ttft_median_ms": row["ttft_median_ms"] if row else None,
            "e2e_mean_ms": (statistics.mean(e2e) if e2e else None),
            "warm_e2e_ms": e2e,
        })
    return rows


def paired_changes(base_rows: list[dict[str, object]],
                   arm_rows_: list[dict[str, object]]) -> dict[str, object]:
    """Same-block candidate/reference pairs; incomplete pairs rejected, not pooled.

    Emits the absolute delta (candidate - reference) and the paired
    percentage ``100*(candidate/reference - 1)`` for every metric.  The
    improving count is defined on the absolute-delta direction (lower TTFT /
    mean E2E or higher throughput is better).
    """
    per_block: dict[str, dict[str, float | None]] = {}
    incomplete: list[int] = []
    for base, other in zip(base_rows, arm_rows_, strict=True):
        metric_defs = (
            ("tps_delta", "tps_pct", other["tps"], base["tps"]),
            ("ttft_delta_ms", "ttft_pct", other["ttft_median_ms"],
             base["ttft_median_ms"]),
            ("e2e_mean_delta_ms", "e2e_mean_pct", other["e2e_mean_ms"],
             base["e2e_mean_ms"]),
        )
        deltas: dict[str, float | None] = {}
        for abs_name, pct_name, candidate, reference in metric_defs:
            if candidate is not None and reference is not None:
                deltas[abs_name] = candidate - reference
                deltas[pct_name] = (100.0 * (candidate / reference - 1.0)
                                    if reference != 0 else None)
            else:
                deltas[abs_name] = None
                deltas[pct_name] = None
        if any(v is None for name, v in deltas.items()
               if name in ("tps_delta", "ttft_delta_ms", "e2e_mean_delta_ms")):
            incomplete.append(base["block"])
            continue
        per_block[str(base["block"])] = deltas

    summary: dict[str, object] = {}
    for abs_name, pct_name, positive in (
            ("ttft_delta_ms", "ttft_pct", False),
            ("e2e_mean_delta_ms", "e2e_mean_pct", False),
            ("tps_delta", "tps_pct", True)):
        abs_vals = [v[abs_name] for v in per_block.values()
                    if v[abs_name] is not None]
        pct_vals = [v[pct_name] for v in per_block.values()
                    if v[pct_name] is not None]
        wins = sum(1 for v in abs_vals if (v > 0 if positive else v < 0))
        summary[abs_name] = {
            "abs": {
                "range": [min(abs_vals), max(abs_vals)] if abs_vals else None,
                "pairs": len(abs_vals),
                "improving_blocks": wins,
                "positive_is_better": positive,
            },
            "pct": {
                "median": statistics.median(pct_vals) if pct_vals else None,
                "range": [min(pct_vals), max(pct_vals)] if pct_vals else None,
                "pairs": len(pct_vals),
                "improving_blocks": wins,
            },
        }
    return {
        "per_block": per_block,
        "incomplete_blocks": incomplete,
        "summary": summary,
    }


def analyze(raw_path: Path, blocks_expected: int,
            reference: str = DEFAULT_REFERENCE) -> dict[str, object]:
    cells = load_cells(raw_path)
    merged = merge_cells(cells)
    duplicates = duplicate_cells(cells)
    arms = list(KNOWN_ARMS)
    for config in sorted({row["config"] for row in merged.values()}):
        if config not in arms:
            arms.append(config)
    observed_blocks = sorted({row["block"] for row in merged.values()})
    blocks = sorted(set(observed_blocks) | set(range(blocks_expected)))

    rows_by_arm = {arm: arm_rows(merged, arm, blocks) for arm in arms}
    per_arm: dict[str, object] = {}
    for arm in arms:
        rows = rows_by_arm[arm]
        tps_vals = [r["tps"] for r in rows if r["tps"] is not None]
        ttft_vals = [r["ttft_median_ms"] for r in rows if r["ttft_median_ms"] is not None]
        per_arm[arm] = {
            "tps_per_block": {str(r["block"]): r["tps"] for r in rows if r["tps"] is not None},
            "ttft_median_ms_per_block": {str(r["block"]): r["ttft_median_ms"]
                                         for r in rows if r["ttft_median_ms"] is not None},
            "median_tps": statistics.median(tps_vals) if tps_vals else None,
            "median_ttft_ms": statistics.median(ttft_vals) if ttft_vals else None,
            "blocks_measured": sum(
                r["tps"] is not None and r["ttft_median_ms"] is not None
                and r["e2e_mean_ms"] is not None for r in rows),
        }

    cell_table = [
        {"block": r["block"], "config": arm,
         "tps": r["tps"], "ttft_median_ms": r["ttft_median_ms"],
         "e2e_mean_ms": r["e2e_mean_ms"], "warm_e2e_n": len(r["warm_e2e_ms"]),
         "warm_e2e_ms": r["warm_e2e_ms"]}
        for arm in arms for r in rows_by_arm[arm]
    ]

    paired: dict[str, object] = {}
    if reference in rows_by_arm:
        for arm in arms:
            if arm != reference:
                paired[arm] = paired_changes(rows_by_arm[reference], rows_by_arm[arm])

    expected_cells = blocks_expected * len(arms)
    missing = sorted(
        f"block {b}/{arm}" for b in range(blocks_expected) for arm in arms
        if (b, arm) not in merged)
    measured = sum(row["blocks_measured"] for row in per_arm.values())
    complete = not missing and all(
        r["tps"] is not None and r["ttft_median_ms"] is not None
        and r["warm_e2e_ms"] for arm in arms for r in rows_by_arm[arm])
    status = "complete" if complete else "partial"
    return {
        "status": status,
        "raw_file": str(raw_path),
        "expected_cells": expected_cells,
        "cells_present": len(merged),
        "cells_measured": measured,
        "missing_cells": missing,
        "blocks": blocks,
        "arms": arms,
        "reference_arm": reference,
        "duplicate_cells": duplicates,
        "pairing_rule": (
            "paired aggregation uses same-block candidate/reference pairs "
            "only; an incomplete block pair is rejected from pairing, never "
            "pooled; all individual observations remain in arm medians and "
            "the per-cell table"),
        "fields_used": [
            "block", "config",
            "warm_phase.output_tokens_per_s",
            "warm_phase.warm_ttft_median_ms",
            "requests[phase=warm].e2e_ms",
        ],
        "per_arm": per_arm,
        "cell_table": cell_table,
        "paired_vs_reference": paired,
    }


def _f(value: float | None, fmt: str = "{:.1f}") -> str:
    return fmt.format(value) if value is not None else "-"


def render_markdown(result: dict[str, object]) -> str:
    lines: list[str] = []
    status = str(result["status"]).upper()
    lines.append("# GDS async-prefetch 575 analysis")
    lines.append("")
    lines.append(f"- raw: `{result['raw_file']}`")
    lines.append(
        f"- status: **{status}** "
        f"({result['cells_present']}/{result['expected_cells']} expected cells, "
        f"{result['cells_measured']} fully measured)")
    if result["missing_cells"]:
        lines.append(f"- missing cells: {', '.join(map(str, result['missing_cells']))}")
    if result["duplicate_cells"]:
        lines.append(
            "- duplicate (block, config) records: "
            + ", ".join(f"block {d['block']}/{d['config']} x{d['records']}"
                        for d in result["duplicate_cells"])
            + " (warm E2E merged; tps/TTFT keep the first non-null record)")
    lines.append(f"- pairing: {result['pairing_rule']}")
    lines.append("")
    lines.append("## Arm medians across blocks")
    lines.append("")
    lines.append("| arm | median throughput (tok/s) | median TTFT (ms) | blocks measured |")
    lines.append("|---|---|---|---|")
    for arm, row in result["per_arm"].items():
        lines.append(
            f"| {arm} | {_f(row['median_tps'])} | {_f(row['median_ttft_ms'])} "
            f"| {row['blocks_measured']}/{len(result['blocks'])} |")
    lines.append("")
    lines.append("## Per-cell values")
    lines.append("")
    lines.append("| block | arm | tps (tok/s) | TTFT median (ms) | mean E2E (ms) | warm n |")
    lines.append("|---|---|---|---|---|---|")
    for row in result["cell_table"]:
        lines.append(
            f"| {row['block']} | {row['config']} | {_f(row['tps'])} "
            f"| {_f(row['ttft_median_ms'])} | {_f(row['e2e_mean_ms'])} "
            f"| {row['warm_e2e_n']} |")
    lines.append("")
    lines.append(f"## Paired change by block vs `{result['reference_arm']}`")
    for arm, paired in result["paired_vs_reference"].items():
        lines.append("")
        lines.append(f"### {arm}")
        lines.append("")
        lines.append("| block | dTTFT (ms) | dTTFT (%) | dmean E2E (ms) | "
                     "dmean E2E (%) | dtok/s | dtok/s (%) |")
        lines.append("|---|---|---|---|---|---|---|")
        for block in sorted(paired["per_block"], key=int):
            d = paired["per_block"][block]
            lines.append(
                f"| {block} | {_f(d['ttft_delta_ms'], '{:+.1f}')} "
                f"| {_f(d['ttft_pct'], '{:+.2f}')} "
                f"| {_f(d['e2e_mean_delta_ms'], '{:+.1f}')} "
                f"| {_f(d['e2e_mean_pct'], '{:+.2f}')} "
                f"| {_f(d['tps_delta'], '{:+.1f}')} "
                f"| {_f(d['tps_pct'], '{:+.2f}')} |")
        if paired["incomplete_blocks"]:
            lines.append(
                f"- incomplete block pairs rejected from pairing: "
                f"{', '.join(map(str, paired['incomplete_blocks']))}")
        for name, label in (("ttft_delta_ms", "TTFT"), ("e2e_mean_delta_ms", "mean E2E"),
                            ("tps_delta", "throughput")):
            info = paired["summary"][name]
            abs_info, pct_info = info["abs"], info["pct"]
            if abs_info["range"] is None:
                lines.append(f"- {label}: no complete block pairs")
                continue
            low, high = abs_info["range"]
            if pct_info["range"] is None:
                pct_part = "; pct unavailable (zero reference)"
            else:
                plo, phi = pct_info["range"]
                pct_part = (
                    f"; pct median {pct_info['median']:+.2f}%, range "
                    f"{plo:+.2f}% to {phi:+.2f}%; improving blocks "
                    f"{pct_info['improving_blocks']}/{pct_info['pairs']}")
            lines.append(
                f"- {label}: abs {low:+.1f} to {high:+.1f} over "
                f"{abs_info['pairs']} complete pairs; improving blocks "
                f"{abs_info['improving_blocks']}/{abs_info['pairs']}{pct_part}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a gds-async-prefetch-575 raw.jsonl (CPU only, stdlib).")
    parser.add_argument("raw", type=Path,
                        help="path to raw.jsonl or to its containing directory")
    parser.add_argument("--blocks", type=int, default=5,
                        help="expected rotated blocks (default 5)")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help=f"reference arm for paired changes "
                             f"(default {DEFAULT_REFERENCE})")
    parser.add_argument("--format", choices=("markdown", "json", "both"),
                        default="markdown", dest="format")
    args = parser.parse_args(argv)
    raw_path = args.raw
    if raw_path.is_dir():
        raw_path = raw_path / RAW_NAME
    if not raw_path.is_file():
        print(f"NOT FOUND: {raw_path}", file=sys.stderr)
        return 2
    try:
        result = analyze(raw_path, args.blocks, args.reference)
    except (ValueError, OSError) as error:
        print(f"PARSE ERROR: {error}", file=sys.stderr)
        return 2
    if result["duplicate_cells"]:
        print(f"WARNING: duplicate (block, config) records "
              f"{result['duplicate_cells']}; warm E2E observations are merged, "
              "tps/TTFT keep the first non-null record", file=sys.stderr)
    if not result["paired_vs_reference"]:
        print(f"WARNING: reference arm {args.reference!r} absent from the raw "
              "file; no paired comparisons", file=sys.stderr)
    if args.format in ("markdown", "both"):
        print(render_markdown(result))
    if args.format in ("json", "both"):
        print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

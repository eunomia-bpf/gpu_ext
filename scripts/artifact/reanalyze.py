#!/usr/bin/env python3
"""CPU-only reanalysis entrypoint for completed raw evidence.

Recomputes cell metrics and paired statistics from existing raw
result.json records. Four campaigns are supported; the default
(``--campaign all``) includes all four (70 cells: the existing 55
plus 15 MoE):

  lm       LMCache disk physical-reclaim serving, pXYN4F
  xsched   XSched Level-2 device policy pair, buBjns (+ reused bOiYh5
           native block 0)
  storage  LMCache GDS write-budget serving, five-block-02
           (25 cells: 5 blocks x fifo/native10/bpf10/native200/bpf200)
  moe      MoE paper-v3-575 postboot timing,
           timing-849ea75d-02-postboot
           (15 cells: 5 blocks x native-off/paper-native/paper-bpf)

The storage and moe campaigns are cell-summary statistical
reanalyses: they recompute arm medians and within-block paired
statistics from the per-cell metrics recorded in each result.json.
They do not re-derive percentiles or throughput from the per-request
records retained in the same files, and they are not new GPU
measurements. For moe, primary throughput is
verified_output_tokens/duration_s over the full eight-request window
including final drain, TTFT is the per-cell median first visible
text (not first model token), and the paired statistic is the
geometric mean of the per-block candidate/reference ratios - not
the median of paired ratios.

Read-only: opens JSON records only. No GPU, no builds, no process
control, no gates. The default action prints the report to stdout.
--output writes the same report to an explicit path, created exclusively
(an existing file aborts the run, nothing is deleted or overwritten).
Paths under docs/paper and inside the raw campaign directories are
rejected so the tool can never replace raw evidence or published results.

All paths resolve relative to the repository root (default: two levels
above this file's scripts/artifact/ location; override with --repo-root).
No home-directory or /tmp dependency.

Definitions (kept explicit in the report):
  median of paired ratios  median over blocks of 100*(candidate/reference - 1)
  ratio of medians         100*(median(candidate blocks)/median(reference blocks) - 1)
These differ in general; both are reported and labeled.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BLOCKS = range(5)
LM_REL = "workloads/lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F"
XS_BUBJNS_REL = "workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns"
XS_BOIYH5_REL = "workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5"
XS_ARMS = ("baseline", "native_port", "bpf_port")
ST_REL = "workloads/lmcache-disk/raw/gds-write-budget-575-20260907-five-block-02"
ST_ARMS = ("fifo", "native10", "bpf10", "native200", "bpf200")
ST_PAIRS = (
    ("bpf200", "bpf10"),
    ("native200", "native10"),
    ("bpf200", "native200"),
    ("bpf10", "native10"),
    ("bpf200", "fifo"),
    ("native200", "fifo"),
    ("bpf10", "fifo"),
    ("native10", "fifo"),
)
ST_KEY_MAP = (
    ("read_scheduled_p99_ms", "read_scheduled_offer_to_completion_p99_ms"),
    ("read_scheduled_p50_ms", "read_scheduled_offer_to_completion_p50_ms"),
    ("write_throughput_mib_s", "write_completion_throughput_mib_s"),
    ("total_bandwidth_mib_s", "total_storage_bandwidth_mib_s"),
)
MOE_REL = "workloads/moe-infinity/raw/paper-v3-575/timing-849ea75d-02-postboot"
MOE_BLOCKS = (1, 2, 3, 4, 5)
MOE_ARMS = ("native-off", "paper-native", "paper-bpf")
MOE_PAIRS = (
    ("paper-bpf", "paper-native"),
    ("paper-native", "native-off"),
    ("paper-bpf", "native-off"),
)


def die(msg: str, code: int = 1) -> None:
    print(f"reanalyze: error: {msg}", file=sys.stderr)
    sys.exit(code)


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        die(f"cannot read {path}: {exc}")


def median(values: List[float]) -> float:
    return float(statistics.median(values))


def num(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def fmt(value: Optional[float], spec: str = ".6f", suffix: str = "") -> str:
    if value is None:
        return "missing"
    return f"{value:{spec}}{suffix}"


def full(value: Optional[float]) -> str:
    """Exact shortest-repr rendering; medians and geometric means only."""
    return "missing" if value is None else repr(float(value))


def geomean(values: List[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------- lm campaign


def load_lm_cells(root: Path) -> Tuple[Dict[Tuple[int, str], dict], List[str]]:
    """Return {(block, arm): record} plus human-readable missing notes."""
    base = root / LM_REL / "cells"
    cells: Dict[Tuple[int, str], dict] = {}
    if not base.is_dir():
        return cells, [f"campaign directory not found: {LM_REL}"]
    for result_path in sorted(base.glob("block-*/position-*/result.json")):
        rec = read_json(result_path)
        if not isinstance(rec, dict):
            continue
        arm = rec.get("arm")
        block = num(rec.get("block"))
        warm = rec.get("warm_phase") if isinstance(rec.get("warm_phase"), dict) else {}
        cells[(int(block), str(arm))] = {
            "path": str(result_path.relative_to(root)),
            "output_tokens_per_s": num(warm.get("output_tokens_per_s")),
            "ttft_median_ms": num(warm.get("warm_ttft_median_ms")),
            "ready": rec.get("ready"),
            "error": rec.get("error"),
        }
    missing = []
    for block in BLOCKS:
        for arm in ("stock", "native", "bpf"):
            if (block, arm) not in cells:
                missing.append(f"block {block} arm {arm}: no result.json")
    return cells, missing


def lm_incomplete(cells: Dict[Tuple[int, str], dict]) -> List[str]:
    notes = []
    for (block, arm), rec in sorted(cells.items()):
        if rec["output_tokens_per_s"] is None:
            notes.append(f"block {block} arm {arm}: output_tokens_per_s missing")
        elif rec["error"] is not None or rec["ready"] is not True:
            notes.append(f"block {block} arm {arm}: error={rec['error']} ready={rec['ready']}")
    return notes


def lm_report(root: Path, cells: dict, missing: List[str], incomplete: List[str]) -> str:
    lines = [f"== LMCache disk physical-reclaim ({LM_REL.split('/')[-1]}) =="]
    lines.append(f"source: {LM_REL}")
    lines.append(f"cells: {len(cells)} present / {len(BLOCKS) * 3} expected")
    if missing:
        lines.append("missing cells:")
        lines.extend(f"  {n}" for n in missing)
    if incomplete:
        lines.append("incomplete cells:")
        lines.extend(f"  {n}" for n in incomplete)
    if not cells:
        lines.append("no usable cells; stopping campaign report.")
        return "\n".join(lines)

    lines.append("per-cell warm generation throughput (token/s) and TTFT median (ms):")
    for block in BLOCKS:
        for arm in ("stock", "native", "bpf"):
            rec = cells.get((block, arm))
            if rec is None:
                continue
            lines.append(
                f"  block {block} {arm:<6} tok_s={fmt(rec['output_tokens_per_s'])} "
                f"ttft_median_ms={fmt(rec['ttft_median_ms'], '.2f')}  ({rec['path']})"
            )

    lines.append("per-arm marginal medians over complete blocks (token/s):")
    med: Dict[str, float] = {}
    for arm in ("stock", "native", "bpf"):
        vals = [
            rec["output_tokens_per_s"]
            for (b, a), rec in cells.items()
            if a == arm and rec["output_tokens_per_s"] is not None
        ]
        med[arm] = median(vals) if vals else None
        lines.append(f"  {arm:<6} {fmt(med[arm])}")

    lines.append(
        "paired statistics, percent (positive = candidate faster than reference):"
    )
    for cand, ref in (("bpf", "native"), ("bpf", "stock"), ("native", "stock")):
        per_block: Dict[int, float] = {}
        for block in BLOCKS:
            c = cells.get((block, cand), {}).get("output_tokens_per_s")
            r = cells.get((block, ref), {}).get("output_tokens_per_s")
            if c is not None and r:
                per_block[block] = 100.0 * (c / r - 1.0)
        lines.append(f"  {cand}_vs_{ref}:")
        if not per_block:
            lines.append("    insufficient complete blocks")
            continue
        lines.append(
            "    per block: "
            + " ".join(f"b{b}={fmt(v, '.4f')}" for b, v in sorted(per_block.items()))
        )
        lines.append(
            f"    median of paired ratios: {fmt(median(list(per_block.values())), '.6f')}%"
        )
        cand_vals = [cells[(b, cand)]["output_tokens_per_s"] for b in per_block]
        ref_vals = [cells[(b, ref)]["output_tokens_per_s"] for b in per_block]
        lines.append(
            f"    ratio of medians:         {fmt(100.0 * (median(cand_vals) / median(ref_vals) - 1.0), '.6f')}%"
        )
        lines.append(
            "    (median of paired ratios is taken over per-block ratios; ratio of"
        )
        lines.append("     medians is computed from the marginal medians above.)")

    summary_path = root / LM_REL / "analysis.json"
    summary = read_json(summary_path)
    if isinstance(summary, dict):
        lines.append("cross-check against existing summary analysis.json:")
        ok = True
        smed = summary.get("medians") if isinstance(summary.get("medians"), dict) else {}
        for arm in ("stock", "native", "bpf"):
            ref_v = num(smed.get(arm))
            mine = med.get(arm)
            if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
                ok = False
                lines.append(f"  medians.{arm}: summary={ref_v} recomputed={mine} (MISMATCH)")
        paired = summary.get("paired") if isinstance(summary.get("paired"), dict) else {}
        for cand, ref in (("bpf", "native"), ("bpf", "stock"), ("native", "stock")):
            key = f"{cand}_vs_{ref}_pct"
            entry = paired.get(key) if isinstance(paired, dict) else None
            ref_v = num(entry.get("median")) if isinstance(entry, dict) else None
            per_block: Dict[int, float] = {}
            for block in BLOCKS:
                c = cells.get((block, cand), {}).get("output_tokens_per_s")
                r = cells.get((block, ref), {}).get("output_tokens_per_s")
                if c is not None and r:
                    per_block[block] = 100.0 * (c / r - 1.0)
            mine = median(list(per_block.values())) if per_block else None
            if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
                ok = False
                lines.append(f"  paired.{key}.median: summary={ref_v} recomputed={mine} (MISMATCH)")
        if ok:
            lines.append("  all medians and paired medians match the existing summary.")
    else:
        lines.append("cross-check: analysis.json not present; skipped.")
    return "\n".join(lines)


# ------------------------------------------------------------- xsched campaign


def xsched_sources(root: Path) -> Dict[Tuple[int, str], Path]:
    """Explicit source map: block 0 native is reused from bOiYh5."""
    src: Dict[Tuple[int, str], Path] = {}
    src[(0, "native_port")] = root / XS_BOIYH5_REL / "cells/block-01-native_port/result.json"
    for arm in ("baseline", "bpf_port"):
        src[(0, arm)] = root / XS_BUBJNS_REL / f"block0-missing/block-01-{arm}/result.json"
    for block in range(1, 5):
        for arm in XS_ARMS:
            src[(block, arm)] = root / XS_BUBJNS_REL / f"remaining/block-0{block}-{arm}/result.json"
    return src


def load_xsched_cells(root: Path) -> Tuple[Dict[Tuple[int, str], dict], List[str]]:
    cells: Dict[Tuple[int, str], dict] = {}
    missing: List[str] = []
    for (block, arm), path in sorted(xsched_sources(root).items()):
        rec = read_json(path)
        if not isinstance(rec, dict):
            missing.append(f"global block {block} arm {arm}: no result.json at {path.relative_to(root)}")
            continue
        lc = rec.get("lc_service") if isinstance(rec.get("lc_service"), dict) else {}
        be = rec.get("be_service") if isinstance(rec.get("be_service"), dict) else {}
        lc_p99_us = num(lc.get("p99_us"))
        cells[(block, arm)] = {
            "path": str(path.relative_to(root)),
            "lc_service_p99_ms": (lc_p99_us / 1000.0) if lc_p99_us is not None else None,
            "be_kernels_per_s": num(rec.get("be_kernels_per_s")),
            "be_service_p99_ms": (num(be.get("p99_us")) / 1000.0) if num(be.get("p99_us")) is not None else None,
            "lc_host_elapsed_ns": num(rec.get("lc_host_elapsed_ns")),
            "be_host_elapsed_ns": num(rec.get("be_host_elapsed_ns")),
            "sample_diagnostics": rec.get("sample_diagnostics") or [],
        }
    return cells, missing


def xsched_report(root: Path, cells: dict, missing: List[str]) -> str:
    lines = [f"== XSched Level-2 device policy pair ({XS_BUBJNS_REL.split('/')[-1]}) =="]
    lines.append(f"sources: block0 native_port from {XS_BOIYH5_REL}")
    lines.append(f"          block0 baseline/bpf_port from {XS_BUBJNS_REL}/block0-missing/")
    lines.append(f"          blocks 1..4 from {XS_BUBJNS_REL}/remaining/")
    lines.append(f"cells: {len(cells)} present / {len(BLOCKS) * len(XS_ARMS)} expected")
    if missing:
        lines.append("missing cells:")
        lines.extend(f"  {n}" for n in missing)
    diag = [(b, a) for (b, a), r in sorted(cells.items()) if r["sample_diagnostics"]]
    if diag:
        lines.append(f"cells with sample_diagnostics: {sorted(diag)}")
    else:
        lines.append("sample_diagnostics: empty in all present cells")
    if not cells:
        lines.append("no usable cells; stopping campaign report.")
        return "\n".join(lines)

    lines.append("per-cell metrics (lc_service p99 in ms, be kernels/s; host elapsed retained in ns):")
    for block in BLOCKS:
        for arm in XS_ARMS:
            rec = cells.get((block, arm))
            if rec is None:
                continue
            lines.append(
                f"  block {block} {arm:<12} lc_p99_ms={fmt(rec['lc_service_p99_ms'])} "
                f"be_kps={fmt(rec['be_kernels_per_s'])} be_p99_ms={fmt(rec['be_service_p99_ms'])} "
                f"lc_host_ns={fmt(rec['lc_host_elapsed_ns'], '.0f')} "
                f"be_host_ns={fmt(rec['be_host_elapsed_ns'], '.0f')}  ({rec['path']})"
            )

    lines.append("per-arm marginal medians over complete blocks:")
    med: Dict[Tuple[str, str], Optional[float]] = {}
    for metric in ("lc_service_p99_ms", "be_kernels_per_s"):
        for arm in XS_ARMS:
            vals = [
                r[metric] for (b, a), r in cells.items()
                if a == arm and isinstance(r.get(metric), float)
            ]
            med[(metric, arm)] = median(vals) if vals else None
            lines.append(f"  {metric:<20} {arm:<12} {fmt(med[(metric, arm)])}")

    lines.append(
        "paired statistics, percent (positive = candidate above reference; for p99, positive = worse tail):"
    )
    for metric in ("lc_service_p99_ms", "be_kernels_per_s"):
        for cand, ref in (("bpf_port", "native_port"), ("bpf_port", "baseline"), ("native_port", "baseline")):
            per_block: Dict[int, float] = {}
            for block in BLOCKS:
                c = cells.get((block, cand), {}).get(metric)
                r = cells.get((block, ref), {}).get(metric)
                if isinstance(c, float) and isinstance(r, float) and r:
                    per_block[block] = 100.0 * (c / r - 1.0)
            lines.append(f"  {metric} {cand}_vs_{ref}:")
            if not per_block:
                lines.append("    insufficient complete blocks")
                continue
            lines.append(
                "    per block: "
                + " ".join(f"b{b}={fmt(v, '.4f')}" for b, v in sorted(per_block.items()))
            )
            lines.append(
                f"    median of paired ratios: {fmt(median(list(per_block.values())), '.6f')}%"
            )
            cand_vals = [cells[(b, cand)][metric] for b in per_block]
            ref_vals = [cells[(b, ref)][metric] for b in per_block]
            lines.append(
                f"    ratio of medians:         {fmt(100.0 * (median(cand_vals) / median(ref_vals) - 1.0), '.6f')}%"
            )

    lines.append("cross-check against existing campaign summary.json medians:")
    for sub, label in (("block0-missing", "block0-missing/summary.json"), ("remaining", "remaining/summary.json")):
        summary = read_json(root / XS_BUBJNS_REL / sub / "summary.json")
        if not isinstance(summary, dict):
            lines.append(f"  {label}: not present; skipped")
            continue
        configs = summary.get("configs") if isinstance(summary.get("configs"), dict) else {}
        for arm in XS_ARMS:
            entry = configs.get(arm) if isinstance(configs, dict) else None
            if not isinstance(entry, dict):
                continue
            ref_us = num(entry.get("lc_service_p99_median_us"))
            # the summary medians cover only the blocks in that subdirectory
            subset = [
                r["lc_service_p99_ms"] for (b, a), r in cells.items()
                if a == arm and isinstance(r.get("lc_service_p99_ms"), float)
                and (sub == "block0-missing" and b == 0 or sub == "remaining" and b >= 1)
            ]
            mine_us = (median(subset) * 1000.0) if subset else None
            tag = "match" if (ref_us is not None and mine_us is not None and abs(ref_us - mine_us) <= 1e-6 * max(1.0, abs(ref_us))) else "MISMATCH"
            lines.append(f"  {label} {arm} lc p99 median_us: summary={ref_us} recomputed={mine_us} ({tag})")
    return "\n".join(lines)


# ------------------------------------------------------------- storage campaign


def load_st_cells(root: Path) -> Tuple[Dict[Tuple[int, str], dict], List[str]]:
    """Return {(block, arm): record} plus human-readable notes.

    Arm is taken from the position-<n>-<arm> directory name created by the
    runner; per-cell metrics come from each cell's result.json. The
    budget-exhausted write count is read from the campaign's existing
    paired-analysis.json row for the same (block, position).
    """
    base = root / ST_REL
    cells: Dict[Tuple[int, str], dict] = {}
    notes: List[str] = []
    if not base.is_dir():
        return cells, [f"campaign directory not found: {ST_REL}"]
    summary = read_json(base / "paired-analysis.json")
    rows: Dict[Tuple[int, int], dict] = {}
    if isinstance(summary, dict):
        for row in summary.get("rows") or []:
            if isinstance(row, dict):
                b, p = num(row.get("block")), num(row.get("position"))
                if b is not None and p is not None:
                    rows[(int(b), int(p))] = row
    for result_path in sorted(base.glob("block-*/position-*/result.json")):
        rec = read_json(result_path)
        if not isinstance(rec, dict):
            notes.append(f"not a JSON object: {result_path.relative_to(root)}")
            continue
        parts = result_path.parent.name.split("-", 2)
        bpart = result_path.parent.parent.name.split("-", 1)
        if len(parts) != 3 or parts[0] != "position" or len(bpart) != 2:
            notes.append(f"unexpected cell directory name: {result_path.relative_to(root)}")
            continue
        try:
            block, position = int(bpart[1]), int(parts[1])
        except ValueError:
            notes.append(f"unexpected cell directory name: {result_path.relative_to(root)}")
            continue
        arm = parts[2]
        if arm not in ST_ARMS:
            notes.append(f"unexpected arm {arm!r}: {result_path.relative_to(root)}")
        metrics = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
        row = rows.get((block, position), {})
        cells[(block, arm)] = {
            "path": str(result_path.relative_to(root)),
            "read_scheduled_p99_ms": num(metrics.get("read_scheduled_offer_to_completion_p99_ms")),
            "read_scheduled_p50_ms": num(metrics.get("read_scheduled_offer_to_completion_p50_ms")),
            "write_throughput_mib_s": num(metrics.get("write_completion_throughput_mib_s")),
            "total_bandwidth_mib_s": num(metrics.get("total_storage_bandwidth_mib_s")),
            "budget_exhausted": num(row.get("write_budget_exhausted")),
            "error": rec.get("error"),
            "cleanup_errors": rec.get("cleanup_errors") or [],
        }
    missing = []
    for block in BLOCKS:
        for arm in ST_ARMS:
            if (block, arm) not in cells:
                missing.append(f"block {block} arm {arm}: no result.json")
    return cells, missing + notes


def st_incomplete(cells: Dict[Tuple[int, str], dict]) -> List[str]:
    notes = []
    for (block, arm), rec in sorted(cells.items()):
        for key in ST_KEY_MAP:
            if rec[key[0]] is None:
                notes.append(f"block {block} arm {arm}: {key[0]} missing")
        if rec["error"] is not None:
            notes.append(f"block {block} arm {arm}: error={rec['error']}")
        if rec["cleanup_errors"]:
            notes.append(f"block {block} arm {arm}: cleanup_errors={rec['cleanup_errors']}")
    return notes


def st_report(root: Path, cells: dict, missing: List[str], incomplete: List[str]) -> str:
    lines = [f"== LMCache GDS write budget ({ST_REL.split('/')[-1]}) =="]
    lines.append(f"source: {ST_REL}")
    lines.append(f"cells: {len(cells)} present / {len(BLOCKS) * len(ST_ARMS)} expected")
    lines.append(
        "note: cell-summary statistical reanalysis. Arm medians and paired"
    )
    lines.append(
        "      ratios are recomputed from the per-cell metrics recorded in each"
    )
    lines.append(
        "      result.json; the per-request records in the same files are not"
    )
    lines.append(
        "      reprocessed. This is not a raw-request reconstruction, not a new"
    )
    lines.append("      GPU measurement, and not evidence for GPUDirect P2P.")
    if missing:
        lines.append("missing cells:")
        lines.extend(f"  {n}" for n in missing)
    if incomplete:
        lines.append("incomplete cells:")
        lines.extend(f"  {n}" for n in incomplete)
    if not cells:
        lines.append("no usable cells; stopping campaign report.")
        return "\n".join(lines)

    lines.append(
        "per-cell scheduled-arrival read p99/p50 (ms), write and total storage"
    )
    lines.append(
        "throughput (MiB/s); budget-exhausted writes from paired-analysis.json:"
    )
    for block in BLOCKS:
        for arm in ST_ARMS:
            rec = cells.get((block, arm))
            if rec is None:
                continue
            lines.append(
                f"  block {block} {arm:<10} p99_ms={fmt(rec['read_scheduled_p99_ms'], '.3f')} "
                f"p50_ms={fmt(rec['read_scheduled_p50_ms'], '.3f')} "
                f"write_mib_s={fmt(rec['write_throughput_mib_s'], '.3f')} "
                f"total_mib_s={fmt(rec['total_bandwidth_mib_s'], '.3f')} "
                f"budget_exhausted={fmt(rec['budget_exhausted'], '.0f')}  ({rec['path']})"
            )

    lines.append("per-arm marginal medians over available per-arm numeric values:")
    med: Dict[Tuple[str, str], Optional[float]] = {}
    for key, _src in ST_KEY_MAP:
        for arm in ST_ARMS:
            vals = [
                r[key] for (b, a), r in cells.items()
                if a == arm and r[key] is not None
            ]
            med[(key, arm)] = median(vals) if vals else None
            lines.append(f"  {key:<28} {arm:<10} {fmt(med[(key, arm)], '.6f')}")

    lines.append(
        "paired statistics, percent (p99: positive = slower tail; "
        "write throughput: positive = higher):"
    )
    for cand, ref in ST_PAIRS:
        for key in ("read_scheduled_p99_ms", "write_throughput_mib_s"):
            per_block: Dict[int, float] = {}
            for block in BLOCKS:
                c = cells.get((block, cand), {}).get(key)
                r = cells.get((block, ref), {}).get(key)
                if c is not None and r:
                    per_block[block] = 100.0 * (c / r - 1.0)
            lines.append(f"  {key} {cand}_vs_{ref}:")
            if not per_block:
                lines.append("    insufficient complete blocks")
                continue
            lines.append(
                "    per block: "
                + " ".join(f"b{b}={fmt(v, '.4f')}" for b, v in sorted(per_block.items()))
            )
            lines.append(
                f"    median of paired ratios: {fmt(median(list(per_block.values())), '.6f')}%"
            )
            cand_vals = [cells[(b, cand)][key] for b in per_block]
            ref_vals = [cells[(b, ref)][key] for b in per_block]
            lines.append(
                f"    ratio of medians:         {fmt(100.0 * (median(cand_vals) / median(ref_vals) - 1.0), '.6f')}%"
            )

    lines.append("cross-check against existing paired-analysis.json:")
    summary = read_json(root / ST_REL / "paired-analysis.json")
    if not isinstance(summary, dict):
        lines.append("  paired-analysis.json not present; skipped.")
        return "\n".join(lines)
    ok = True
    arms = summary.get("arms") if isinstance(summary.get("arms"), dict) else {}
    for arm in ST_ARMS:
        entry = arms.get(arm) if isinstance(arms, dict) else None
        src_med = entry.get("medians") if isinstance(entry, dict) else None
        src_med = src_med if isinstance(src_med, dict) else {}
        for key, src in ST_KEY_MAP:
            ref_v = num(src_med.get(src))
            mine = med.get((key, arm))
            if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
                ok = False
                lines.append(f"  medians.{arm}.{src}: summary={ref_v} recomputed={mine} (MISMATCH)")
    comparisons = summary.get("comparisons") if isinstance(summary.get("comparisons"), dict) else {}
    for cand, ref in ST_PAIRS:
        entry = comparisons.get(f"{cand}/{ref}") if isinstance(comparisons, dict) else None
        for key in ("read_scheduled_p99_ms", "write_throughput_mib_s"):
            _, src = next(item for item in ST_KEY_MAP if item[0] == key)
            src_entry = entry.get(src) if isinstance(entry, dict) else None
            ref_v = num(src_entry.get("median_pct")) if isinstance(src_entry, dict) else None
            per_block: Dict[int, float] = {}
            for block in BLOCKS:
                c = cells.get((block, cand), {}).get(key)
                r = cells.get((block, ref), {}).get(key)
                if c is not None and r:
                    per_block[block] = 100.0 * (c / r - 1.0)
            mine = median(list(per_block.values())) if per_block else None
            if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
                ok = False
                lines.append(
                    f"  comparisons.{cand}/{ref}.{src}.median_pct: "
                    f"summary={ref_v} recomputed={mine} (MISMATCH)"
                )
    if ok:
        lines.append("  all arm medians and paired medians match the existing summary.")
    return "\n".join(lines)


# ------------------------------------------------------------------ moe campaign


def moe_cell_flagged(rec: dict) -> List[str]:
    """Status issues of one cell; empty means clean."""
    flags: List[str] = []
    if rec["passed"] is not True:
        flags.append(f"passed={rec['passed']}")
    if rec["block_passed"] is not True:
        flags.append(f"block_passed={rec['block_passed']}")
    if rec["server_exit_code"] not in (0, None):
        flags.append(f"server_exit_code={rec['server_exit_code']}")
    if rec["cleanup_errors"]:
        flags.append(f"cleanup_errors={rec['cleanup_errors']}")
    return flags


def load_moe_cells(root: Path) -> Tuple[Dict[Tuple[int, str], dict], List[str]]:
    """Return {(block, arm): record} plus human-readable missing notes.

    Source: the top-level block result.json files only
    (block-01..05-attempt-01/result.json), each holding a ``cells``
    list with one entry per mode. The per-mode subdirectory records
    (SSE dumps, telemetry, launch/admission files) are not read.
    """
    base = root / MOE_REL
    cells: Dict[Tuple[int, str], dict] = {}
    missing: List[str] = []
    if not base.is_dir():
        return cells, [f"campaign directory not found: {MOE_REL}"]
    for block in MOE_BLOCKS:
        result_path = base / f"block-{block:02d}-attempt-01/result.json"
        rec = read_json(result_path)
        if not isinstance(rec, dict):
            missing.extend(
                f"block {block} arm {arm}: no result.json" for arm in MOE_ARMS
            )
            continue
        block_passed = rec.get("passed")
        entries: Dict[str, dict] = {}
        for cell in rec.get("cells") or []:
            if isinstance(cell, dict) and isinstance(cell.get("mode"), str):
                entries[cell["mode"]] = cell
        for arm in MOE_ARMS:
            cell = entries.get(arm)
            if not isinstance(cell, dict):
                missing.append(
                    f"block {block} arm {arm}: no cell entry in "
                    f"{result_path.relative_to(root)}"
                )
                continue
            vtok = num(cell.get("verified_output_tokens"))
            dur = num(cell.get("duration_s"))
            cells[(block, arm)] = {
                "path": str(result_path.relative_to(root)),
                "throughput": (vtok / dur) if (vtok is not None and dur) else None,
                "ttft": num(cell.get("first_text_ttft_median_ms")),
                "passed": cell.get("passed"),
                "block_passed": block_passed,
                "server_exit_code": cell.get("server_exit_code"),
                "cleanup_errors": cell.get("cleanup_errors") or [],
            }
    return cells, missing


def moe_incomplete(cells: Dict[Tuple[int, str], dict]) -> List[str]:
    notes: List[str] = []
    for (block, arm), rec in sorted(cells.items()):
        if rec["throughput"] is None:
            notes.append(
                f"block {block} arm {arm}: "
                "throughput (verified_output_tokens/duration_s) unavailable"
            )
        if rec["ttft"] is None:
            notes.append(
                f"block {block} arm {arm}: first_text_ttft_median_ms missing"
            )
        notes.extend(f"block {block} arm {arm}: {f}" for f in moe_cell_flagged(rec))
    return notes


def moe_report(root: Path, cells: dict, missing: List[str], incomplete: List[str]) -> str:
    lines = [f"== MoE paper-v3-575 postboot timing ({MOE_REL.split('/')[-1]}) =="]
    lines.append(f"source: {MOE_REL} (top-level block-01..05-attempt-01/result.json only)")
    lines.append(f"cells: {len(cells)} present / {len(MOE_BLOCKS) * len(MOE_ARMS)} expected")
    lines.append("note: cell-summary statistical reanalysis. throughput =")
    lines.append("      verified_output_tokens/duration_s over the full 8-request")
    lines.append("      window including final drain; TTFT = per-cell median first")
    lines.append("      visible text (not first model token). native-off is the")
    lines.append("      baseline (dispatcher count-cache eviction), paper-native")
    lines.append("      the native arm, paper-bpf the BPF arm (userspace")
    lines.append("      bpftime JIT selectors). Not a fresh SSE/correctness")
    lines.append("      audit, not original-hardware or full-artifact")
    lines.append("      reproduction.")
    if missing:
        lines.append("missing cells:")
        lines.extend(f"  {n}" for n in missing)
    if incomplete:
        lines.append(
            "incomplete or error-status cells (numeric values retained when available):"
        )
        lines.extend(f"  {n}" for n in incomplete)
    if not cells:
        lines.append("no usable cells; stopping campaign report.")
        return "\n".join(lines)

    lines.append(
        "per-cell throughput (verified_output_tokens/duration_s, token/s) and"
    )
    lines.append("TTFT median (ms); * = cell with an error/incomplete status:")
    for block in MOE_BLOCKS:
        for arm in MOE_ARMS:
            rec = cells.get((block, arm))
            if rec is None:
                continue
            star = "*" if moe_cell_flagged(rec) else ""
            lines.append(
                f"  block {block} {arm:<12} throughput={fmt(rec['throughput'])} "
                f"ttft_ms={fmt(rec['ttft'], '.6f')}{star}  ({rec['path']})"
            )

    lines.append("per-arm marginal medians over available per-arm numeric values:")
    med: Dict[Tuple[str, str], Optional[float]] = {}
    for key in ("throughput", "ttft"):
        for arm in MOE_ARMS:
            vals = [
                r[key] for (b, a), r in cells.items()
                if a == arm and r[key] is not None
            ]
            med[(key, arm)] = median(vals) if vals else None
            lines.append(f"  {key:<10} {arm:<12} {full(med[(key, arm)])}")

    lines.append("paired statistics; per-block ratios retained:")
    lines.append("      geometric-mean ratio = exp(mean(log(candidate/reference)))")
    lines.append("      over the available blocks; it is not the median of paired")
    lines.append("      ratios. throughput: >1 favors candidate; ttft: <1 favors")
    lines.append("      candidate. bN* = flagged block (see incomplete cells).")
    geo: Dict[Tuple[str, str, str], Optional[float]] = {}
    for cand, ref in MOE_PAIRS:
        for key in ("throughput", "ttft"):
            per_block: Dict[int, float] = {}
            flagged: Dict[int, List[str]] = {}
            for block in MOE_BLOCKS:
                c = cells.get((block, cand), {}).get(key)
                r = cells.get((block, ref), {}).get(key)
                if c is not None and r:
                    per_block[block] = c / r
                    fl = (
                        moe_cell_flagged(cells[(block, cand)])
                        + moe_cell_flagged(cells[(block, ref)])
                    )
                    if fl:
                        flagged[block] = fl
            lines.append(f"  {key} {cand}/{ref}:")
            if not per_block:
                lines.append("    insufficient complete blocks")
                geo[(key, cand, ref)] = None
                continue
            lines.append(
                "    per block: "
                + " ".join(
                    f"b{b}={fmt(per_block[b], '.6f')}" + ("*" if b in flagged else "")
                    for b in sorted(per_block)
                )
            )
            if flagged:
                lines.append(
                    "    flagged blocks (status issue, numeric ratios retained): "
                    + "; ".join(
                        f"b{b}: {'; '.join(sorted(set(flagged[b])))}"
                        for b in sorted(flagged)
                    )
                )
            geo[(key, cand, ref)] = geomean(
                [per_block[b] for b in sorted(per_block)]
            )
            lines.append(
                f"    geometric-mean ratio: {full(geo[(key, cand, ref)])}"
            )

    lines.append("cross-check against existing audited-analysis-final.json:")
    summary = read_json(root / MOE_REL / "audited-analysis-final.json")
    if not isinstance(summary, dict):
        lines.append("  audited-analysis-final.json not present; skipped.")
        return "\n".join(lines)
    ok = True
    analysis = summary.get("analysis") if isinstance(summary.get("analysis"), dict) else {}
    modes = analysis.get("modes") if isinstance(analysis.get("modes"), dict) else {}
    for arm in MOE_ARMS:
        entry = modes.get(arm) if isinstance(modes, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        for label, metric in (
            ("output_throughput_tokens_per_s", "throughput"),
            ("first_text_ttft_median_ms", "ttft"),
        ):
            ref_v = num(entry.get(label))
            mine = med.get((metric, arm))
            if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
                ok = False
                lines.append(
                    f"  analysis.modes.{arm}.{label}: "
                    f"summary={full(ref_v)} recomputed={full(mine)} (MISMATCH)"
                )
    paired = analysis.get("paired") if isinstance(analysis.get("paired"), dict) else {}
    secondary = summary.get("secondary") if isinstance(summary.get("secondary"), dict) else {}
    sec = secondary.get("first_visible_text_ttft") if isinstance(secondary, dict) else None
    sec_paired = sec.get("paired") if isinstance(sec, dict) else None
    sec_paired = sec_paired if isinstance(sec_paired, dict) else {}
    for cand, ref in MOE_PAIRS:
        key = f"{cand}/{ref}"
        entry = paired.get(key) if isinstance(paired, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        ref_v = num(entry.get("geometric_throughput_ratio"))
        mine = geo.get(("throughput", cand, ref))
        if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
            ok = False
            lines.append(
                f"  analysis.paired.{key}.geometric_throughput_ratio: "
                f"summary={full(ref_v)} recomputed={full(mine)} (MISMATCH)"
            )
        sec_entry = sec_paired.get(key) if isinstance(sec_paired, dict) else None
        sec_entry = sec_entry if isinstance(sec_entry, dict) else {}
        ref_v = num(sec_entry.get("geometric_ttft_ratio"))
        mine = geo.get(("ttft", cand, ref))
        if ref_v is None or mine is None or abs(ref_v - mine) > 1e-9 * abs(ref_v):
            ok = False
            lines.append(
                f"  secondary.first_visible_text_ttft.paired.{key}.geometric_ttft_ratio: "
                f"summary={full(ref_v)} recomputed={full(mine)} (MISMATCH)"
            )
    if ok:
        lines.append("  all arm medians and geometric-mean ratios match the")
        lines.append("  existing audited analysis.")
    lines.append("  note: paired_block_bootstrap_ci95 is retained in the audited")
    lines.append("        analysis; this tool does not recompute or refresh any CI.")
    return "\n".join(lines)


# --------------------------------------------------------------------- output


def guarded_output(path: Path, root: Path) -> Path:
    target = path.expanduser().resolve()
    paper = (root / "docs" / "paper").resolve()
    if target == paper or paper in target.parents:
        die("--output must be outside docs/paper")
    for rel in (LM_REL, XS_BUBJNS_REL, XS_BOIYH5_REL, ST_REL, MOE_REL):
        raw = (root / rel).resolve()
        if target == raw or raw in target.parents:
            die(f"--output must not write into raw campaign directory {rel}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only reanalysis of completed raw evidence (read-only)."
    )
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="repository root (default: derived from this file's location)")
    parser.add_argument("--campaign",
                        choices=("all", "lm", "xsched", "storage", "moe"),
                        default="all",
                        help="campaign to reanalyze; default runs all four")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the report to this path instead of stdout "
                             "(created exclusively, never overwritten; rejected "
                             "under docs/paper or inside raw campaign dirs)")
    args = parser.parse_args()

    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if not (root / "workloads").is_dir():
        die(f"no workloads/ under {root}; pass --repo-root")

    sections: List[str] = []
    if args.campaign in ("all", "lm"):
        cells, missing = load_lm_cells(root)
        incomplete = lm_incomplete(cells)
        sections.append(lm_report(root, cells, missing, incomplete))
    if args.campaign in ("all", "xsched"):
        cells, missing = load_xsched_cells(root)
        sections.append(xsched_report(root, cells, missing))
    if args.campaign in ("all", "storage"):
        cells, missing = load_st_cells(root)
        incomplete = st_incomplete(cells)
        sections.append(st_report(root, cells, missing, incomplete))
    if args.campaign in ("all", "moe"):
        cells, missing = load_moe_cells(root)
        incomplete = moe_incomplete(cells)
        sections.append(moe_report(root, cells, missing, incomplete))
    if not sections:
        die("no campaign data found; nothing to report")

    report = "Reanalysis (CPU-only, read-only; existing completed evidence only)\n" \
             f"repo root: {root}\n\n" + "\n\n".join(sections) + "\n"
    if args.output:
        target = guarded_output(args.output, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x", encoding="utf-8") as fh:
                fh.write(report)
        except FileExistsError:
            die(f"output already exists: {target} (remove it first; "
                "reanalyze never overwrites)")
        print(f"reanalyze: report written to {target}")
    else:
        print(report)


if __name__ == "__main__":
    main()

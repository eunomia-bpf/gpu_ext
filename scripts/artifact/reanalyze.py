#!/usr/bin/env python3
"""CPU-only reanalysis entrypoint for completed raw evidence.

Recomputes cell metrics and paired statistics from existing raw
result.json records. Two supplemental campaigns are supported:

  lm      LMCache disk physical-reclaim serving, pXYN4F
  xsched  XSched Level-2 device policy pair, buBjns (+ reused bOiYh5
          native block 0)

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
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BLOCKS = range(5)
LM_REL = "workloads/lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F"
XS_BUBJNS_REL = "workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns"
XS_BOIYH5_REL = "workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5"
XS_ARMS = ("baseline", "native_port", "bpf_port")


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


# --------------------------------------------------------------------- output


def guarded_output(path: Path, root: Path) -> Path:
    target = path.expanduser().resolve()
    paper = (root / "docs" / "paper").resolve()
    if target == paper or paper in target.parents:
        die("--output must be outside docs/paper")
    for rel in (LM_REL, XS_BUBJNS_REL, XS_BOIYH5_REL):
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
    parser.add_argument("--campaign", choices=("all", "lm", "xsched"), default="all")
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

#!/usr/bin/env python3
"""Fig.13 fast results analyzer (descriptive only; no gates).

Input: one completed run directory produced by run_fig13_fast.py, containing
fig13_fast.csv plus per-arm meta.json and tool logs.

Output: fig13_fast_analysis.json and fig13_fast_analysis.md written into
--output-dir. All CSV rows are preserved verbatim; failures, missing values,
and zero/missing engagement counters stay visible. A missing number is never
substituted with zero and never turned into a speedup; counter zero/missing
is reported as an interpretation limitation, never a performance gate.

Per arm (baseline/memory_only/sched_only/combined) the report summarizes the
high/low independent wall latency (seconds) and the high/low internal median
kernel time (ms): all raw values, median, min/max.

Paired deltas are computed within each block between the two named arms:
pct = 100*(num/den - 1) per block, then median/range over blocks. Ratios of
medians are never reported as paired values.

Performance medians and paired deltas include only row-role records whose
tenant rc is 0 (successful completion); records with rc nonzero or missing
keep their raw values and are listed as excluded with explicit reasons.
rc/timeout/counters are classification metadata and never gate the report.

Counter provenance (exact distinction, metadata only; never a gate):
- mem: run_fig13_fast.py recorded CSV meta via re.search, which captures the
  FIRST matching line; since prefetch_eviction_pid prints per-PID entries and
  a Summary block every 5s, that value is an early per-PID first entry, not
  the final aggregate. It is retained here labeled "legacy first entry".
  Independently, this analyzer parses the LAST "=== Summary ===" block of
  mem_tool.log (printed at detach) for final aggregate activated/used/allow/
  deny counters.
- sched: gpu_sched_set_timeslices prints its statistics once at exit, so the
  legacy CSV metadata equals the final values; the last "=== Statistics ==="
  block is also parsed independently for the full counter set.

No retries, no pass/fail gates, no invented metrics, no composite fairness
scores, no confidence intervals, no plotting; standard library only.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("baseline", "memory_only", "sched_only", "combined")
ARM_TOOLS = {
    "baseline": (),
    "memory_only": ("mem",),
    "sched_only": ("sched",),
    "combined": ("sched", "mem"),
}
METRICS = (
    ("high_latency_s", "high wall latency (s)"),
    ("low_latency_s", "low wall latency (s)"),
    ("high_median_ms", "high median kernel time (ms)"),
    ("low_median_ms", "low median kernel time (ms)"),
)
PAIRS = (
    ("memory_only", "baseline"),
    ("sched_only", "baseline"),
    ("combined", "baseline"),
    ("combined", "memory_only"),
    ("combined", "sched_only"),
)
MEM_SUMMARY_FIELDS = (
    ("total_current_chunks", r"Total current chunks:\s*(\d+)"),
    ("total_activated", r"Total activated:\s*(\d+)"),
    ("total_used_calls", r"Total used calls:\s*(\d+)"),
    ("policy_allow_moved", r"Policy allow \(moved\):\s*(\d+)"),
    ("policy_deny_not_moved", r"Policy deny \(not moved\):\s*(\d+)"),
)
SCHED_STAT_FIELDS = (
    ("task_init", r"task_init:\s*(\d+)"),
    ("bind", r"\bbind:\s*(\d+)"),
    ("task_destroy", r"task_destroy:\s*(\d+)"),
    ("timeslice_mod", r"timeslice_mod:\s*(\d+)"),
    ("policy_hit", r"policy_hit:\s*(\d+)"),
    ("policy_miss", r"policy_miss:\s*(\d+)"),
    ("interleave_mod", r"interleave_mod:\s*(\d+)"),
    ("interleave_observed", r"interleave_observed:\s*(\d+)"),
    ("interleave_mismatch", r"interleave_mismatch:\s*(\d+)"),
    ("setter_error", r"setter_error:\s*(\d+)"),
    ("control_override", r"control_override:\s*(\d+)"),
)
PCT_DEF = ("pct = 100*(num/den - 1), computed within each block between the "
           "two arms' rows of that block; positive means num took longer "
           "than den on that metric")
COUNTER_NOTE = ("counter zero/missing reflects what the policy tools recorded "
                "(or failed to record); it is an interpretation limitation "
                "only, never a performance gate")
RC_NOTE = ("performance medians and paired deltas include only row-role "
           "records whose tenant rc is 0 (successful completion); records "
           "with rc nonzero or missing keep their raw values and are listed "
           "as excluded with explicit reasons. rc/timeout/counter fields are "
           "classification metadata and never gate the report.")
PROVENANCE_NOTE = ("mem: values in CSV meta columns were recorded by "
                   "run_fig13_fast.py parse_mem_meta via re.search, which "
                   "captures the FIRST matching line (per-PID entry in the "
                   "first periodic print of prefetch_eviction_pid), not the "
                   "final aggregate; they are labeled 'legacy first entry'. "
                   "'Final summary' values are parsed independently from the "
                   "LAST '=== Summary ===' block of mem_tool.log (printed at "
                   "detach) for final aggregate activated/used/allow/deny. "
                   "sched: the scheduler prints its statistics once at exit, "
                   "so legacy CSV metadata equals final values; the last "
                   "'=== Statistics ===' block is parsed independently for "
                   "the full final counter set. These provenance classes are "
                   "kept separate and are never merged.")


def load_json(path):
    try:
        return json.loads(path.read_text()), "ok"
    except FileNotFoundError:
        return None, "absent"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_log(path):
    try:
        return path.read_text(errors="replace"), "ok"
    except FileNotFoundError:
        return None, "absent"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def to_float(v):
    s = str(v).strip() if v is not None else ""
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def block_int(r):
    return to_int(r.get("block"))


def block_sort_key(b):
    if b is None:
        return (2, 0, "")
    if isinstance(b, int):
        return (0, b, "")
    return (1, 0, str(b))


def find_block_dirs(run_dir):
    dirs = {}
    if not run_dir.is_dir():
        return dirs
    for child in sorted(run_dir.iterdir()):
        m = re.fullmatch(r"block(\d+)_(.+)", child.name)
        if child.is_dir() and m:
            dirs[(int(m.group(1)), m.group(2))] = child
    return dirs


def split_notes(s):
    if s is None:
        return []
    if isinstance(s, list):
        return [str(t).strip() for t in s if str(t).strip()]
    return [t.strip() for t in str(s).split(";") if t.strip()]


def summarize(values):
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def parse_last_mem_summary(text):
    idx = text.rfind("=== Summary ===")
    if idx < 0:
        return None
    out = {"summary_block_number": text.count("=== Summary ===")}
    for name, pat in MEM_SUMMARY_FIELDS:
        m = re.search(pat, text[idx:])
        out[name] = int(m.group(1)) if m else None
    return out


def parse_last_sched_stats(text):
    idx = text.rfind("=== Statistics ===")
    if idx < 0:
        return None
    out = {}
    for name, pat in SCHED_STAT_FIELDS:
        m = re.search(pat, text[idx:])
        out[name] = int(m.group(1)) if m else None
    return out


def policy_log_section(arm_dir, arm):
    tools = ARM_TOOLS.get(arm, ())
    section = {}
    for tool in ("mem", "sched"):
        used = tool in tools
        if not used:
            section[tool] = {"tool_used_by_arm": False,
                             "log_status": "tool_not_used_by_this_arm"}
            continue
        if arm_dir is None:
            section[tool] = {"tool_used_by_arm": True,
                             "log_status": "arm_dir_absent"}
            continue
        entry = {"tool_used_by_arm": True}
        text, status = read_log(arm_dir / f"{tool}_tool.log")
        entry["log_status"] = status
        if text is not None:
            if tool == "mem":
                entry["final_summary_last_block"] = parse_last_mem_summary(text)
                entry["summary_blocks_seen"] = text.count("=== Summary ===")
                entry["detach_marker_seen"] = "Detaching struct_ops" in text
            else:
                entry["final_statistics"] = parse_last_sched_stats(text)
        section[tool] = entry
    return section


def arm_section(arm_rows):
    metrics = {}
    for col, _label in METRICS:
        role = col.split("_", 1)[0]
        raw = []
        for r in arm_rows:
            rc = to_int(r.get(f"{role}_rc"))
            value = to_float(r.get(col))
            if value is None:
                included, reason = False, "value_missing"
            elif rc is None:
                included, reason = False, f"{role}_rc_missing"
            elif rc != 0:
                included, reason = False, f"{role}_rc={rc}"
            else:
                included, reason = True, None
            raw.append({
                "csv_row": r["csv_row"],
                "block": r.get("block"),
                "role_rc": rc,
                "value": value,
                "in_performance": included,
                "exclusion_reason": reason,
            })
        numeric = [it["value"] for it in raw if it["value"] is not None]
        perf_vals = [it["value"] for it in raw if it["in_performance"]]
        excluded = [
            {"csv_row": it["csv_row"], "block": it["block"],
             "role_rc": it["role_rc"], "value": it["value"],
             "reason": it["exclusion_reason"]}
            for it in raw
            if it["value"] is not None and not it["in_performance"]
        ]
        metrics[col] = {
            "role": role,
            "raw": raw,
            "n_rows": len(raw),
            "n_value_numeric": len(numeric),
            "n_value_missing": len(raw) - len(numeric),
            "n_excluded_numeric": len(excluded),
            "excluded_numeric": excluded,
            "performance": summarize(perf_vals),
        }
    rc_nonzero = {"high": 0, "low": 0}
    for r in arm_rows:
        for role in ("high", "low"):
            rc = to_int(r.get(f"{role}_rc"))
            if rc is not None and rc != 0:
                rc_nonzero[role] += 1
    return {"metrics": metrics, "n_rows_nonzero_rc": rc_nonzero}


def paired_section(rows):
    by = {}
    for r in rows:
        b = block_int(r)
        if b is None:
            b = str(r.get("block"))
        by.setdefault((b, r.get("arm") or ""), []).append(r)
    blocks = sorted({k[0] for k in by}, key=block_sort_key)
    pairs_out = {}
    unpaired = []
    for num_arm, den_arm in PAIRS:
        per_metric = {}
        pkey = f"{num_arm}/{den_arm}"
        for col, _label in METRICS:
            pairs, skip = [], []
            for b in blocks:
                nr = by.get((b, num_arm), [])
                dr = by.get((b, den_arm), [])
                if len(nr) != 1:
                    skip.append({"block": b, "reason": f"numerator_rows={len(nr)}"})
                    continue
                if len(dr) != 1:
                    skip.append({"block": b, "reason": f"denominator_rows={len(dr)}"})
                    continue
                role = col.split("_", 1)[0]
                num_rc = to_int(nr[0].get(f"{role}_rc"))
                den_rc = to_int(dr[0].get(f"{role}_rc"))
                if num_rc is None:
                    skip.append({"block": b,
                                 "reason": f"numerator_{role}_rc_missing"})
                    continue
                if num_rc != 0:
                    skip.append({"block": b,
                                 "reason": f"numerator_{role}_rc={num_rc}"})
                    continue
                if den_rc is None:
                    skip.append({"block": b,
                                 "reason": f"denominator_{role}_rc_missing"})
                    continue
                if den_rc != 0:
                    skip.append({"block": b,
                                 "reason": f"denominator_{role}_rc={den_rc}"})
                    continue
                num = to_float(nr[0].get(col))
                den = to_float(dr[0].get(col))
                if den is None:
                    skip.append({"block": b, "reason": "denominator_missing"})
                    continue
                if num is None:
                    skip.append({"block": b, "reason": "numerator_missing"})
                    continue
                if den == 0:
                    skip.append({"block": b, "reason": "denominator_zero"})
                    continue
                pairs.append({
                    "block": b,
                    "num": num,
                    "den": den,
                    "pct": 100.0 * (num / den - 1.0),
                })
            stats = summarize([p["pct"] for p in pairs])
            per_metric[col] = {
                "pairs": pairs,
                "unpaired": skip,
                "n_paired": stats["n"],
                **stats,
            }
            if skip:
                unpaired.append({"pair": pkey, "metric": col, "blocks": skip})
        pairs_out[pkey] = {"metrics": per_metric}
    return {"definition": PCT_DEF, "pairs": pairs_out}, unpaired


def failure_section(rows, block_dirs):
    per_row = []
    arm_counts = {}
    csv_notes = Counter()
    meta_notes = Counter()
    for r in rows:
        b = block_int(r)
        arm = r.get("arm") or ""
        key = (str(r.get("block")), arm) if b is None else (b, arm)
        arm_dir = block_dirs.get(key)
        meta, meta_status = None, "dir_absent"
        if arm_dir is not None:
            meta, meta_status = load_json(arm_dir / "meta.json")
        csv_note_tokens = split_notes(r.get("notes"))
        csv_notes.update(csv_note_tokens)
        meta_note_tokens = []
        if isinstance(meta, dict):
            meta_note_tokens = split_notes(meta.get("notes"))
            meta_notes.update(meta_note_tokens)
        logs = policy_log_section(arm_dir, arm)
        logs["mem"]["legacy_first_entry"] = r.get("mem_meta")
        logs["sched"]["legacy_csv_metadata"] = r.get("sched_meta")
        high_rc = to_int(r.get("high_rc"))
        low_rc = to_int(r.get("low_rc"))
        high_to = to_int(r.get("high_timeout"))
        low_to = to_int(r.get("low_timeout"))
        per_row.append({
            "csv_row": r["csv_row"],
            "block": r.get("block"),
            "arm": arm,
            "high_rc": high_rc,
            "low_rc": low_rc,
            "high_timeout_csv": high_to,
            "low_timeout_csv": low_to,
            "notes": r.get("notes", ""),
            "meta_json_file": arm_dir.name if arm_dir else None,
            "meta_json_status": meta_status,
            "meta_notes": meta_note_tokens,
            "meta_timeouts": meta.get("timeouts") if isinstance(meta, dict) else None,
            "rc_sched_tool": meta.get("rc_sched_tool") if isinstance(meta, dict) else None,
            "rc_mem_tool": meta.get("rc_mem_tool") if isinstance(meta, dict) else None,
            "meta_json_engagement": (
                {t: (meta.get("engagement_metadata") or {}).get(t)
                 for t in ("sched", "mem")}
                if isinstance(meta, dict) else None
            ),
            "policy_logs": logs,
        })
        c = arm_counts.setdefault(arm, {
            "rows": 0,
            "high_rc_nonzero": 0, "high_rc_missing": 0,
            "low_rc_nonzero": 0, "low_rc_missing": 0,
            "high_timeout": 0, "low_timeout": 0,
            "meta_json_absent_or_unreadable": 0,
        })
        c["rows"] += 1
        if high_rc is None:
            c["high_rc_missing"] += 1
        elif high_rc != 0:
            c["high_rc_nonzero"] += 1
        if low_rc is None:
            c["low_rc_missing"] += 1
        elif low_rc != 0:
            c["low_rc_nonzero"] += 1
        if high_to == 1:
            c["high_timeout"] += 1
        if low_to == 1:
            c["low_timeout"] += 1
        if meta_status != "ok":
            c["meta_json_absent_or_unreadable"] += 1
    return {
        "per_row": per_row,
        "arm_counts": arm_counts,
        "note_counts": {"csv": dict(csv_notes), "meta_json": dict(meta_notes)},
        "policy_log_aggregates": aggregate_policy_logs(per_row),
        "counter_provenance_note": PROVENANCE_NOTE,
        "counter_note": COUNTER_NOTE,
    }


def aggregate_policy_logs(per_row):
    agg = {}
    spec = (
        ("mem", "final_summary_last_block",
         ("total_activated", "total_used_calls",
          "policy_allow_moved", "policy_deny_not_moved")),
        ("sched", "final_statistics",
         ("policy_hit", "policy_miss", "timeslice_mod")),
    )
    for tool, block_key, fields in spec:
        rows_for_tool = 0
        rows_log_absent = 0
        rows_log_ok_no_final_block = 0
        rows_with_final_block = 0
        cf = {k: {"present": 0, "zero": 0, "absent_in_final_block": 0}
              for k in fields}
        for pr in per_row:
            entry = (pr.get("policy_logs") or {}).get(tool)
            if not isinstance(entry, dict) or not entry.get("tool_used_by_arm"):
                continue
            rows_for_tool += 1
            if entry.get("log_status") != "ok":
                rows_log_absent += 1
                continue
            block = entry.get(block_key)
            if block is None:
                rows_log_ok_no_final_block += 1
                continue
            rows_with_final_block += 1
            for k in fields:
                v = block.get(k)
                if v is None:
                    cf[k]["absent_in_final_block"] += 1
                else:
                    cf[k]["present"] += 1
                    if v == 0:
                        cf[k]["zero"] += 1
        agg[tool] = {
            "rows_for_tool_arms": rows_for_tool,
            "rows_log_absent": rows_log_absent,
            "rows_log_ok_no_final_block": rows_log_ok_no_final_block,
            "rows_with_final_block": rows_with_final_block,
            "counters": cf,
        }
    return agg


def build_limitations(rows, per_arm, unpaired):
    lims = []
    for name, sect in per_arm.items():
        for col, ms in sect["metrics"].items():
            if ms["n_value_missing"]:
                lims.append(
                    f"{name}: {ms['n_value_missing']}/{ms['n_rows']} rows lack "
                    f"{col}; kept missing, not zeroed")
            if ms["n_excluded_numeric"]:
                reasons = Counter(e["reason"] for e in ms["excluded_numeric"])
                detail = ", ".join(
                    f"{k} x{n}" for k, n in sorted(reasons.items()))
                lims.append(
                    f"{name}: {ms['n_excluded_numeric']} numeric {col} "
                    f"values excluded from performance stats ({detail})")
        nz = sect["n_rows_nonzero_rc"]
        if nz["high"] or nz["low"]:
            lims.append(
                f"{name}: {nz['high']} high / {nz['low']} low tenants exited "
                "nonzero; their rows are preserved, not treated as "
                "measurements of successful runs")
    for u in unpaired:
        blocks = "; ".join(
            f"block {s['block']}: {s['reason']}" for s in u["blocks"])
        lims.append(f"pair {u['pair']} {u['metric']} not paired ({blocks})")
    counts = Counter((str(r.get("block")), r.get("arm") or "") for r in rows)
    dup = sorted((k, n) for k, n in counts.items() if n > 1)
    if dup:
        lims.append(
            "duplicate (block, arm) rows; pairing skipped for them: "
            + "; ".join(f"block={b} arm={a} x{n}" for (b, a), n in dup))
    extra_arms = sorted({r.get("arm") or "" for r in rows} - set(ARMS))
    if extra_arms:
        lims.append(f"non-canonical arm values present: {extra_arms}")
    lims.append(RC_NOTE)
    lims.append(COUNTER_NOTE)
    lims.append(PROVENANCE_NOTE)
    return lims


def fnum(v, col):
    if v is None:
        return "n/a"
    if col.endswith("_s"):
        return f"{v:.6f}"
    return f"{v:g}"


def fpct(v):
    return "n/a" if v is None else f"{v:+.2f}%"


def fmt_mem_final(fs):
    if fs is None:
        return "no final summary"
    return ("act={} used={} allow={} deny={}".format(
        *(fs.get(k) if fs.get(k) is not None else "na" for k in (
            "total_activated", "total_used_calls",
            "policy_allow_moved", "policy_deny_not_moved"))))


def fmt_sched_final(st):
    if st is None:
        return "no final statistics"
    return "hit={} miss={} mod={}".format(
        *(st.get(k) if st.get(k) is not None else "na" for k in (
            "policy_hit", "policy_miss", "timeslice_mod")))


def md_escape(s):
    return str(s).replace("|", "\\|") if s is not None else "n/a"


def build_markdown(run_dir, source, run_json, run_json_status, per_arm,
                   paired, failures, limitations):
    L = []
    L.append(f"# fig13-fast analysis: {run_dir.name}")
    L.append("")
    L.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    L.append("")
    L.append(f"Source: `{run_dir}` ({source['csv_row_count']} CSV data rows, "
             "all preserved)")
    L.append("")
    L.append("Descriptive only: no pass/fail gates, no retries, no raw-row filtering, "
             "no invented metrics, no composite scores, no confidence "
             "intervals. Missing numbers are never replaced by zero and never "
             "turned into speedups.")
    L.append("")
    L.append("## Run overview")
    L.append("")
    if run_json is not None:
        for key in ("mode", "started_utc", "timeout_s", "kernel",
                    "size_factor", "iterations", "mem_policy", "sched_policy",
                    "engagement"):
            if key in run_json:
                L.append(f"- {key}: `{run_json[key]}`")
        L.append(f"- mem_tool: `{run_json.get('mem_tool', 'n/a')}`")
        L.append(f"- sched_tool: `{run_json.get('sched_tool', 'n/a')}`")
    else:
        L.append(f"- run.json {run_json_status}; overview limited to CSV")
    L.append("")
    L.append(f"- blocks present: {md_escape(source['blocks_present'])}")
    L.append(f"- arms present: {md_escape(source['arms_present'])}")
    L.append("")
    L.append("Wall latency is each tenant's own SIGCONT-to-exit time in "
             "seconds; median kernel time is the uvmbench-reported median "
             "time in ms parsed from the tenant log.")
    L.append("")
    L.append("## Per-arm metric summary")
    L.append("")
    L.append("| arm | metric | rows | rc-ok numeric | rc-excluded numeric | "
             "value missing | median | min | max |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in source["arms_present"]:
        sect = per_arm[name]
        for col, label in METRICS:
            ms = sect["metrics"][col]
            perf = ms["performance"]
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, label, ms["n_rows"], perf["n"],
                ms["n_excluded_numeric"], ms["n_value_missing"],
                fnum(perf["median"], col), fnum(perf["min"], col),
                fnum(perf["max"], col)))
    L.append("")
    L.append(RC_NOTE)
    L.append("")
    L.append("Values missing from a CSV cell are counted as missing and are "
             "never zeroed; rc-excluded numeric values keep their raw numbers "
             "in the JSON raw lists together with the explicit exclusion "
             "reasons.")
    L.append("")
    L.append("## Block-paired deltas")
    L.append("")
    L.append(f"Definition: {PCT_DEF}")
    L.append("")
    L.append("| pair | metric | blocks paired | median pct | min pct | "
             "max pct | unpaired blocks |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    for pkey, psect in paired["pairs"].items():
        for col, label in METRICS:
            ms = psect["metrics"][col]
            unpaired_txt = "; ".join(
                f"block {s['block']}: {s['reason']}" for s in ms["unpaired"])
            L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                pkey, label, ms["n_paired"], fpct(ms["median"]),
                fpct(ms["min"]), fpct(ms["max"]),
                md_escape(unpaired_txt) if unpaired_txt else "none"))
    L.append("")
    L.append("### Paired delta raw values")
    L.append("")
    L.append("| pair | metric | block | num | den | pct |")
    L.append("|---|---|---:|---:|---:|---:|")
    for pkey, psect in paired["pairs"].items():
        for col, label in METRICS:
            for p in psect["metrics"][col]["pairs"]:
                L.append("| {} | {} | {} | {} | {} | {} |".format(
                    pkey, label, p["block"],
                    fnum(p["num"], col), fnum(p["den"], col),
                    fpct(p["pct"])))
    L.append("")
    L.append("## Per-row status and failures")
    L.append("")
    L.append("| row | block | arm | high_rc | low_rc | high_timeout | "
             "low_timeout | rc_sched_tool | rc_mem_tool | notes | "
             "meta_notes |")
    L.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    for pr in failures["per_row"]:
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |"
                 .format(
                     pr["csv_row"], pr["block"], pr["arm"],
                     "n/a" if pr["high_rc"] is None else pr["high_rc"],
                     "n/a" if pr["low_rc"] is None else pr["low_rc"],
                     "n/a" if pr["high_timeout_csv"] is None
                     else pr["high_timeout_csv"],
                     "n/a" if pr["low_timeout_csv"] is None
                     else pr["low_timeout_csv"],
                     "n/a" if pr["rc_sched_tool"] is None
                     else pr["rc_sched_tool"],
                     "n/a" if pr["rc_mem_tool"] is None
                     else pr["rc_mem_tool"],
                     md_escape(pr["notes"]) if pr["notes"] else "-",
                     md_escape(";".join(pr["meta_notes"]))
                     if pr["meta_notes"] else "-"))
    L.append("")
    tool_rc_note = ("rc_sched_tool / rc_mem_tool come from meta.json "
                    "(n/a when the arm does not use that tool or meta.json "
                    "is absent); non-integer or missing values stay as "
                    "reported by the harness.")
    L.append(tool_rc_note)
    L.append("")
    L.append("## Policy tool counters (metadata only)")
    L.append("")
    L.append(PROVENANCE_NOTE)
    L.append("")
    agg = failures["policy_log_aggregates"]
    L.append("| tool | rows for tool arms | log absent | log ok but no "
             "final block | rows with final block |")
    L.append("|---|---:|---:|---:|---:|")
    for tool in ("mem", "sched"):
        a = agg[tool]
        L.append("| {} | {} | {} | {} | {} |".format(
            tool, a["rows_for_tool_arms"], a["rows_log_absent"],
            a["rows_log_ok_no_final_block"], a["rows_with_final_block"]))
    L.append("")
    L.append("| tool | counter | present | zero | absent in final block |")
    L.append("|---|---|---:|---:|---:|")
    for tool in ("mem", "sched"):
        for k, v in agg[tool]["counters"].items():
            L.append("| {} | {} | {} | {} | {} |".format(
                tool, k, v["present"], v["zero"], v["absent_in_final_block"]))
    L.append("")
    L.append(COUNTER_NOTE)
    L.append("")
    L.append("### Per-row counter provenance")
    L.append("")
    L.append("| row | block | arm | mem final summary (LAST block) | "
             "mem legacy first entry (CSV) | sched final statistics | "
             "sched legacy (CSV) |")
    L.append("|---:|---:|---|---|---|---|---|")
    for pr in failures["per_row"]:
        pl = pr["policy_logs"]
        mem, sched = pl["mem"], pl["sched"]
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            pr["csv_row"], pr["block"], pr["arm"],
            (md_escape(fmt_mem_final(mem.get("final_summary_last_block")))
             if mem.get("log_status") == "ok"
             else md_escape(f"log {mem.get('log_status')}")),
            md_escape(mem.get("legacy_first_entry")),
            (md_escape(fmt_sched_final(sched.get("final_statistics")))
             if sched.get("log_status") == "ok"
             else md_escape(f"log {sched.get('log_status')}")),
            md_escape(sched.get("legacy_csv_metadata"))))
    L.append("")
    L.append("## Notes token counts")
    L.append("")
    L.append("- CSV notes column: " + md_escape(failures["note_counts"]["csv"]))
    L.append("- meta.json notes: " + md_escape(failures["note_counts"]["meta_json"]))
    L.append("")
    L.append("## Summary counts per arm")
    L.append("")
    L.append("| arm | rows | high_rc != 0 | low_rc != 0 | high_rc missing | "
             "low_rc missing | high_timeouts | low_timeouts | meta.json "
             "absent/unreadable |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, c in failures["arm_counts"].items():
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm, c["rows"], c["high_rc_nonzero"], c["low_rc_nonzero"],
            c["high_rc_missing"], c["low_rc_missing"], c["high_timeout"],
            c["low_timeout"], c["meta_json_absent_or_unreadable"]))
    L.append("")
    L.append("Timeouts are CSV high_timeout/low_timeout columns (1 = timed "
             "out); meta.json timeout booleans are in the JSON report.")
    L.append("")
    L.append("## Interpretation limitations")
    L.append("")
    for lim in limitations:
        L.append(f"- {lim}")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Analyze one fig13-fast run directory (descriptive; no gates)")
    ap.add_argument("run_dir", help="completed fig13-fast run directory "
                    "(contains fig13_fast.csv, blockNN_<arm>/ dirs)")
    ap.add_argument("--output-dir", required=True,
                    help="directory to write fig13_fast_analysis.json/.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    csv_path = run_dir / "fig13_fast.csv"
    if not csv_path.is_file():
        print(f"error: {csv_path} not found", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = []
        for i, r in enumerate(reader, start=2):
            item = {k: v for k, v in r.items() if k is not None}
            extra = r.get(None)
            if extra:
                item["_extra_fields"] = extra
            item["csv_row"] = i
            rows.append(item)

    run_json, run_json_status = load_json(run_dir / "run.json")
    block_dirs = find_block_dirs(run_dir)

    per_arm = {}
    for arm in ARMS:
        per_arm[arm] = arm_section([r for r in rows
                                    if (r.get("arm") or "") == arm])
    extra_arms = sorted({(r.get("arm") or "") for r in rows} - set(ARMS))
    for arm in extra_arms:
        per_arm[arm] = arm_section([r for r in rows
                                    if (r.get("arm") or "") == arm])
    arms_present = [a for a in list(ARMS) + extra_arms
                    if per_arm[a]["metrics"][METRICS[0][0]]["n_rows"] > 0]

    paired, unpaired = paired_section(rows)
    failures = failure_section(rows, block_dirs)
    limitations = build_limitations(rows, per_arm, unpaired)

    def first_prev(name):
        for col, _ in METRICS:
            for a in arms_present:
                if per_arm[a]["metrics"][col]["raw"]:
                    return per_arm[a]["metrics"][col]["raw"][0]["block"]
        return None

    source = {
        "run_dir": str(run_dir),
        "csv": "fig13_fast.csv",
        "csv_columns": columns,
        "csv_row_count": len(rows),
        "blocks_present": sorted(
            {str(r.get("block")) for r in rows}, key=block_sort_key),
        "arms_present": arms_present,
        "run_json_status": run_json_status,
        "arm_dirs_found": len(block_dirs),
    }

    report = {
        "report": "fig13_fast analyze_results",
        "descriptive_only": True,
        "no_gates": "no pass/fail gates, no retries, no raw-row filtering; "
                    "failures preserved; missing numbers never zeroed",
        "performance_inclusion_note": RC_NOTE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source": source,
        "metric_definitions": {col: label for col, label in METRICS},
        "run_json": run_json,
        "rows_raw": rows,
        "per_arm": per_arm,
        "paired": paired,
        "failures_and_engagement": failures,
        "limitations": limitations,
    }

    json_path = out_dir / "fig13_fast_analysis.json"
    md_path = out_dir / "fig13_fast_analysis.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(build_markdown(
        run_dir, source, run_json, run_json_status, per_arm,
        paired, failures, limitations).rstrip() + "\n")
    print(f"written: {json_path}")
    print(f"written: {md_path}")


if __name__ == "__main__":
    main()

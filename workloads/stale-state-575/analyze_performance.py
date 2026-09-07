#!/usr/bin/env python3
"""Phase-align stale-state-575 policy decisions against workload phase windows.

Reads one raw campaign directory (e.g. raw/stale-state-575-performance-gds-20260907-01),
streams each cell's policy-decisions.jsonl[.gz] line by line without sorting or
loading it fully, and reports snapshot-age / wrong-phase / outside-phase counts
per cell plus per-arm medians. Offline, stdlib only.

Usage: analyze_performance.py --input CAMPAIGN_DIR --output RESULT_JSON [--skip-final]
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from statistics import median

MARKER = '"policy_decision"'
CELL_RE = re.compile(r"^block-(\d+)-(.+)$")
SCHEMA = "stale-state-575-analyze-performance-v1"


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def load_json(path, issues, label):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        issues.append("%s unread/parse (%s): %s" % (label, path, exc))
        return None


def load_phases(cell_dir, issues):
    path = os.path.join(cell_dir, "workload-result.json")
    if not os.path.isfile(path):
        issues.append("missing: %s" % path)
        return None
    data = load_json(path, issues, "workload-result.json")
    if data is None:
        return None
    phases = []
    entries = data.get("phases")
    if not isinstance(entries, list):
        issues.append("workload-result.json: missing phases[]")
        return None
    for entry in entries:
        try:
            start, end = entry["start_mono_ns"], entry["end_mono_ns"]
            name = str(entry["phase"])
            if isinstance(start, bool) or isinstance(end, bool):
                raise TypeError
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise TypeError
        except (KeyError, TypeError):
            issues.append("workload-result.json: invalid phase entry %r" % (entry,))
            continue
        phases.append({"phase": name, "start_mono_ns": start, "end_mono_ns": end})
    if not phases:
        issues.append("workload-result.json: no usable phase windows")
        return None
    return phases, path


def load_policy_final(cell_dir, issues):
    path = os.path.join(cell_dir, "policy-final.json")
    if not os.path.isfile(path):
        issues.append("missing: %s" % path)
        return None
    data = load_json(path, issues, "policy-final.json")
    if data is None and os.path.isfile(path + ".gz"):
        try:
            with gzip.open(path + ".gz", "rt", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            issues.append("policy-final.json.gz unread/parse: %s" % exc)
    return data


def load_final_uvm(cell_dir, issues):
    plain = os.path.join(cell_dir, "uvm-events.jsonl")
    if os.path.isfile(plain):
        path = plain
    elif os.path.isfile(plain + ".gz"):
        path = plain + ".gz"
    else:
        issues.append("missing: uvm-events.jsonl[.gz]")
        return None
    last = None
    try:
        with open_text(path) as fh:
            for line in fh:
                if "final_uvm_stats" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    issues.append("uvm-events.jsonl: unparsable final_uvm_stats line")
                    continue
                if isinstance(rec, dict) and rec.get("event") == "final_uvm_stats":
                    last = rec
    except OSError as exc:
        issues.append("uvm-events.jsonl read failed: %s" % exc)
        return last
    if last is None:
        issues.append("no final_uvm_stats event in %s" % os.path.basename(path))
    return last


def match_phase(phases, ts):
    for phase in phases:
        if phase["start_mono_ns"] <= ts < phase["end_mono_ns"]:
            return phase["phase"]
    return None


def stream_decisions(path, phases):
    lines = events = matched = 0
    wrong = outside = age_missing = parse_errors = invalid_ts = 0
    age_sum = age_n = 0
    non_event = 0
    snap_counts = {}
    per_phase = {}
    with open_text(path) as fh:
        for line in fh:
            lines += 1
            if MARKER not in line:
                non_event += 1
                continue
            events += 1
            try:
                rec = json.loads(line)
            except ValueError:
                parse_errors += 1
                continue
            if not isinstance(rec, dict):
                invalid_ts += 1
                continue
            ts = rec.get("decision_mono_ns")
            if isinstance(ts, bool) or not isinstance(ts, (int, float)):
                invalid_ts += 1
                continue
            snap_phase = rec.get("snapshot_phase")
            if isinstance(snap_phase, str):
                snap_counts[snap_phase] = snap_counts.get(snap_phase, 0) + 1
            actual = match_phase(phases, ts)
            if actual is None:
                outside += 1
                continue
            matched += 1
            stats = per_phase.setdefault(
                actual,
                {"measured_count": 0, "wrong_phase_count": 0, "age_sum_ns": 0, "age_n": 0},
            )
            stats["measured_count"] += 1
            wrong_dec = snap_phase != actual
            if wrong_dec:
                wrong += 1
                stats["wrong_phase_count"] += 1
            age = rec.get("decision_age_ns")
            if isinstance(age, bool) or not isinstance(age, (int, float)):
                age_missing += 1
            else:
                age_sum += age
                age_n += 1
                stats["age_sum_ns"] += age
                stats["age_n"] += 1
    return {
        "lines": lines,
        "events": events,
        "non_event_lines": non_event,
        "parse_errors": parse_errors,
        "invalid_ts": invalid_ts,
        "measured": matched,
        "wrong": wrong,
        "outside": outside,
        "age_missing": age_missing,
        "age_sum": age_sum,
        "age_n": age_n,
        "snap_counts": snap_counts,
        "per_phase": per_phase,
    }


def analyze_cell(cell_dir, skip_final):
    name = os.path.basename(cell_dir)
    obs = {}
    issues = []
    match = CELL_RE.match(name)
    obs["cell"] = name
    obs["block"] = int(match.group(1)) if match else None
    obs["arm"] = match.group(2) if match else name
    obs["dir"] = os.path.abspath(cell_dir)

    loaded = load_phases(cell_dir, issues)
    if loaded is None:
        obs["error"] = "workload-result.json unavailable"
        obs["status"] = "error"
        obs["issues"] = issues
        return obs, False
    phases, wl_path = loaded
    obs["workload_result_path"] = wl_path
    obs["phases"] = phases

    plain = os.path.join(cell_dir, "policy-decisions.jsonl")
    if os.path.isfile(plain):
        dec_path, fmt = plain, "plain"
    elif os.path.isfile(plain + ".gz"):
        dec_path, fmt = plain + ".gz", "gzip"
    elif "uvm_default" in name:
        obs["status"] = "not_applicable"
        obs["error"] = None
        obs["decisions_path"] = None
        obs["decision_lines"] = None
        obs["policy_decision_events"] = None
        obs["non_event_lines"] = None
        obs["parse_error_count"] = None
        obs["invalid_timestamp_count"] = None
        obs["measured_count"] = None
        obs["outside_phase_count"] = None
        obs["wrong_phase_count"] = None
        obs["wrong_phase_fraction"] = None
        obs["age_missing_count"] = None
        obs["mean_age_ns"] = None
        obs["phase_stats"] = None
        obs["snapshot_phase_counts"] = None
        if not skip_final:
            if os.path.isfile(os.path.join(cell_dir, "policy-final.json")):
                obs["policy_final"] = load_policy_final(cell_dir, issues)
            else:
                obs["policy_final"] = None
            obs["final_uvm_stats"] = load_final_uvm(cell_dir, issues)
        obs["issues"] = issues
        return obs, True
    else:
        obs["error"] = "policy-decisions.jsonl[.gz] missing"
        obs["status"] = "error"
        obs["issues"] = issues
        return obs, False
    obs["decisions_path"] = dec_path
    obs["decisions_format"] = fmt

    try:
        counts = stream_decisions(dec_path, phases)
    except OSError as exc:
        obs["error"] = "streaming decisions failed: %s" % exc
        obs["status"] = "error"
        obs["issues"] = issues
        return obs, False

    measured = counts["measured"]
    obs["error"] = None
    obs["status"] = "measured"
    obs["decision_lines"] = counts["lines"]
    obs["policy_decision_events"] = counts["events"]
    obs["non_event_lines"] = counts["non_event_lines"]
    obs["parse_error_count"] = counts["parse_errors"]
    obs["invalid_timestamp_count"] = counts["invalid_ts"]
    obs["measured_count"] = measured
    obs["outside_phase_count"] = counts["outside"]
    obs["wrong_phase_count"] = counts["wrong"]
    obs["wrong_phase_fraction"] = (
        counts["wrong"] / measured if measured else None
    )
    obs["age_missing_count"] = counts["age_missing"]
    obs["mean_age_ns"] = (
        counts["age_sum"] / counts["age_n"] if counts["age_n"] else None
    )
    obs["phase_stats"] = {
        name_: {
            "measured_count": stats["measured_count"],
            "wrong_phase_count": stats["wrong_phase_count"],
            "mean_age_ns": (
                stats["age_sum_ns"] / stats["age_n"] if stats["age_n"] else None
            ),
        }
        for name_, stats in sorted(counts["per_phase"].items())
    }
    obs["snapshot_phase_counts"] = counts["snap_counts"]

    if not skip_final:
        obs["policy_final"] = load_policy_final(cell_dir, issues)
        obs["final_uvm_stats"] = load_final_uvm(cell_dir, issues)
    obs["issues"] = issues
    return obs, measured > 0


def arm_summary(cells):
    groups = {}
    for obs in cells:
        groups.setdefault(obs.get("arm"), []).append(obs)
    arms = {}
    for arm in sorted(groups, key=str):
        members = groups[arm]
        fracs = [
            c["wrong_phase_fraction"]
            for c in members
            if c.get("wrong_phase_fraction") is not None
        ]
        ages = [c["mean_age_ns"] for c in members if c.get("mean_age_ns") is not None]
        outside = [
            c["outside_phase_count"]
            for c in members
            if isinstance(c.get("outside_phase_count"), int)
        ]
        arms[arm] = {
            "cells": len(members),
            "usable_cells": len(fracs),
            "median_wrong_phase_fraction": median(fracs) if fracs else None,
            "median_mean_age_ns": median(ages) if ages else None,
            "total_outside_phase_count": sum(outside),
        }
    return arms


def cell_dirs(campaign, errors):
    names = []
    for name in sorted(os.listdir(campaign)):
        path = os.path.join(campaign, name)
        if not os.path.isdir(path):
            continue
        for fname in (
            "workload-result.json",
            "policy-decisions.jsonl",
            "policy-decisions.jsonl.gz",
        ):
            if os.path.isfile(os.path.join(path, fname)):
                names.append(path)
                break
    if not names:
        errors.append("no cell directories found in %s" % campaign)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--input", required=True, help="raw campaign directory")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument(
        "--skip-final", action="store_true",
        help="skip policy-final.json / final_uvm_stats extraction",
    )
    args = parser.parse_args(argv)

    started = time.time()
    campaign = os.path.abspath(args.input)
    errors = []
    if not os.path.isdir(campaign):
        errors.append("campaign directory not found: %s" % campaign)
        cells = []
    else:
        cells = [analyze_cell(path, args.skip_final) for path in cell_dirs(campaign, errors)]

    observations = [obs for obs, _ in cells]
    ok_cells = sum(1 for _, ok in cells if ok)
    measured_cells = sum(
        1 for c in observations
        if isinstance(c.get("measured_count"), int) and c["measured_count"] > 0
    )
    napp_cells = sum(1 for c in observations if c.get("status") == "not_applicable")
    total_measured = sum(
        c["measured_count"] for c in observations
        if isinstance(c.get("measured_count"), int)
    )
    result = {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 1),
        "rawpath": campaign,
        "errors": errors,
        "cells_ok": ok_cells,
        "cells_with_data": measured_cells,
        "cells_not_applicable": napp_cells,
        "total_measured_count": total_measured,
        "arms": arm_summary(observations),
        "cells": observations,
    }

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=False)
            fh.write("\n")
    except OSError as exc:
        sys.stderr.write("error: cannot write %s: %s\n" % (out_path, exc))
        return 1

    print("wrote %s (%d cells: %d measured, %d not-applicable, %d ok; %d aligned decisions)"
          % (out_path, len(observations), measured_cells, napp_cells, ok_cells,
             total_measured))
    for obs in observations:
        if obs.get("error"):
            print("%-28s ERROR %s" % (obs["cell"], obs["error"]))
            for issue in obs.get("issues", []):
                print("    %s" % issue)
            continue
        if obs.get("status") == "not_applicable":
            print("%-28s not_applicable (uvm_default; no policy decisions)" % obs["cell"])
            continue
        frac = obs["wrong_phase_fraction"]
        mean_age = obs["mean_age_ns"]
        print("%-28s measured=%d wrong=%d (%.4f) outside=%d mean_age_ns=%s"
              % (obs["cell"], obs["measured_count"], obs["wrong_phase_count"],
                 frac if frac is not None else float("nan"),
                 obs["outside_phase_count"],
                 "n/a" if mean_age is None else "%.0f" % mean_age))
    for arm, summary in result["arms"].items():
        print("arm %-22s cells=%d median_wrong=%.4f median_mean_age_ns=%.0f"
              % (arm, summary["cells"],
                 summary["median_wrong_phase_fraction"]
                 if summary["median_wrong_phase_fraction"] is not None else float("nan"),
                 summary["median_mean_age_ns"]
                 if summary["median_mean_age_ns"] is not None else 0.0))
    return 0 if observations and ok_cells > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

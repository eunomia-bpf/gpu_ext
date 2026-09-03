"""Independently audit and recompute a frozen five-block MoE campaign offline.

Usage: .venv/bin/python -B analyze_paper_comparison.py CAMPAIGN [--output NEW.json]
Exit codes: 0 complete, 1 incomplete, 2 rejected artifacts or input/output error.
No admission, GPU commands, subprocesses, torch or NumPy are used. The existing
raw auditor requires the manifest's runtime/model/corpus paths to remain present
and unchanged; this is not a relocation or missing-artifact bypass.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

import paper_result_audit as audit
import run_paper_comparison as runner


def read_manifest(campaign):
    manifest = audit._json(campaign, "manifest.json")
    audit.require(isinstance(manifest, dict), "manifest must be a JSON object")
    fixed = {"protocol": runner.PROTOCOL, "required_blocks": runner.BLOCKS,
             "warmup_prompt": 0, "measured_input_output_tokens": [512, 64],
             "memory_budget": 0.75, "kv_blocks": 128,
             "timing_shadow_verification": False}
    for key, value in fixed.items():
        actual = manifest.get(key)
        audit.require(type(actual) is type(value) and actual == value,
                      f"manifest protocol field differs: {key}")
    seed = manifest.get("seed")
    audit.require(type(seed) is int, "manifest seed must be an integer")
    schedule = manifest.get("schedule")
    audit.require(isinstance(schedule, list) and len(schedule) == runner.BLOCKS,
                  "manifest must contain five scheduled blocks")
    for item in schedule:
        audit.require(isinstance(item, dict) and type(item.get("block")) is int and
                      isinstance(item.get("prompts"), list) and
                      all(type(number) is int for number in item["prompts"]),
                      "invalid schedule item")
    audit.require(schedule == runner.schedule(seed), "manifest differs from seeded schedule")
    runtime = manifest.get("runtime_inventory")
    audit.require(isinstance(runtime, dict) and isinstance(manifest.get("driver_stage"), str)
                  and runtime.get("driver_stage") == manifest["driver_stage"],
                  "manifest runtime/driver inventory differs")
    return manifest


def secondary_ttft(blocks):
    """Paired block GM of the pre-existing first-visible-text latency metric.

    Match the primary analysis's deterministic 10,000-draw percentile bootstrap,
    resampling whole paired blocks rather than treating requests as independent.
    """
    result = {"priority": "secondary", "metric": "first_text_ttft_median_ms",
              "definition": "per-cell median first nonempty text, not first model token",
              "direction": "lower_is_better", "unit": "ms", "blocks": len(blocks),
              "bootstrap_draws": 10000, "bootstrap_seed": runner.SEED + 1, "paired": {}}
    if not blocks:
        return result
    rows = [{cell["mode"]: cell["first_text_ttft_median_ms"] for cell in block["cells"]}
            for block in blocks]
    for row in rows:
        for value in row.values():
            audit._number(value, "first_text_ttft_median_ms", positive=True)
    rng = random.Random(runner.SEED + 1)
    samples = [[rng.randrange(len(blocks)) for _ in blocks] for _ in range(10000)]
    for numerator, denominator in (("paper-bpf", "paper-native"),
                                   ("paper-native", "native-off"), ("paper-bpf", "native-off")):
        logs = [math.log(row[numerator] / row[denominator]) for row in rows]
        ci = None
        if len(blocks) >= 2:
            boot = sorted(math.exp(statistics.mean(logs[i] for i in sample)) for sample in samples)
            ci = [boot[249], boot[9749]]
        result["paired"][f"{numerator}/{denominator}"] = {
            "geometric_ttft_ratio": math.exp(statistics.mean(logs)),
            "paired_block_bootstrap_ci95": ci, "interpretation": "ratio < 1 favors numerator"}
    return result


def analyze_campaign(campaign):
    campaign = Path(campaign)
    audit.require(campaign.is_dir() and not campaign.is_symlink(), "missing/aliased campaign")
    campaign = campaign.resolve()
    manifest = read_manifest(campaign)
    attempts = {item["block"]: [] for item in manifest["schedule"]}
    failed, incomplete, rejected, accepted, valid = [], [], [], [], []
    for path in sorted(campaign.iterdir()):
        if not path.name.startswith("block-"):
            continue
        match = re.fullmatch(r"block-([0-9]{2})-attempt-([0-9]{2,})", path.name)
        number, attempt = map(int, match.groups()) if match else (0, 0)
        canonical = f"block-{number:02d}-attempt-{attempt:02d}"
        if (number not in attempts or attempt < 1 or path.name != canonical
                or not path.is_dir() or path.is_symlink()):
            rejected.append({"path": str(path), "reason": "unexpected/nonregular attempt"})
        else:
            attempts[number].append(path)

    for item in manifest["schedule"]:
        successful = []
        for path in attempts[item["block"]]:
            record = {"block": item["block"], "path": str(path)}
            result_path = path / "result.json"
            if not result_path.exists() and not result_path.is_symlink():
                incomplete.append({**record, "reason": "block result.json not published"})
                continue
            try:
                result = audit._json(path, "result.json")
                audit.require(isinstance(result, dict) and type(result.get("passed")) is bool,
                              "block result must contain a boolean passed field")
            except (OSError, ValueError, TypeError, audit.AuditError) as exc:
                rejected.append({**record, "reason": str(exc)})
                continue
            if not result["passed"]:
                failed.append({**record, "reason": "producer recorded failure",
                               "failure": {key: result[key] for key in
                                   ("error_type", "error", "cleanup_error", "cleanup_errors")
                                   if key in result}})
                continue
            # Audit every success claim, including duplicates. Never select the
            # one whose surviving files or reported performance look preferable.
            try:
                block = audit.audit_block(path, item, manifest["runtime_inventory"])
                successful.append((record, block, None))
            except audit.AuditError as exc:
                successful.append((record, None, str(exc)))
        duplicate = len(successful) > 1
        for record, block, error in successful:
            if duplicate or error is not None:
                reason = "multiple successful attempts for one scheduled block" if duplicate else error
                rejected.append({**record, "reason": reason,
                                 **({"audit_error": error} if error is not None else {})})
            else:
                valid.append(block)
                accepted.append(record)

    analysis = runner.analyze(valid)
    complete = analysis["complete"] and not rejected
    return {"campaign": str(campaign), "audit": "paper_result_audit.audit_block",
            "schedule": manifest["schedule"], "analysis": analysis,
            "secondary": {"first_visible_text_ttft": secondary_ttft(valid)},
            "accepted_attempts": accepted, "failed_attempts": failed,
            "incomplete_attempts": incomplete, "rejected_attempts": rejected,
            "unverified_blocks": [number for number in attempts
                                  if number not in {block["block"] for block in valid}],
            "complete": complete,
            "status": "rejected" if rejected else "complete" if complete else "incomplete"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, help="create a new report; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(f"refusing to overwrite {args.output}")
        report = analyze_campaign(args.campaign)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        sys.stdout.write(text)
        return {"complete": 0, "incomplete": 1, "rejected": 2}[report["status"]]
    except (OSError, ValueError, TypeError, KeyError, audit.AuditError) as exc:
        print(json.dumps({"status": "error", "complete": False,
                          "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

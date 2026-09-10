#!/usr/bin/env python3
"""Independent CPU-only audit of the adaptive-prefetch governor campaign.

Recomputes the validity gates, per-arm medians, whole-block paired geometric
throughput ratios and 10k-draw paired-block bootstrap CIs directly from the
per-cell result.json records, then cross-checks them against the runner's
analysis.json. No RNG cell resampling: the block-resample index stream is
reseeded with the same published seed (20260910) and regenerated identically.

Read-only; no hashes. Paths under raw/ are immutable evidence.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "workloads/moe-infinity/raw/adaptive-prefetch-575"

ARMS = ("unbounded-native", "fixed-native", "adaptive-native",
        "adaptive-bpf", "demand-only")
CELL_POSITIONS = (0, 1, 2, 3, 0, 1)
BLOCKS = 5
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 10000

COMPARISONS = (
    ("adaptive-bpf", "unbounded-native"),
    ("adaptive-native", "unbounded-native"),
    ("adaptive-bpf", "adaptive-native"),
    ("fixed-native", "adaptive-bpf"),
    ("unbounded-native", "demand-only"),
    ("adaptive-bpf", "demand-only"),
)


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    campaign = REL / "full-20260910-s"
    preflight = REL / "preflight-20260910-s"
    analysis = read(campaign / "analysis.json")
    manifest = read(campaign / "manifest.json")
    if not isinstance(analysis, dict) or not isinstance(manifest, dict):
        print("FAIL: analysis.json or manifest.json missing/unreadable")
        return
    problems: list[str] = []

    # -- evidence inventory ------------------------------------------------
    print("== campaign inventory ==")
    print(f"manifest: blocks={manifest['required_blocks']} "
          f"cells_planned={manifest['planned_cells']} "
          f"requests_planned={manifest['planned_measured_requests']}")
    pf = read(preflight / "result.json")
    print(f"preflight-s passed: {pf.get('passed')}, arms: {[c['arm'] for c in pf.get('cells', [])]}")

    goldens = read(REL / "held-out-goldens.json")
    # One golden per cohort row: cell_position 0..3 indexes the row; the
    # six cell slots map onto rows via request_positions (A=[0,1] B=[2,3]
    # repeated), enforced per-cell by the positions check below.
    gold_rows = [e["cell_position"] for e in goldens["goldens"]]
    print(f"frozen goldens rows: {gold_rows} (cohort indices "
          f"{[e['source_index'] for e in goldens['goldens']]})")
    if gold_rows != [0, 1, 2, 3]:
        problems.append("goldens must cover exactly the four cohort rows")

    # -- per-block recomputation -------------------------------------------
    print("\n== per-block cell recomputation from result.json artifacts ==")
    valid = []
    for block in range(1, BLOCKS + 1):
        brec = read(campaign / f"block-{block:02d}-attempt-01" / "result.json")
        if not isinstance(brec, dict):
            problems.append(f"block {block}: result.json missing")
            continue
        cells = brec.get("cells", [])
        # per-arm cell dirs hold the authoritative result.json
        by_arm = {}
        for arm in ARMS:
            crec = read(campaign / f"block-{block:02d}-attempt-01" / arm / "result.json")
            if crec is None or None or not isinstance(crec, dict):
                problems.append(f"block {block} {arm}: cell result.json missing")
                continue
            reqs = crec.get("requests", [])
            checks = {
                "cell passed": crec.get("passed") is True,
                "six requests": len(reqs) == len(CELL_POSITIONS),
                "positions": crec.get("request_positions") == brec.get("request_positions") == list(CELL_POSITIONS),
                "exit 0": crec.get("server_exit_code") == 0,
                "cleanup clean": crec.get("cleanup_errors") == [],
                "no error": crec.get("error") in (None, ""),
                "verified counters": crec.get("verified_requests") == 6 and crec.get("verified_output_tokens") == 384,
            }
            positions_ok = all(r.get("passed") is True for r in reqs)
            thr = crec.get("output_throughput_tokens_per_s", 0)
            checks["throughput finite"] = isinstance(thr, (int, float)) and math.isfinite(thr) and thr > 0
            checks["requests all passed"] = positions_ok
            bad = [k for k, v in checks.items() if not v]
            if bad:
                problems.append(f"block {block} {arm}: failed checks {bad}")
            else:
                by_arm[arm] = crec
        if len(by_arm) == len(ARMS) and brec.get("passed") is True:
            valid.append((block, by_arm))
            e2e = {arm: round(statistics.median(
                r["e2e_ms"] for r in by_arm[arm]["requests"]), 1) for arm in ARMS}
            print(f"block {block}: valid; cell-median e2e ms {e2e}")
        else:
            problems.append(f"block {block}: not all five arms valid (have {sorted(by_arm)})")

    print(f"\nvalid blocks: {len([b for b, _ in valid])}/{BLOCKS}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("problems: none")

    # -- per-arm medians and cross-check vs analysis.json -------------------
    print("\n== per-arm medians vs runner analysis.json ==")
    for arm in ARMS:
        vals = [statistics.median(r["e2e_ms"] for r in by_arm[arm]["requests"])
                for _, by_arm in valid]
        mine = statistics.median(vals)
        ref = analysis["modes"][arm]["e2e_median_ms"]
        tag = "match" if abs(mine - ref) < 1e-9 else "MISMATCH, runner differs"
        print(f"  {arm:<16} e2e_median recomputed={mine:9.4f} "
              f"analysis={ref:9.4f} ({tag})")
        if tag != "match":
            problems.append(f"{arm}: e2e median mismatch")

    # -- paired geometric ratios + bootstrap, cross-check -------------------
    print("\n== paired whole-block geometric throughput ratios (recomputed) ==")
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [[rng.randrange(max(len(valid), 1)) for _ in valid]
               for _ in range(BOOTSTRAP_DRAWS)]
    for numerator, denominator in COMPARISONS:
        logs = [math.log(by_arm[numerator]["output_throughput_tokens_per_s"] /
                         by_arm[denominator]["output_throughput_tokens_per_s"])
                for _, by_arm in valid]
        boot = sorted(math.exp(statistics.mean(logs[i] for i in sample))
                      for sample in samples)
        mine_ratio = math.exp(statistics.mean(logs))
        mine_ci = [boot[249], boot[9749]]
        ref = analysis["paired"][f"{numerator}/{denominator}"]
        ratio_ok = abs(mine_ratio - ref["geometric_throughput_ratio"]) < 1e-9
        ci_ok = all(abs(a - b) < 1e-9 for a, b in zip(mine_ci, ref["paired_block_bootstrap_ci95"]))
        status = "match" if (ratio_ok and ci_ok) else "MISMATCH"
        print(f"  {numerator}/{denominator}:")
        print(f"    ratio recomputed={mine_ratio:.10f} analysis={ref['geometric_throughput_ratio']:.10f}"
              f" ({'ok' if ratio_ok else 'MISMATCH'})")
        print(f"    ci95 recomputed=[{mine_ci[0]:.10f}, {mine_ci[1]:.10f}] "
              f"analysis=[{ref['paired_block_bootstrap_ci95'][0]:.10f}, {ref['paired_block_bootstrap_ci95'][1]:.10f}]"
              f" ({'ok' if ci_ok else 'MISMATCH'})")
        if status != "match":
            problems.append(f"{numerator}/{denominator}: paired stats mismatch")

    # -- conservation / engagement sanity from counters --------------------
    print("\n== governor engagement (summed over valid blocks, recomputed) ==")
    for arm in ("adaptive-native", "adaptive-bpf"):
        calls = submitted = admitted = updates = decr = incr = 0
        for _, by_arm in valid:
            gd = by_arm[arm].get("governor_delta") or {}
            calls += gd.get("governor_admission_calls", 0)
            submitted += gd.get("governor_admission_submitted", 0)
            admitted += gd.get("governor_admission_admitted", 0)
            updates += gd.get("governor_budget_updates", 0)
            decr += gd.get("governor_budget_decreases", 0)
            incr += gd.get("governor_budget_increases", 0)
        ok = admitted <= submitted and calls > 0 and updates > 0
        print(f"  {arm}: calls={calls} submitted={submitted} admitted={admitted} "
              f"budget_updates={updates} (dec={decr},inc={incr}) "
              f"{'ok' if ok else 'ENGAGEMENT PROBLEM'}")
        if not ok:
            problems.append(f"{arm}: governor engagement inconsistent")
    for arm in ("fixed-native", "adaptive-native", "adaptive-bpf"):
        final = statistics.median(int((by_arm[arm].get("governor_after") or {}).get("governor_budget_bytes", 0))
                                  for _, by_arm in valid)
        print(f"  {arm}: median final budget bytes = {final}")

    # -- verification totals ------------------------------------------------
    print("\n== verification totals vs analysis.json ==")
    for key in ("valid_blocks", "valid_cells", "verified_measured_requests", "verified_output_tokens"):
        mine = len(valid) if key == "valid_blocks" else None
        if key == "valid_cells":
            mine = len(valid) * len(ARMS)
        elif key == "verified_measured_requests":
            mine = len(valid) * len(ARMS) * len(CELL_POSITIONS)
        elif key == "verified_output_tokens":
            mine = len(valid) * len(ARMS) * len(CELL_POSITIONS) * 64
        ref = analysis[key]
        tag = "match" if mine == ref else "MISMATCH"
        print(f"  {key}: recomputed={mine} analysis={ref} ({tag})")
        if tag != "match":
            problems.append(f"{key} mismatch")
    print(f"  complete: analysis={analysis['complete']}; recomputed={len(valid) == BLOCKS}")

    print("\n== verdict ==")
    if problems:
        print(f"FAIL: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("AUDIT PASSED: blocks, cells, medians, paired ratios, bootstrap CIs, "
          "engagement counters and totals all reproduce from raw result.json.")


if __name__ == "__main__":
    main()

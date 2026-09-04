#!/usr/bin/env python3
"""Independently replay and analyze the fixed-work precision campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import analyze_fixed_work as base
import run_fixed_work_precision as precision


def analyze(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    precision.configure()
    analysis = base.analyze(result, result_path)
    analysis["precision_design"] = {
        "prior_campaign_pooled": False,
        "blocks": precision.PRECISION_BLOCKS,
        "warmup_launches": precision.PRECISION_WARMUP,
        "timed_launches_per_cuda_event_interval": precision.PRECISION_LAUNCHES,
        "hook_repeats_per_thread_per_launch": precision.PRECISION_HOOK_REPEATS,
        "fixed_sample_no_optional_stopping": True,
        "old_to_new_launch_aggregation_ratio": 64,
    }
    return analysis


def render_markdown(analysis: dict[str, Any]) -> str:
    rendered = base.render_markdown(analysis)
    return rendered.replace(
        "# Fixed-work trampoline analysis",
        "# Fixed-work trampoline precision analysis",
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    analysis = analyze(result, args.result)
    prefix = args.output_prefix or args.result.parent / "fixed-work-precision-analysis"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    prefix.with_suffix(".md").write_text(render_markdown(analysis))
    print(json.dumps({"status": analysis["run_status"], "output": str(prefix)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

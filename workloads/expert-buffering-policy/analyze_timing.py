#!/usr/bin/env python3
"""Summarize the frozen five-block paired expert-policy timing run."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive observations")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_summary(ratios: list[float]) -> dict[str, Any]:
    # Enumerate every size-N paired bootstrap resample. With five blocks this is
    # only 5^5 = 3,125 samples and avoids a random-seed dependency.
    resampled = [
        geometric_mean([ratios[index] for index in selection])
        for selection in itertools.product(range(len(ratios)), repeat=len(ratios))
    ]
    estimate = geometric_mean(ratios)
    low = quantile(resampled, 0.025)
    high = quantile(resampled, 0.975)
    return {
        "block_ratios": ratios,
        "geometric_mean_ratio": estimate,
        "effect_percent": 100 * (estimate - 1),
        "bootstrap_method": "exact enumeration of all size-5 paired resamples",
        "bootstrap_samples": len(resampled),
        "ci95_ratio": [low, high],
        "ci95_effect_percent": [100 * (low - 1), 100 * (high - 1)],
    }


def paired_difference_summary(differences: list[float]) -> dict[str, Any]:
    resampled = [
        sum(differences[index] for index in selection) / len(differences)
        for selection in itertools.product(
            range(len(differences)), repeat=len(differences)
        )
    ]
    estimate = sum(differences) / len(differences)
    return {
        "block_differences_bytes": differences,
        "mean_difference_bytes": estimate,
        "bootstrap_method": "exact enumeration of all size-5 paired resamples",
        "bootstrap_samples": len(resampled),
        "ci95_difference_bytes": [
            quantile(resampled, 0.025), quantile(resampled, 0.975)
        ],
    }


def analyze(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 5:
        raise ValueError("timing result must contain exactly five blocks")
    if [int(block["block"]) for block in blocks] != [1, 2, 3, 4, 5]:
        raise ValueError("timing blocks must be ordered 1 through 5")
    if any(block.get("status") != "passed" for block in blocks):
        raise ValueError("every timing block must be passed")

    throughput = [block["throughput_tokens_per_s"] for block in blocks]
    activation = [
        (float(block["gpubpf_profile_protect"]["repeated_hot_activation_bytes"]),
         float(block["gpubpf_observe"]["repeated_hot_activation_bytes"]))
        for block in blocks
    ]
    return {
        "blocks": 5,
        "mechanism_observe_over_plain": paired_summary([
            float(value["gpubpf_observe"]) / float(value["plain_uvm"])
            for value in throughput
        ]),
        "policy_protect_over_observe": paired_summary([
            float(value["gpubpf_profile_protect"]) / float(value["gpubpf_observe"])
            for value in throughput
        ]),
        "context_llama_ncmoe32_over_plain": paired_summary([
            float(value["llama_ncmoe32"]) / float(value["plain_uvm"])
            for value in throughput
        ]),
        "repeated_hot_activation_protect_over_observe": paired_summary([
            protected / observed for protected, observed in activation
        ]),
        "repeated_hot_activation_protect_minus_observe": paired_difference_summary([
            protected - observed for protected, observed in activation
        ]),
        "arithmetic_mean_throughput_tokens_per_s": {
            config: sum(float(value[config]) for value in throughput) / len(throughput)
            for config in (
                "plain_uvm", "gpubpf_observe", "gpubpf_profile_protect", "llama_ncmoe32"
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=HERE / "timing-results.json"
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.input), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

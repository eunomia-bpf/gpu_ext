"""Projection and rendering tests; synthetic fixtures are not experiment data."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plot_phase_full as plot


def fixture() -> dict:
    arms = plot.ARMS

    def arm_values(base: float) -> dict[str, list[float]]:
        return {arm: [base * scale + block for block in range(plot.BLOCKS)]
                for arm, scale in zip(arms, (1.0, 1.1, 1.3))}

    def medians(values: dict[str, list[float]]) -> dict[str, float]:
        return {arm: values[arm][2] for arm in arms}

    def ratio(numerator: list[float], denominator: list[float]) -> dict:
        values = [left / right for left, right in zip(numerator, denominator)]
        estimate = math.exp(sum(math.log(value) for value in values) / len(values))
        return {"geometric_mean_ratio": estimate, "block_ratios": values,
                "confidence_interval_95": [estimate * .99, estimate * 1.01],
                "blocks": plot.BLOCKS, "lower_is_better": True}

    operator_blocks = {metric: arm_values(10 + index * 10)
                       for index, (metric, _) in enumerate(plot.OPERATOR_METRICS)}
    phase_blocks = {
        "steady_samples_ns": arm_values(1000),
        "pre_python_main_ns": arm_values(100),
    }
    return {
        "complete": True,
        "formal_complete": True,
        "protocol": "pod-device-setup-phases-v1",
        "numeric_protocol": "pod-fp16-upstream-match-v2",
        "fresh_process_cells": 15,
        "blocks": 5,
        "measured_operator_samples_per_cell": 100,
        "phase_observations_per_arm": 5,
        "phase_estimator": "median of five fresh-process block durations",
        "ratio_estimator": "geometric mean of five paired block ratios; lower is better",
        "uncertainty": {"method": "whole-block percentile bootstrap with shared resamples",
                        "draws": 10000, "seed": 20260907, "confidence": .95,
                        "scope": ("pointwise intervals; no equivalence test or "
                                  "multiple-comparison adjustment")},
        "claim_boundary": ("RTX 5090; not a generic attach-latency estimate; "
                           "not operator latency or an end-to-end serving workload"),
        "operator_latency": {
            "cell_estimator": "arithmetic mean of all 100 unfiltered synchronized samples",
            "block_means_ms": operator_blocks,
            "median_of_five_cell_means_ms": {
                metric: medians(values) for metric, values in operator_blocks.items()},
            "paired_ratios": {"device_bpf_vs_cuda_adapter": {
                metric: ratio(values["pod_bpf"], values["pod_cuda"])
                for metric, values in operator_blocks.items()}},
        },
        "block_phase_ms": phase_blocks,
        "median_phase_ms": {key: medians(values) for key, values in phase_blocks.items()},
        "paired_ratios": {"device_bpf_vs_cuda_adapter": {
            key: ratio(values["pod_bpf"], values["pod_cuda"])
            for key, values in phase_blocks.items()}},
    }


class PlotPhaseFullTests(unittest.TestCase):
    def test_real_analysis_has_exact_three_boundaries(self):
        data = plot.load_plot_data()
        self.assertEqual(len(data["operator"]), 2)
        self.assertEqual(data["blocks"], 5)
        self.assertEqual(data["samples_per_cell"], 100)
        self.assertAlmostEqual(data["operator"][0]["overhead_percent"], 1.77717481572053)
        self.assertAlmostEqual(data["operator"][1]["overhead_percent"], 1.80937799875418)
        self.assertAlmostEqual(data["steady"]["ratio"], 2.372263830369445)
        self.assertAlmostEqual(data["cold"]["medians_s"]["pod_bpf"], 271.224901753)
        self.assertAlmostEqual(data["cold"]["ratio"], 12680.36561999033)

    def test_incomplete_bad_median_ratio_and_nonfinite_rejected(self):
        changes = [
            lambda data: data.update(complete=False),
            lambda data: data.update(fresh_process_cells=14),
            lambda data: data["operator_latency"]["block_means_ms"]["cuda_ms"]
                ["pod_bpf"].pop(),
            lambda data: data["median_phase_ms"]["steady_samples_ns"]
                .update(pod_bpf=999),
            lambda data: data["paired_ratios"]["device_bpf_vs_cuda_adapter"]
                ["pre_python_main_ns"]["block_ratios"].__setitem__(0, 99),
            lambda data: data["block_phase_ms"]["pre_python_main_ns"]
                ["pod_bpf"].__setitem__(0, float("nan")),
        ]
        for change in changes:
            analysis = fixture()
            change(analysis)
            with self.subTest(change=change), self.assertRaises(ValueError):
                plot.project_analysis(analysis)

    def test_render_uses_fixed_analysis_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=plot.HERE, prefix="phase-plot-test-") as directory:
            prefix = Path(directory) / "synthetic-not-results"
            with patch.object(plot, "load_plot_data", return_value=plot.project_analysis(fixture())):
                outputs = plot.render(prefix)
                self.assertEqual({path.suffix for path in outputs}, {".pdf", ".png", ".md"})
                self.assertTrue(all(path.stat().st_size > 0 for path in outputs))
                with self.assertRaises(FileExistsError):
                    plot.render(prefix)

    def test_output_cannot_escape_workload_directory(self):
        with self.assertRaisesRegex(ValueError, "must remain"):
            plot.output_paths(Path(tempfile.gettempdir()) / "outside-pod-phase")

    def test_fixed_source_is_the_only_read(self):
        payload = json.dumps(fixture())
        with patch.object(Path, "read_text", autospec=True, return_value=payload) as reader:
            plot.load_plot_data()
        reader.assert_called_once_with(plot.SOURCE)


if __name__ == "__main__":
    unittest.main()

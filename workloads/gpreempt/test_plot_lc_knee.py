"""Synthetic contracts for the LC-knee result plot; no raw campaign is read."""

import copy
import tempfile
import unittest
from pathlib import Path

import plot_lc_knee as plot


def fixture():
    audit = {
        "schema": "gpreempt_lc_knee_audit_v1",
        "study": "lc-knee",
        "evidence_role": "supporting",
        "mode": "full",
        "complete": True,
        "formal_eligible": True,
        "formal_complete": True,
        "valid_cells": 27,
        "required_cells": 27,
        "rejected_cells": [],
        "incomplete_cells": [],
        "unexpected_cells": [],
        "prespecified_lc_rates_rps": [500, 625, 800],
        "post_hoc_rate_additions_allowed": False,
        "preflight_gate": {
            "campaign": "/separate/preflight",
            "schema": "gpreempt_lc_knee_audit_v1",
            "study": "lc-knee",
            "mode": "preflight",
            "scenario": "lc800",
            "valid_cells": 3,
            "complete": True,
            "formal_complete": False,
        },
        "scenarios": {},
    }
    for scenario, rate in zip(plot.SCENARIOS, plot.RATES):
        points = []
        for block in range(3):
            for arm_index, arm in enumerate(plot.ARMS):
                coverage = 1.0
                points.append({
                    "block": block,
                    "scenario": scenario,
                    "arm": arm,
                    "begin_ns": block * 100_000_000_000,
                    "end_ns": block * 100_000_000_000 + 60_000_000_000,
                    "metrics": {
                        "vgg_rt": {
                            "offered_requests": rate * 60,
                            "p99_response_us": 1000.0 + rate + arm_index * 100 + block,
                            "completion_coverage": coverage,
                            "p99_is_conditional": False,
                            "response_p99_population": (
                                "all_started_and_verified_including_after_window"
                            ),
                            "p99_sample_count": rate * 60,
                        },
                        "resnet152_be": {"goodput_rps": 200.0 - arm_index * 10 - block},
                    },
                })
        audit["scenarios"][scenario] = {
            "complete": True,
            "valid_paired_blocks": 3,
            "required_blocks": 3,
            "per_cell_points": points,
        }
    return audit


class LCKneePlotTests(unittest.TestCase):
    def test_exact_paired_points_and_medians(self):
        data = plot.plot_data(fixture())
        self.assertEqual(len(data["points"]), 27)
        self.assertEqual(len(data["groups"]), 9)
        for group in data["groups"]:
            self.assertEqual(group["blocks"], [0, 1, 2])
            self.assertEqual(group["cell_count"], 3)
            self.assertEqual(group["median_completion_coverage"], 1.0)
            self.assertFalse(group["any_conditional_p99"])
        native_500 = next(group for group in data["groups"]
                          if group["rate_rps"] == 500 and group["arm"] == "native")
        self.assertEqual(native_500["median_lc_response_p99_ms"], 1.501)
        self.assertEqual(native_500["median_be_goodput_rps"], 199.0)

    def test_incomplete_rejected_preflight_or_unplanned_audit_is_never_plotted(self):
        mutations = (
            lambda a: a.update(complete=False),
            lambda a: a.update(complete=1),
            lambda a: a.update(formal_eligible=False),
            lambda a: a.update(formal_complete=False),
            lambda a: a.update(mode="preflight"),
            lambda a: a.update(schema="wrong"),
            lambda a: a.update(study="load"),
            lambda a: a.update(evidence_role="decisive"),
            lambda a: a.update(valid_cells=26),
            lambda a: a["rejected_cells"].append({"cell": 1}),
            lambda a: a["incomplete_cells"].append({"cell": 1}),
            lambda a: a["unexpected_cells"].append("extra"),
            lambda a: a.update(prespecified_lc_rates_rps=[500, 625, 800, 900]),
            lambda a: a.update(post_hoc_rate_additions_allowed=True),
            lambda a: a.pop("preflight_gate"),
            lambda a: a["preflight_gate"].update(complete=False),
            lambda a: a["preflight_gate"].update(complete=1),
            lambda a: a["preflight_gate"].update(formal_complete=True),
            lambda a: a["preflight_gate"].update(mode="full"),
            lambda a: a["preflight_gate"].update(valid_cells=2),
            lambda a: a["preflight_gate"].update(campaign="relative/preflight"),
            lambda a: a["scenarios"].pop("lc625"),
            lambda a: a["scenarios"]["lc625"].update(complete=False),
            lambda a: a["scenarios"]["lc625"].update(valid_paired_blocks=2),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                audit = fixture()
                mutate(audit)
                with self.assertRaises((ValueError, KeyError)):
                    plot.plot_data(audit)

    def test_cell_schema_and_conditional_coverage_fail_closed(self):
        mutations = (
            lambda p: p.update(block=3),
            lambda p: p.update(arm="unknown"),
            lambda p: p.update(scenario="lc625"),
            lambda p: p.update(end_ns=p["begin_ns"] + 10_000_000_000),
            lambda p: p["metrics"]["vgg_rt"].update(offered_requests=1),
            lambda p: p["metrics"]["vgg_rt"].update(p99_response_us=0),
            lambda p: p["metrics"]["vgg_rt"].update(completion_coverage=None),
            lambda p: p["metrics"]["vgg_rt"].update(completion_coverage=.9),
            lambda p: p["metrics"]["vgg_rt"].update(p99_is_conditional="yes"),
            lambda p: p["metrics"]["vgg_rt"].update(response_p99_population="survivors"),
            lambda p: p["metrics"]["vgg_rt"].update(p99_sample_count=0),
            lambda p: p["metrics"]["resnet152_be"].update(goodput_rps=-1),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                audit = fixture()
                mutate(audit["scenarios"]["lc500"]["per_cell_points"][0])
                with self.assertRaises(ValueError):
                    plot.plot_data(audit)

        duplicate = fixture()
        points = duplicate["scenarios"]["lc500"]["per_cell_points"]
        points[-1] = copy.deepcopy(points[0])
        with self.assertRaises(ValueError):
            plot.plot_data(duplicate)

    def test_conditional_p99_retains_coverage_and_adverse_point(self):
        audit = fixture()
        foreground = audit["scenarios"]["lc800"]["per_cell_points"][0]["metrics"]["vgg_rt"]
        foreground.update(completion_coverage=.75, p99_is_conditional=True,
                          p99_sample_count=36000)
        data = plot.plot_data(audit)
        point = next(point for point in data["points"]
                     if point["rate_rps"] == 800 and point["block"] == 0
                     and point["arm"] == "native")
        self.assertTrue(point["conditional_p99"])
        self.assertEqual(point["completion_coverage"], .75)
        group = next(group for group in data["groups"]
                     if group["rate_rps"] == 800 and group["arm"] == "native")
        self.assertTrue(group["any_conditional_p99"])
        self.assertEqual(group["minimum_completion_coverage"], .75)

    def test_render_vector_and_raster_without_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="gpreempt-lc-knee-plot-") as temporary:
            prefix = Path(temporary) / "synthetic-not-results"
            outputs = plot.render(fixture(), prefix)
            self.assertEqual([path.suffix for path in outputs], [".pdf", ".png"])
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))
            with self.assertRaises(FileExistsError):
                plot.render(fixture(), prefix)


if __name__ == "__main__":
    unittest.main()

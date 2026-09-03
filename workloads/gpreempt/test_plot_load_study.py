"""Synthetic plot-contract tests; these values are never experimental evidence."""
import copy
import tempfile
import unittest
from pathlib import Path

import plot_load_study as plot


def fixture():
    audit = {"complete": True, "formal_eligible": True, "rejected_cells": [],
             "incomplete_cells": [], "scenarios": {}}
    for scenario in plot.SCENARIOS:
        points = [{"block": block, "arm": arm, "metrics": {
            "vgg_rt": {"p99_response_us": 1000 + block, "completion_coverage": 1.0,
                       "p99_is_conditional": False},
            "resnet152_be": {"goodput_rps": 100 + block}}}
            for block in range(5) for arm in plot.ARMS]
        audit["scenarios"][scenario] = {"complete": True, "valid_paired_blocks": 5,
                                         "per_cell_points": points}
    return audit


class PlotTests(unittest.TestCase):
    def test_full_inventory(self):
        self.assertEqual(len(plot.plot_points(fixture())), 45)

    def test_partial_duplicate_nonfinite_rejected(self):
        for mutation in (lambda a: a.update(complete=False),
                         lambda a: a.update(formal_eligible=False),
                         lambda a: a["scenarios"]["be100"]["per_cell_points"].pop(),
                         lambda a: a["scenarios"]["be100"]["per_cell_points"].append(
                             a["scenarios"]["be100"]["per_cell_points"][0]),
                         lambda a: a["scenarios"]["be100"]["per_cell_points"][0]["metrics"][
                             "vgg_rt"].update(p99_response_us=float("nan"))):
            audit = fixture()
            mutation(audit)
            with self.assertRaises(ValueError):
                plot.plot_points(audit)

    def test_conditional_latency_explicit(self):
        audit = fixture()
        lc = audit["scenarios"]["be200"]["per_cell_points"][0]["metrics"]["vgg_rt"]
        lc.update(completion_coverage=0.5)
        with self.assertRaises(ValueError):
            plot.plot_points(audit)
        lc.update(p99_is_conditional=True)
        self.assertEqual(sum(p["conditional"] for p in plot.plot_points(audit)), 1)

    def test_render_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="gpreempt-plot-test-") as directory:
            prefix = Path(directory) / "synthetic-not-results"
            outputs = plot.render(copy.deepcopy(fixture()), prefix)
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))
            with self.assertRaises(FileExistsError):
                plot.render(fixture(), prefix)


if __name__ == "__main__":
    unittest.main()

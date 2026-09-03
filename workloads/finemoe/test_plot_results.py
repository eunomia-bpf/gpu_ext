"""Plot projection tests, not GPU or correctness experiments."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import plot_results as plot


class PlotTests(unittest.TestCase):
    def test_real_twenty_cells_and_byte_conservation(self):
        points = plot.load_points(plot.HERE / "raw/full-v1")
        self.assertEqual(len(points), 20)
        self.assertEqual({(p["block"], p["arm"]) for p in points},
                         {(block, arm) for block in range(5) for arm in plot.ARMS})
        for point in points:
            self.assertEqual(sum(point[f"{kind}_bytes"] for kind in plot.CATEGORIES),
                             point["prefetch_bytes"])
            if point["arm"] == "demand-only":
                self.assertEqual(point["prefetch_bytes"], 0)

    def test_incomplete_and_duplicate_audit_rejected(self):
        audit = json.loads((plot.HERE / "raw/full-v1/independent-analysis.json").read_text())
        incomplete = {**audit, "complete": False}
        duplicate = copy.deepcopy(audit)
        duplicate["cells"][-1] = duplicate["cells"][0]
        for changed in (incomplete, duplicate):
            with patch.object(Path, "read_text", return_value=json.dumps(changed)):
                with self.assertRaises(ValueError):
                    plot.load_points(Path("synthetic-input-not-evidence"))

    def test_raw_arithmetic_mismatch_rejected(self):
        original = plot.analysis.reconstruct

        def changed(worker):
            metrics, requests = original(worker)
            metrics["tokens_per_second"] += 1
            return metrics, requests

        with patch.object(plot.analysis, "reconstruct", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "raw/audit metric mismatch"):
                plot.load_points(plot.HERE / "raw/full-v1")


if __name__ == "__main__":
    unittest.main()

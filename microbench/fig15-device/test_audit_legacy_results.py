#!/usr/bin/env python3
"""CPU-only tests for the legacy Fig. 15 aggregate audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import audit_legacy_results as audit


class LegacyArithmeticTests(unittest.TestCase):
    def test_each_series_uses_its_own_baseline(self) -> None:
        old = {("Baseline (tiny)", "tiny"): 10.0}
        new = {("Baseline (tiny)", "tiny"): 20.0}
        for operation in audit.PANEL_A_OPERATIONS:
            old[(f"{operation} (tiny)", "tiny")] = 14.0
            new[(f"{operation} (tiny)", "tiny")] = 22.0
        item = audit.compare_panel_a(old, new)[0]
        self.assertEqual(item.old_overhead_us, 4.0)
        self.assertEqual(item.new_overhead_us, 2.0)
        self.assertEqual(item.corrected_reduction_pct, 50.0)
        self.assertEqual(item.plotted_old_overhead_us, -6.0)

    def test_duplicate_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.md"
            path.write_text(
                "| Test Name | Workload | Avg Time (us) | x | y |\n"
                "| A | tiny | 1.0 | - | - |\n"
                "| A | tiny | 2.0 | - | - |\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit.parse_markdown(path)


class RetainedAggregateReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old = audit.parse_markdown(audit.DEFAULT_OLD)
        cls.new = audit.parse_markdown(audit.DEFAULT_NEW)
        cls.comparisons = {
            item.operation: item
            for item in audit.compare_panel_a(cls.old, cls.new)
        }

    def test_retained_baselines_and_all_ten_operations_replay(self) -> None:
        self.assertEqual(self.old[("Baseline (tiny)", "tiny")], 5.23)
        self.assertEqual(self.new[("Baseline (tiny)", "tiny")], 5.15)
        self.assertEqual(set(self.comparisons), set(audit.PANEL_A_OPERATIONS))

    def test_entry_exit_disproves_uniform_sixty_to_eighty_percent_text(self) -> None:
        item = self.comparisons["Entry+Exit"]
        self.assertAlmostEqual(item.old_overhead_us, 1.02, places=8)
        self.assertAlmostEqual(item.new_overhead_us, 0.91, places=8)
        self.assertAlmostEqual(item.corrected_reduction_pct, 10.7843137, places=6)
        self.assertLess(item.corrected_reduction_pct, 60.0)

    def test_available_map_ratios_are_not_a_single_six_thousand_factor(self) -> None:
        ratios = audit.map_absolute_ratios(self.new)
        self.assertAlmostEqual(ratios["update"], 33886.86 / 6.95, places=8)
        self.assertAlmostEqual(ratios["lookup"], 33738.82 / 5.75, places=8)
        self.assertNotAlmostEqual(ratios["update"], ratios["lookup"], places=1)


if __name__ == "__main__":
    unittest.main()

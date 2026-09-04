import importlib.util
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("plot_results", HERE / "plot_results.py")
plotter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plotter)


class ResultPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = plotter.load_pairs(HERE / "raw/full-575-01/result.json")

    def test_complete_paired_matrix(self) -> None:
        self.assertEqual(len(self.pairs), 10 * 9 * 3)
        for cell in range(9):
            for arm in ("noop", "counter"):
                self.assertEqual(plotter.paired_us(self.pairs, cell, arm).shape, (10,))

    def test_bootstrap_is_deterministic_and_contains_median(self) -> None:
        values = plotter.paired_us(self.pairs, 8, "counter")
        first = plotter.median_interval(values)
        second = plotter.median_interval(values)
        self.assertEqual(first, second)
        self.assertLessEqual(first[1], first[0])
        self.assertLessEqual(first[0], first[2])

    def test_active_counter_cost_grows_while_return_only_stays_bounded(self) -> None:
        counter_start = np.median(plotter.paired_us(self.pairs, 5, "counter"))
        counter_end = np.median(plotter.paired_us(self.pairs, 8, "counter"))
        noop = [np.median(plotter.paired_us(self.pairs, cell, "noop"))
                for cell in range(4, 9)]
        self.assertGreater(counter_end, 7 * counter_start)
        self.assertLess(max(noop), 3.0)


if __name__ == "__main__":
    unittest.main()

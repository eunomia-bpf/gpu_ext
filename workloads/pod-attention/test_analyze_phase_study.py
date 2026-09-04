"""CPU-only tests for POD phase analysis; no CUDA process is launched."""
import math
import unittest

import analyze_phase_study as analysis
import run_phase_study as phase


class PhaseAnalysisTests(unittest.TestCase):
    def test_paired_ratio_uses_all_five_matched_blocks(self):
        indices = analysis.bootstrap_indices()
        result = analysis.paired_ratio([2, 4, 6, 8, 10], [1, 2, 3, 4, 5], indices)
        self.assertEqual(result['blocks'], 5)
        self.assertEqual(result['block_ratios'], [2] * 5)
        self.assertAlmostEqual(result['geometric_mean_ratio'], 2)
        self.assertEqual(result['confidence_interval_95'], [2, 2])

    def test_paired_ratio_rejects_missing_and_invalid_values(self):
        indices = analysis.bootstrap_indices()
        for numerator, denominator in (([1] * 4, [1] * 4),
                                       ([1, 1, 0, 1, 1], [1] * 5),
                                       ([1, 1, math.inf, 1, 1], [1] * 5)):
            with self.subTest(numerator=numerator), self.assertRaises(ValueError):
                analysis.paired_ratio(numerator, denominator, indices)

    def test_phase_inventory_has_only_positive_cross_arm_durations(self):
        self.assertNotIn('loader_ready_ns', analysis.COMPARABLE_PHASES)
        self.assertIn('pre_python_main_ns', analysis.COMPARABLE_PHASES)
        self.assertIn('steady_samples_ns', analysis.COMPARABLE_PHASES)
        self.assertIn('whole_cell_ns', analysis.COMPARABLE_PHASES)
        self.assertEqual({left for _, left, _ in analysis.COMPARISONS},
                         {'pod_bpf', 'pod_cuda'})
        self.assertEqual(set(phase.ARMS), {'pod_inline', 'pod_cuda', 'pod_bpf'})

    def test_percentile_is_linear_interpolated(self):
        self.assertEqual(analysis.percentile([0, 10], .25), 2.5)
        self.assertEqual(analysis.percentile([0, 10], .5), 5)
        self.assertEqual(analysis.percentile([0, 10], .75), 7.5)


if __name__ == '__main__':
    unittest.main()

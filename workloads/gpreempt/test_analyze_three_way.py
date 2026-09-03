import unittest

import analyze_three_way as analysis


def blocks(count=5):
    result = []
    for i in range(count):
        cells = {}
        for arm, factor in zip(analysis.run.ARMS, (1., .5, .25)):
            cells[arm] = {role: {"p99_latency_us": factor * (1000 + i),
                "throughput_rps": 100., "completed_requests": 6000}
                for role in analysis.run.TASKS}
        result.append({"block": i, "cells": cells})
    return result


class AnalysisTests(unittest.TestCase):
    def test_known_paired_ratios_and_no_posthoc_equivalence_claim(self):
        result = analysis.summarize_blocks(blocks())
        key = "bpf_gpreempt/original_gpreempt:vgg_rt:p99_latency_us"
        self.assertTrue(result["complete"])
        self.assertFalse(result["equivalence_claimed"])
        self.assertEqual(result["paired"][key]["geometric_ratio"], .5)
        self.assertEqual(result["paired"][key]["paired_block_bootstrap_ci95"], [.5, .5])

    def test_one_block_is_preliminary_without_ci(self):
        result = analysis.summarize_blocks(blocks(1))
        self.assertFalse(result["complete"])
        for pair in result["paired"].values():
            self.assertIsNone(pair["paired_block_bootstrap_ci95"])

    def test_duplicate_or_incomplete_block_rejected(self):
        rows = blocks()
        with self.assertRaises(ValueError):
            analysis.summarize_blocks(rows + rows[:1])
        rows[0]["cells"].pop("native")
        with self.assertRaises(ValueError):
            analysis.summarize_blocks(rows)

    def test_invalid_ratio_rejected(self):
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                analysis.estimate_ratios([value])


if __name__ == "__main__":
    unittest.main()

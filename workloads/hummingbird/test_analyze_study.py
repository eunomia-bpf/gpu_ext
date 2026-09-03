"""Small synthetic CPU fixtures only; no performance evidence or GPU calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_study as audit
import test_study as fixtures
import test_load_study as gp_fixtures


def parse(values, slo=2_000_000):
    cfg, report, checks, loads = values
    return audit.parse_client(gp_fixtures.log_fixture(report, checks, loads), cfg, slo)


def cell(block=0, arm='idle_c', p99=100, goodput=100, coverage=1, slo=.98):
    return {'block': block, 'scenario': 'periodic', 'arm': arm,
            'metrics': {audit.LC: {'response_p99_ns': p99, 'completion_coverage': coverage,
                                   'conditional_p99': coverage < 1, 'slo_attainment': slo},
                        audit.BE: {'goodput_rps': goodput}},
            'engagement': {'executor': {'input_small_launches': 1, 'output_small_launches': 1}}}


class AnalyzeStudyTests(unittest.TestCase):
    def test_independent_matrix_config_matches_frozen_runner(self):
        for mode in ('full', 'small-calibration'):
            self.assertEqual(audit.expected_orders(mode), fixtures.run.orders(mode))
        for scenario in audit.SCENARIOS:
            for seconds in (10, 60):
                self.assertEqual(audit.expected_config(scenario, seconds), fixtures.run.config(scenario, seconds))
        with self.assertRaises(ValueError):
            audit.expected_orders('preflight')

    def test_independent_measurement_matches_real_format_and_preserves_backlog(self):
        for scenario in audit.SCENARIOS:
            values = fixtures.fixture(scenario)
            actual = parse(values)['metrics']
            self.assertEqual(actual, fixtures.measure(values)['metrics'])
            lc = actual[audit.LC]
            self.assertEqual(lc['started'], 150)
            self.assertEqual(lc['never_started'], lc['offered'] - 150)
            self.assertEqual(lc['slo_attainment'], lc['slo_met'] / lc['offered'])
            self.assertTrue(lc['conditional_p99'])

    def test_exact_deadline_is_not_goodput_or_slo_but_tail_stays_in_p99(self):
        values = fixtures.fixture(count=1000)
        values[-1][0]['requests'][-1][-1] = values[1]['loadStudyEndNs']
        lc = parse(values, 1_000_000_000)['metrics'][audit.LC]
        self.assertEqual((lc['completed_in_window'], lc['completed_after_window'], lc['slo_met']), (999, 1, 999))
        self.assertEqual(lc['completion_coverage'], .999)
        self.assertFalse(lc['conditional_p99'])
        self.assertEqual(lc['never_started'], 0)

    def test_empty_work_is_explicit_not_dropped(self):
        metrics = parse(fixtures.fixture(count=0))['metrics']
        self.assertEqual(metrics[audit.LC]['slo_attainment'], 0)
        self.assertEqual(metrics[audit.BE]['goodput_rps'], 0)
        self.assertIsNone(metrics[audit.LC]['response_p99_ns'])
        self.assertIsNone(metrics[audit.BE]['offered'])
        self.assertIsNone(metrics[audit.BE]['slo_attainment'])

    def test_raw_corruption_cannot_be_hidden_by_summary(self):
        changes = [lambda v: v[-1][0]['requests'][2].__setitem__(0, 3),
                   lambda v: v[-1][0]['requests'][2].__setitem__(1, 0),
                   lambda v: v[-1][0]['requests'][2].__setitem__(3, 1),
                   lambda v: v[-1][0].update(started=True),
                   lambda v: v[-1][0].update(offered=1000.0),
                   lambda v: v[2][0].update(checked=150),
                   lambda v: v[2][0].update(max_absolute_error=float('nan')),
                   lambda v: v[1]['results'][0]['analyzers'][0].update(requestLatencyNs=[])]
        for change in changes:
            values = fixtures.fixture()
            change(values)
            with self.assertRaises(ValueError):
                parse(values)

    def test_zero_and_negative_ratio_values_keep_all_pairs(self):
        for first in (0, -1, None):
            estimate = audit.ratio_estimate([first, 2], [1, 1])
            self.assertIsNone(estimate['geometric_ratio'])
            self.assertEqual(len(estimate['block_ratios']), 2)
            self.assertEqual(estimate['numerators'][0], first)
        estimate = audit.ratio_estimate([1, 2], [0, 1])
        self.assertEqual(estimate['block_ratios'], [None, 2])

    def test_paired_ci_and_strict_classification(self):
        pairs = [(cell(b, goodput=120), cell(b, goodput=100)) for b in range(5)]
        with patch.object(audit, 'DRAWS', 100):
            result = audit.paired_comparison(pairs, True)
            self.assertEqual(result['classification'], 'win')
            self.assertEqual(result['be_goodput_ratio']['paired_block_bootstrap_ci95'], [1.2, 1.2])
            self.assertEqual(result['slo_attainment_difference_pp']['paired_block_bootstrap_ci95_pp'], [0, 0])
            self.assertEqual(audit.paired_comparison(pairs, False)['classification'], 'incomplete_campaign')
            pairs[0][0]['metrics'][audit.LC]['completion_coverage'] = .99
            self.assertEqual(audit.paired_comparison(pairs, True)['classification'], 'incomplete_lc_coverage')
            for a, _ in pairs:
                a['metrics'][audit.LC].update(completion_coverage=1, response_p99_ns=110)
            self.assertEqual(audit.paired_comparison(pairs, True)['classification'], 'throughput_protection_tradeoff')
            for a, _ in pairs:
                a['metrics'][audit.BE]['goodput_rps'] = 0
            self.assertEqual(audit.paired_comparison(pairs, True)['classification'], 'undefined_ratio')

    def test_small_selection_requires_individual_and_combined_safety(self):
        rows = [cell(b, v) for b in range(5) for v in audit.VARIANTS]
        with patch.object(audit, 'DRAWS', 100):
            self.assertEqual(audit.small_selection(rows, True)['selected'], 'both')
            for row in rows:
                if row['arm'] == 'input':
                    row['metrics'][audit.LC]['response_p99_ns'] = 103
            self.assertEqual(audit.small_selection(rows, True)['selected'], 'output')
            for row in rows:
                row['engagement']['executor']['output_small_launches'] = 0
            self.assertEqual(audit.small_selection(rows, True)['selected'], 'none')
            self.assertIsNone(audit.small_selection(rows[:-1], False)['selected'])

    def test_cell_rejects_result_metrics_changed_from_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary).resolve()
            directory = campaign / 'block-00/periodic/native'
            directory.mkdir(parents=True)
            values = fixtures.fixture(count=2)
            values[0]['time'] = values[1]['benchmarkTime(s)'] = 60
            values[1]['loadStudyEndNs'] = values[1]['loadStudyBeginNs'] + 60 * audit.NS
            for row in values[-1]:
                row['end_ns'] = values[1]['loadStudyEndNs']
                if row['offered'] is not None:
                    row['offered'] = 6000
            for row in values[1]['results']:
                row['analyzers'][0]['avgThroughput(req/s)'] = 2 / 60
            cfg, report, checks, loads = values
            log = gp_fixtures.log_fixture(report, checks, loads)
            result = {'status': 'passed', 'arm': 'native', 'returncode': 0,
                      'command': [str(audit.HERE / 'build/hummingbird_client'), str(directory / 'config.json'), '--mode', 'native'],
                      'environment': fixtures.run.gp.environment('original_gpreempt', Path('/sys/fs/bpf/unused-hummingbird')),
                      'metrics': parse(values)['metrics']}
            result['metrics'][audit.BE]['goodput_rps'] = -1
            for name, value in (('config.json', cfg), ('result.json', result), ('execution.json', {'status': 'executed', 'returncode': 0})):
                (directory / name).write_text(json.dumps(value))
            (directory / 'client.log').write_text(log)
            with self.assertRaisesRegex(ValueError, 'raw metrics differ'):
                audit.audit_cell(directory, {'block': 0, 'scenario': 'periodic', 'arm': 'native'},
                                 {'mode': 'full', 'slo_ns': 2_000_000}, campaign)

    def test_full_completeness_rejection_overlap_and_partial_are_distinct(self):
        # Campaign mechanics use tiny synthetic audit rows, not fabricated GPU files.
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            sequence = audit.expected_orders('full')
            (campaign / 'run-order.json').write_text(json.dumps({'mode': 'full', 'seed': audit.SEED,
                'orders': sequence, 'slo_ns': 100}))
            (campaign / 'profile-frozen.json').write_text('{}')
            for spec in sequence:
                directory = campaign / f'block-{spec["block"]:02d}' / spec['scenario'] / spec['arm']
                directory.mkdir(parents=True)
                (directory / 'result.json').write_text('{}')
            def closed(directory, spec, plan, root):
                index = sequence.index(spec)
                row = cell(spec['block'], spec['arm'])
                row.update(spec, begin_ns=index * 100 + 1, end_ns=index * 100 + 50,
                           latest_verified_ns=index * 100 + 51, runtime_inventory={'binary': {'bytes': 1, 'mtime_ns': 1}})
                return row
            with patch.object(audit, 'audit_cell', side_effect=closed), patch.object(audit, 'DRAWS', 100):
                result = audit.analyze(campaign)
                self.assertTrue(result['formal_complete'])
                self.assertEqual(result['accepted_cells'], 50)
                self.assertEqual(result['scenarios']['burstgpt']['complete_blocks'], list(range(5)))
                last = sequence[-1]
                (campaign / f'block-{last["block"]:02d}' / last['scenario'] / last['arm'] / 'result.json').unlink()
                result = audit.analyze(campaign)
                self.assertFalse(result['formal_complete'])
                self.assertEqual(len(result['pending']), 1)
                self.assertEqual(result['scenarios']['periodic']['paired']['idle_bpf/idle_c']['classification'], 'incomplete_campaign')
            def overlap(*args):
                row = closed(*args)
                row['begin_ns'] = 1
                return row
            with patch.object(audit, 'audit_cell', side_effect=overlap), patch.object(audit, 'DRAWS', 100):
                result = audit.analyze(campaign)
                self.assertEqual(result['accepted_cells'], 1)
                self.assertEqual(len(result['rejected']), 48)


if __name__ == '__main__':
    unittest.main()

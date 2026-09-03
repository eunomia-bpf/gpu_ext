"""Synthetic CPU checks only; no GPU calls or performance evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / 'build/src'))
import run_study as run
import analyze_study as audit
import test_study as original_fixtures
import test_load_study as gp_fixtures


def engagement_fixture(arm='idle_bpf', bound=2, peak=None):
    peak = bound if peak is None else peak
    profile = {'kernels': [{'name': 'kernel', 'grid': [2, 1, 1]}],
               'small_input_enabled': False, 'small_output_enabled': False}
    setup = {'mode': arm, 'graph': False, 'configured_lp_inflight_bound': bound,
             'small_input_enabled': False, 'small_output_enabled': False}
    executor = {'requests_accepted': 1, 'requests_completed': 1, 'decisions': 10,
                'jit_decisions': 10 if arm == 'idle_bpf' else 0,
                'configured_lp_inflight_bound': bound, 'max_lp_inflight': peak,
                'completion_fence': 'event-query-before-next-launch' if bound == 1 else 'bounded-kernel-tick-event-ring',
                'split_launches': 2, 'whole_launches': 0, 'small_launches': 0, 'large_launches': 2,
                'lp_events_issued': 2, 'lp_events_retired': 2, 'lp_outstanding_at_report': 0,
                'lp_overlap_launches': 1 if peak == 2 else 0,
                'lp_bound_semantics': 'host-issued-unretired-events-not-device-queue-occupancy',
                'ctas_submitted': 2, 'nop_copies': 0,
                'input_small_launches': 0, 'output_small_launches': 0}
    hp = {'hp_enqueues': 1, 'hp_completions': 1, 'input_bubbles': 0, 'output_bubbles': 0}
    rows = [('HUMMINGBIRD_SETUP', setup), ('HUMMINGBIRD_EXECUTOR', executor),
            ('HUMMINGBIRD_HP_EVENTS', hp)]
    for role, task in enumerate(run.base.TASKS):
        rows += [('HUMMINGBIRD_CLEANUP', {'task': task, 'complete': True}),
                 ('HUMMINGBIRD_CONTEXT', {'role': role, 'hclient': role + 1, 'hobject': role + 1,
                  'timeslice_us': 1_000_000, 'stream_priority': 0,
                  'owned_query_ok': True, 'timeslice_set_ok': True})]
    measured = {'metrics': {task: {'started': 1} for task in run.base.TASKS}}
    return rows, measured, profile


def check_fixture(fixture, arm='idle_bpf', bound=2):
    rows, measured, profile = fixture
    log = '\n'.join(prefix + ' ' + json.dumps(value) for prefix, value in rows)
    with patch.object(run.base, 'check_engagement'):
        return run.engagement(log, arm, measured, profile, bound)


def closed_cell(spec, index):
    return {**spec, 'metrics': {
        audit.LC: {'response_p99_ns': 100, 'completion_coverage': 1,
                   'conditional_p99': False, 'slo_attainment': .98},
        audit.BE: {'goodput_rps': 100}},
        'engagement': {'executor': {'max_lp_inflight': 2 if spec['arm'].endswith('_d2') else 1}},
        'begin_ns': index * 100 + 1, 'end_ns': index * 100 + 50,
        'latest_verified_ns': index * 100 + 51,
        'runtime_inventory': {'synthetic-only': {'bytes': 1, 'mtime_ns': 1}}}


class PipelineStudyTests(unittest.TestCase):
    def test_fixed_matrix_and_independent_config(self):
        for mode, count in (('preflight', 8), ('full', 40)):
            expected = run.orders(mode)
            self.assertEqual(expected, audit.expected_orders(mode))
            self.assertEqual(len(expected), count)
            self.assertEqual(len({tuple(s.values()) for s in expected}), count)
        for scenario in run.SCENARIOS:
            for seconds in (10, 60):
                self.assertEqual(run.config(scenario, seconds), audit.expected_config(scenario, seconds))
        for mode in ('profile', 'small-calibration', 'smoke'):
            with self.assertRaises(ValueError):
                run.orders(mode)
        for arm in ('idle_bpf', 'idle_c_d3', 'native_d2'):
            with self.assertRaises(ValueError):
                run.arm_settings(arm)

    def test_private_paths_not_old_client(self):
        paths = run.runtime_paths()
        self.assertIn(HERE / 'build/hummingbird_client', paths)
        self.assertIn(HERE / 'build/idle_policy.bin', paths)
        self.assertIn(HERE / 'build/src/run_study.py', paths)
        self.assertNotIn(HERE.parent / 'build/hummingbird_client', paths)
        self.assertEqual(len(paths), len(set(paths)))

    def test_native_and_actual_jit_evidence_all_bounds(self):
        for arm in ('idle_c', 'idle_bpf'):
            for bound in (1, 2):
                value = check_fixture(engagement_fixture(arm, bound), arm, bound)
                self.assertEqual(value['executor']['max_lp_inflight'], bound)

    def test_bound_two_not_exercised_is_retained_not_falsely_engaged(self):
        value = check_fixture(engagement_fixture(peak=1))
        self.assertEqual(value['executor']['max_lp_inflight'], 1)
        self.assertEqual(value['executor']['lp_overlap_launches'], 0)

    def test_event_bound_drain_cta_and_jit_corruption_rejected(self):
        changes = [lambda e: e.update(lp_events_retired=1),
                   lambda e: e.pop('lp_events_retired'),
                   lambda e: e.update(lp_events_issued=True),
                   lambda e: e.update(lp_outstanding_at_report=1),
                   lambda e: e.update(max_lp_inflight=3),
                   lambda e: e.update(configured_lp_inflight_bound=True),
                   lambda e: e.update(lp_overlap_launches=0),
                   lambda e: e.update(lp_overlap_launches=3),
                   lambda e: e.update(jit_decisions=0),
                   lambda e: e.update(ctas_submitted=1),
                   lambda e: e.update(completion_fence='event-query-before-next-launch'),
                   lambda e: e.update(lp_bound_semantics='device-queue-depth')]
        for change in changes:
            value = engagement_fixture()
            change(value[0][1][1])
            with self.assertRaises((ValueError, KeyError)):
                check_fixture(value)
        value = engagement_fixture(bound=1)
        value[0][1][1]['lp_overlap_launches'] = 1
        with self.assertRaises(ValueError):
            check_fixture(value, bound=1)

    def test_original_request_arithmetic_and_corruption_checks_preserved(self):
        for scenario in run.SCENARIOS:
            cfg, report, checks, loads = original_fixtures.fixture(scenario)
            log = gp_fixtures.log_fixture(report, checks, loads)
            expected = run.measurement(log, cfg, run.SLO_NS)['metrics']
            self.assertEqual(audit.parse_client(log, cfg, run.SLO_NS)['metrics'], expected)
            self.assertTrue(expected[audit.LC]['conditional_p99'])
            self.assertEqual(expected[audit.LC]['slo_attainment'],
                             expected[audit.LC]['slo_met'] / expected[audit.LC]['offered'])
            checks[0]['timed_checked'] -= 1
            with self.assertRaises(ValueError):
                audit.parse_client(gp_fixtures.log_fixture(report, checks, loads), cfg, run.SLO_NS)

    def test_preflight_admission_requires_all_eight_runtime_profile_and_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'profile-frozen.json').write_text('{}')
            rows = [closed_cell(s, i) for i, s in enumerate(run.orders('preflight'))]
            valid = {'complete': True, 'mode': 'preflight', 'cells': rows, 'runtime_inventory': {'test': 1}}
            with patch.object(audit, 'analyze', return_value=valid), patch.object(run, 'runtime_inventory', return_value={'test': 1}):
                run.validate_preflight(path, {})
                with self.assertRaises(RuntimeError):
                    run.validate_preflight(path, {'changed': 1})
            changes = [lambda r: r.update(complete=False), lambda r: r.update(mode='full'),
                       lambda r: r['cells'].pop(), lambda r: r['cells'].append(copy.deepcopy(r['cells'][0])),
                       lambda r: r.update(runtime_inventory={'changed': 1}),
                       lambda r: next(c for c in r['cells'] if c['arm'].endswith('_d2'))['engagement']['executor'].update(max_lp_inflight=1)]
            for change in changes:
                reviewed = copy.deepcopy(valid)
                change(reviewed)
                with patch.object(audit, 'analyze', return_value=reviewed), patch.object(run, 'runtime_inventory', return_value={'test': 1}):
                    with self.assertRaises(RuntimeError):
                        run.validate_preflight(path, {})

    def test_campaign_completeness_preflight_link_and_unexercised_status(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            preflight, full = parent / 'preflight', parent / 'full'
            def write_campaign(path, mode):
                path.mkdir()
                order = run.orders(mode)
                (path / 'run-order.json').write_text(json.dumps({'mode': mode, 'seed': run.SEED,
                    'orders': order, 'slo_ns': run.SLO_NS, 'preflight': str(preflight) if mode == 'full' else None}))
                (path / 'profile-frozen.json').write_text(run.FROZEN_PROFILE.read_text())
                for spec in order:
                    directory = path / f'block-{spec["block"]:02d}' / spec['scenario'] / spec['arm']
                    directory.mkdir(parents=True)
                    (directory / 'result.json').write_text('{}')
            write_campaign(preflight, 'preflight')
            write_campaign(full, 'full')
            def closed(directory, spec, plan, campaign):
                return closed_cell(spec, run.orders(plan['mode']).index(spec))
            with patch.object(audit, 'audit_cell', side_effect=closed), patch.object(audit, 'DRAWS', 20):
                short = audit.analyze(preflight)
                self.assertTrue(short['complete'])
                self.assertFalse(short['formal_complete'])
                self.assertFalse(short['causal_interpretation_ready'])
                result = audit.analyze(full)
                self.assertTrue(result['formal_complete'])
                self.assertTrue(result['causal_interpretation_ready'])
                self.assertEqual(result['accepted_cells'], 40)
                self.assertEqual(result['scenarios']['burstgpt']['complete_blocks'], list(range(5)))
                spec = run.orders('full')[-1]
                (full / f'block-{spec["block"]:02d}' / spec['scenario'] / spec['arm'] / 'result.json').unlink()
                result = audit.analyze(full)
                self.assertFalse(result['formal_complete'])
                self.assertFalse(result['pipeline_exercised'])
                self.assertEqual(len(result['pending']), 1)
            def unexercised(*args):
                row = closed(*args)
                row['engagement']['executor']['max_lp_inflight'] = 1
                return row
            with patch.object(audit, 'audit_cell', side_effect=unexercised), patch.object(audit, 'DRAWS', 20):
                self.assertFalse(audit.analyze(preflight)['pipeline_exercised'])
                with self.assertRaisesRegex(ValueError, 'did not exercise'):
                    audit.analyze(full)

    def test_nonpositive_pairs_are_not_dropped(self):
        result = audit.ratio_estimate([0, 2], [1, 1])
        self.assertIsNone(result['geometric_ratio'])
        self.assertEqual(result['block_ratios'], [0, 2])


if __name__ == '__main__':
    unittest.main()

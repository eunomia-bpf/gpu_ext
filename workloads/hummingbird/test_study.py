"""CPU fixtures only: original DISB report format, no GPU or performance data."""
import copy
import unittest
import run_study as run
import test_load_study as original_tests


def fixture(scenario='periodic', count=150):
    seconds = 60 if scenario == 'burstgpt' else 10
    _, report, checks, loads = original_tests.fixture('be_continuous', seconds, count)
    cfg = run.config(scenario, seconds)
    if scenario == 'burstgpt':
        offsets = cfg['tasks'][0]['load']['offsets_ns']
        row = loads[0]
        row.update(mode='trace_fifo', interval_ns=0, offered=len(offsets))
        previous = row['begin_ns']
        for index, request in enumerate(row['requests']):
            arrival = row['begin_ns'] + offsets[index]
            start = max(arrival, previous)
            request[:] = [index, arrival, start, start + 1_000_000]
            previous = request[-1]
    return cfg, report, checks, loads


def measure(values, slo_ns=2_000_000):
    cfg, report, checks, loads = values
    return run.measurement(original_tests.log_fixture(report, checks, loads), cfg, slo_ns)


class StudyTests(unittest.TestCase):
    def test_frozen_matrix_and_trace_prefix(self):
        order = run.orders('full')
        self.assertEqual(len(order), 50)
        self.assertEqual(order, run.orders('full'))
        self.assertEqual(len({tuple(r[k] for k in ('block', 'scenario', 'arm')) for r in order}), 50)
        self.assertEqual(len(run.orders('preflight')), 10)
        self.assertEqual(len(run.orders('small-calibration')), 20)
        full = run.config('burstgpt', 60)['tasks'][0]['load']['offsets_ns']
        short = run.config('burstgpt', 10)['tasks'][0]['load']['offsets_ns']
        self.assertEqual(short, [t for t in full if t < 10 * run.NS])
        self.assertEqual(len(full), 6000)
        self.assertFalse(run.config('periodic', 60)['tasks'][0]['client']['use_cuda_graph'])
        self.assertEqual(len(run.config('periodic', 60, True)['tasks']), 1)

    def test_all_offered_slo_denominator_preserves_backlog(self):
        for scenario in run.SCENARIOS:
            lc = measure(fixture(scenario))['metrics']['vgg_rt']
            self.assertEqual(lc['started'], 150)
            self.assertGreater(lc['offered'], 150)
            self.assertEqual(lc['never_started'], lc['offered'] - 150)
            self.assertTrue(lc['conditional_p99'])
            self.assertEqual(lc['slo_attainment'], lc['slo_met'] / lc['offered'])
            self.assertLess(lc['slo_attainment'], 1)

    def test_deadline_tail_is_verified_but_not_goodput_or_slo(self):
        values = fixture(count=1000)
        values[-1][0]['requests'][-1][-1] = values[1]['loadStudyEndNs']
        lc = measure(values, 1_000_000_000)['metrics']['vgg_rt']
        self.assertEqual((lc['completed_in_window'], lc['completed_after_window'], lc['slo_met']), (999, 1, 999))
        self.assertEqual(lc['slo_attainment'], .999)

    def test_continuous_and_empty_results_are_not_hidden(self):
        values = measure(fixture(count=0))['metrics']
        self.assertEqual(values['vgg_rt']['slo_attainment'], 0)
        self.assertIsNone(values['vgg_rt']['response_p99_ns'])
        self.assertIsNone(values['resnet152_be']['offered'])
        self.assertIsNone(values['resnet152_be']['slo_attainment'])

    def test_raw_arrival_and_numerical_corruption_fail(self):
        changes = [lambda v: v[-1][0]['requests'][3].__setitem__(1, 1),
                   lambda v: v[-1][0]['requests'][3].__setitem__(0, 4),
                   lambda v: v[-1][0]['requests'][3].__setitem__(3, 1),
                   lambda v: v[-1][0].update(offered=1000.0),
                   lambda v: v[2][0].update(timed_checked=149),
                   lambda v: v[2][0].update(max_absolute_error=float('nan'))]
        for scenario in run.SCENARIOS:
            for change in changes:
                values = fixture(scenario)
                change(values)
                with self.assertRaises(ValueError):
                    measure(values)

    def test_isolated_source_report(self):
        values = fixture(count=1000)
        cfg, report, checks, loads = values
        cfg['tasks'] = cfg['tasks'][:1]
        report['results'] = report['results'][:1]
        result = measure((cfg, report, checks[:1], loads[:1]), None)
        self.assertEqual(set(result['metrics']), {'vgg_rt'})
        self.assertEqual(result['metrics']['vgg_rt']['completion_coverage'], 1)

    def test_small_pattern_selection_requires_engagement_and_all_pairs(self):
        results = []
        for block in range(5):
            for variant in ('none', 'input', 'output', 'both'):
                results.append({'block': block, 'variant': variant,
                                'metrics': {'vgg_rt': {'completion_coverage': 1, 'response_p99_ns': 100}},
                                'engagement': {'executor': {'input_small_launches': 1, 'output_small_launches': 1}}})
        self.assertEqual(run.choose_small_patterns(results)[0], 'both')
        one_unsafe = copy.deepcopy(results)
        for row in one_unsafe:
            if row['variant'] == 'input':
                row['metrics']['vgg_rt']['response_p99_ns'] = 103
        self.assertEqual(run.choose_small_patterns(one_unsafe)[0], 'output')
        for row in results:
            if row['variant'] == 'both':
                row['metrics']['vgg_rt']['response_p99_ns'] = 103
        self.assertEqual(run.choose_small_patterns(results)[0], 'input')
        for row in results:
            row['engagement']['executor']['input_small_launches'] = 0
        self.assertEqual(run.choose_small_patterns(results)[0], 'output')
        for row in results:
            row['metrics']['vgg_rt']['completion_coverage'] = .99
        self.assertEqual(run.choose_small_patterns(results)[0], 'none')
        with self.assertRaises(ValueError):
            run.choose_small_patterns(results + [copy.deepcopy(results[0])])
        with self.assertRaises(ValueError):
            run.choose_small_patterns(results[:-1])


if __name__ == '__main__':
    unittest.main()

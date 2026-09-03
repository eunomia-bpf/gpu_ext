"""Synthetic projection fixtures only; no raw campaign, rendering or GPU access."""
import copy
import unittest

import plot_results as plot


def fixture():
    cells = []
    for scenario in plot.SCENARIOS:
        for block in range(5):
            for label in plot.ARMS:
                arm, bound = label.rsplit('_d', 1)
                cells.append({'scenario': scenario, 'block': block, 'arm': label, 'actual_arm': arm,
                    'begin_ns': 1, 'end_ns': 60_000_000_001,
                    'engagement': {'executor': {'configured_lp_inflight_bound': int(bound),
                        'max_lp_inflight': int(bound), 'decisions': 10,
                        'jit_decisions': 10 if arm == 'idle_bpf' else 0}},
                    'metrics': {'vgg_rt': {'response_p99_ns': (block + 1) * 1_000_000,
                        'completion_coverage': 1, 'conditional_p99': False},
                        'resnet152_be': {'goodput_rps': block}}})
    return {'mode': 'full', 'complete': True, 'formal_complete': True,
            'pipeline_exercised': True, 'causal_interpretation_ready': True,
            'required_cells': 40, 'accepted_cells': 40, 'pending': [], 'rejected': [], 'unexpected': [],
            'statistics': {'paired_blocks': 5, 'draws': 10000, 'seed': 20260903},
            'scenarios': {s: {'complete_blocks': list(range(5))} for s in plot.SCENARIOS}, 'cells': cells}


class PlotTests(unittest.TestCase):
    def test_all_forty_points_units_zero_goodput_and_coverage_retained(self):
        audit = fixture()
        audit['cells'][0]['metrics']['vgg_rt'].update(completion_coverage=.5, conditional_p99=True)
        points = plot.plot_points(audit)
        self.assertEqual(len(points), 40)
        self.assertEqual({p['lc_p99_ms'] for p in points}, {1, 2, 3, 4, 5})
        self.assertEqual(sum(p['be_goodput_rps'] == 0 for p in points), 8)
        self.assertTrue(points[0]['lc_incomplete_coverage'])
        self.assertTrue(points[0]['lc_conditional_p99'])
        self.assertEqual(len({(p['scenario'], p['block'], p['arm']) for p in points}), 40)

    def test_partial_preflight_duplicate_and_unexercised_audits_rejected(self):
        changes = [lambda a: a.update(mode='preflight'), lambda a: a.update(formal_complete=False),
                   lambda a: a.update(pipeline_exercised=False), lambda a: a.update(causal_interpretation_ready=False),
                   lambda a: a.update(accepted_cells=39), lambda a: a['cells'].pop(),
                   lambda a: a['cells'].__setitem__(-1, copy.deepcopy(a['cells'][0])),
                   lambda a: a['cells'][0].update(block=False),
                   lambda a: a['pending'].append({'block': 4}),
                   lambda a: a['scenarios']['periodic'].update(complete_blocks=[0, 1, 2, 3])]
        for change in changes:
            audit = fixture()
            change(audit)
            with self.assertRaises(ValueError):
                plot.plot_points(audit)

    def test_invalid_metrics_short_window_and_false_policy_evidence_rejected(self):
        changes = [lambda c: c['metrics']['vgg_rt'].update(response_p99_ns=None),
                   lambda c: c['metrics']['vgg_rt'].update(response_p99_ns=float('nan')),
                   lambda c: c['metrics']['vgg_rt'].update(response_p99_ns=0),
                   lambda c: c['metrics']['resnet152_be'].update(goodput_rps=-1),
                   lambda c: c['metrics']['resnet152_be'].update(goodput_rps=True),
                   lambda c: c.update(end_ns=10_000_000_001),
                   lambda c: c['engagement']['executor'].update(max_lp_inflight=1),
                   lambda c: c['engagement']['executor'].update(jit_decisions=0)]
        for change in changes:
            audit = fixture()
            change(next(c for c in audit['cells'] if c['arm'] == 'idle_bpf_d2'))
            with self.assertRaises(ValueError):
                plot.plot_points(audit)


if __name__ == '__main__':
    unittest.main()

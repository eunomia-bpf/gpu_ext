"""Synthetic parser unit tests only; these records are not experiment data."""
import copy
import json
import unittest

from gpreempt_context_smoke_check import analyze


class CanaryParserTests(unittest.TestCase):
    def setUp(self):
        self.client = [dict(event='gpreempt_context_smoke', passed=True, policy='bpf',
                            validated_values=2048, negative_cases=17)]
        self.rpc = []
        for role in (0, 1):
            self.client.append(dict(event='role_context', pid=7, tid=10 + role, role=role,
                                    hclient=20, htsg=30 + role, cuda_context=40 + role,
                                    channels=3, kernel_begin_ns=1000))
            self.rpc.append(dict(event='gsp_timeslice_rpc', pid=7, tid=10 + role,
                                 hclient=20, hobject=30 + role, command=0xa06c0103,
                                 params_size=8, timeslice_us=1000000 if role == 0 else 1,
                                 source='kernel_open_rpc_completion', input_valid=1, wire_size=8,
                                 transport_status=0, gsp_status=0, gsp_status_valid=1, completed_ns=110))
        self.rpc.append(dict(event='gpreempt_rpc_observer_summary', observed=2, completed=2,
                             received=2, read_errors=0, ring_drops=0))
        self.bridge = ('gpreempt_bridge_stats: backend=ubpf-jit scopes=2 registered=2 ended=2 '
                       'errors=0 hint=1 block=1 release=1\n')
        self.policy = ('gpreempt_policy_stats: scope_enter=2 scope_leave=2 gr_init=2 timeslice_ok=2 '
                       'alloc_captured=2 registered=2 destroy=2 unknown_engine=0 setter_error=0 '
                       'alloc_error=0 register_error=0 bind_shadow_mismatch=0 map_error=0 '
                       'scope_error=0 bind_shadow_match=2\n')

    def run_case(self):
        return analyze('\n'.join(map(json.dumps, self.client)) + '\n' + self.bridge,
                       '\n'.join(map(json.dumps, self.rpc)), 'bpf', self.policy)

    def test_complete(self):
        result = self.run_case()
        self.assertTrue(result['firmware_timeslice_status_observed'])
        self.assertFalse(result['physical_quantum_measured'])
        self.assertFalse(result['interleave_tested'])

    def test_context_only_explicitly_lacks_rpc_evidence(self):
        result = analyze('\n'.join(map(json.dumps, self.client)) + '\n' + self.bridge,
                         None, 'bpf', self.policy)
        self.assertTrue(result['passed'])
        self.assertFalse(result['firmware_timeslice_status_observed'])
        self.assertIn('expected_timeslice_us', result['roles'][0])
        self.assertNotIn('timeslice_us', result['roles'][0])

    def test_failure_variants(self):
        for field, bad in [('gsp_status_valid', 0), ('input_valid', 0), ('gsp_status', 5),
                           ('transport_status', 1), ('source', 'host_shadow'),
                           ('params_size', 4), ('timeslice_us', 200), ('completed_ns', 1100),
                           ('hobject', 77), ('pid', 88)]:
            with self.subTest(field=field):
                original = self.rpc[0][field]
                self.rpc[0][field] = bad
                with self.assertRaises(ValueError):
                    self.run_case()
                self.rpc[0][field] = original

    def test_later_overwrite_rejected(self):
        late = copy.deepcopy(self.rpc[0])
        late.update(timeslice_us=3000, completed_ns=500)
        self.rpc.insert(2, late)
        self.rpc[-1].update(observed=3, completed=3, received=3)
        with self.assertRaises(ValueError):
            self.run_case()

    def test_lost_events_rejected(self):
        self.rpc[-1]['ring_drops'] = 1
        with self.assertRaises(ValueError):
            self.run_case()

    def test_no_policy_engagement_rejected(self):
        self.policy = self.policy.replace('registered=2', 'registered=0')
        with self.assertRaises(ValueError):
            self.run_case()

    def test_missing_negative_tests_rejected(self):
        self.client[0]['negative_cases'] = 16
        with self.assertRaises(ValueError):
            self.run_case()


if __name__ == '__main__':
    unittest.main()

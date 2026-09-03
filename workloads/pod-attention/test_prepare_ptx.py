"""CPU transport/extraction fixtures only, not attention/GPU execution."""
from pathlib import Path
import unittest
from unittest.mock import patch

from prepare_ptx import TUS, extract_one, transform_pruned

HEADER = '.version 8.8\n.target sm_120\n.address_size 64\n'


class PrepareTests(unittest.TestCase):
    def test_actual_adapter_preserves_fixture_entries_and_arguments(self):
        text = HEADER + '.func pod_device_selector(.param .b64 ctx, .param .b64 len);\n'
        text += '.visible .entry true_fused_tb_fwd_kernel_fixture() {\n'
        text += 'call.uni pod_device_selector, (ctx_argument, len_argument);\nret;\n}\n'
        output, proof = transform_pruned(text)
        self.assertEqual(proof['official_entries'], 1)
        self.assertEqual(proof['typed_calls'], 1)
        self.assertLess(proof['response_json_bytes'], proof['transport_capacity'])
        self.assertIn('call.uni pod_device_bpf_selector, (ctx_argument, len_argument);', output)

    def test_no_hook_or_wrong_representative_is_rejected(self):
        text = HEADER + '.visible .entry true_fused_tb_fwd_kernel_fixture() { ret; }\n'
        with self.assertRaises(ValueError):
            transform_pruned(text)
        with self.assertRaises(ValueError):
            transform_pruned(text, 'not_a_real_entry')

    def test_extraction_preserves_ptx_and_rejects_ambiguous_or_wrong_arch(self):
        module = HEADER + '.visible .entry kernel() { ret; }\n'
        with patch('prepare_ptx.subprocess.check_output', return_value='cuobjdump banner\n' + module):
            self.assertEqual(extract_one(Path('input.o'), Path('cuobjdump')), module)
        for text in (module + module, module.replace('sm_120', 'sm_100')):
            with patch('prepare_ptx.subprocess.check_output', return_value=text), self.assertRaises(ValueError):
                extract_one(Path('input.o'), Path('cuobjdump'))

    def test_complete_planned_official_tu_set(self):
        self.assertEqual(set(TUS), {
            'truefused_fwd_hdim128_fp16_causal_fo9_sm80',
            'truefused_fwd_hdim128_fp16_causal_split_fo9_sm80',
            'truefused_fwd_hdim128_fp16_causal_fo11_sm80',
            'truefused_fwd_hdim128_fp16_causal_split_fo11_sm80'})


if __name__ == '__main__':
    unittest.main()

import unittest
from ptx_partition import independent_entries, partition_ptx

HEADER = '.version 8.8\n.target sm_120\n.address_size 64\n'


class IndependenceTests(unittest.TestCase):
    def test_packets_keep_exact_entry_union_bodies_helpers_and_data(self):
        helper = '.func helper() { ret; }'
        entries = ['.visible .entry a() { call helper, (); ret; }',
                   '.visible .entry b() { call helper, (); ret; }',
                   '.visible .entry c() { call helper, (); ret; }']
        data = '.global .b32 unused;'
        source = HEADER + '\n'.join([data, helper, *entries]) + '\n'
        packets, proof = partition_ptx(source)
        self.assertEqual(len(packets), 2)
        self.assertEqual(sorted(sum(proof['packet_entry_names'], [])), ['a', 'b', 'c'])
        for entry in entries:
            self.assertEqual(sum(entry in packet for packet in packets), 1)
        for packet in packets:
            self.assertIn(helper, packet)
            self.assertIn(data, packet)
        self.assertEqual(partition_ptx(source), (packets, proof))

    def test_cta_local_shared_and_unused_global_are_allowed(self):
        text = HEADER + '''
.global .b32 unused;
.extern .shared .b8 scratch[];
.func helper() { ret; }
.visible .entry a() { call helper, (); st.shared.u32 [scratch], 1; ret; }
.visible .entry b() { call helper, (); ret; }
'''
        _, proof = independent_entries(text)
        self.assertEqual(proof['entry_names'], ['a', 'b'])
        self.assertEqual(proof['data_counts'], {'global': 1, 'shared': 1})
        self.assertEqual(proof['referenced_data_counts'], {'shared': 1})

    def test_even_read_only_global_reference_is_conservatively_rejected(self):
        for space in ('global', 'const'):
            with self.subTest(space=space), self.assertRaises(ValueError):
                independent_entries(HEADER + f'.{space} .b32 value;\n'
                    '.visible .entry a() { ld.global.u32 %r, [value]; ret; }\n')

    def test_address_taken_even_when_also_directly_called_is_rejected(self):
        with self.assertRaises(ValueError):
            independent_entries(HEADER + '.func helper() { ret; }\n'
                '.visible .entry a() { call helper, (); mov.u64 %r, helper; ret; }\n')

    def test_module_function_address_table_is_rejected(self):
        with self.assertRaises(ValueError):
            independent_entries(HEADER + '.global .u64 callback = helper;\n'
                '.func helper() { ret; }\n.visible .entry a() { ret; }\n')

    def test_entry_address_or_call_is_rejected(self):
        for use in ('mov.u64 %r, b;', 'call b, ();'):
            with self.subTest(use=use), self.assertRaises(ValueError):
                independent_entries(HEADER + '.visible .entry a() { ' + use + ' ret; }\n'
                    '.visible .entry b() { ret; }\n')

    def test_unknown_external_helper_or_call_form_is_rejected(self):
        for text in ('.extern .func unknown();\n.visible .entry a() { call unknown, (); ret; }',
                     '.func known() { ret; }\n.visible .entry a() { call.bogus known, (); ret; }',
                     '.func known() { ret; }\n.visible .entry a() { .reg .b64 known; call known, (); ret; }'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                independent_entries(HEADER + text)


if __name__ == '__main__':
    unittest.main()

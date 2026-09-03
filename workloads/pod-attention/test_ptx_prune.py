import unittest
from ptx_prune import prune

HEADER = '.version 8.8\n.target sm_120\n.address_size 64\n'


class PruneTests(unittest.TestCase):
    def test_transitive_calls_address_taken_and_data_roots(self):
        text = HEADER + '''
.global .u64 callback = from_data;
.extern .func external();
.func leaf() { ret; }
.func addressed() { ret; }
.func from_data() { ret; }
.func middle() { mov.u64 %r, addressed; call leaf, (); ret; }
.func unused() { ret; }
.visible .entry real_attention() { call middle, (); ret; }
'''
        output, inventory = prune(text)
        self.assertEqual(inventory['removed_function_names'], ['unused'])
        self.assertIn('.global .u64 callback = from_data;', output)
        self.assertIn('.visible .entry real_attention() { call middle, (); ret; }', output)

    def test_returns_prototypes_clone_names_and_nested_braces(self):
        text = HEADER + '''
.func (.param .b32 retval) work$3(.param .b64 x);
.visible .func (.param .b32 retval) work$3(.param .b64 x) { { ret; } }
.visible .entry kernel() { call.uni (%r), work$3, (%x); ret; }
'''
        output, inventory = prune(text)
        self.assertEqual(output, text)
        self.assertEqual(inventory['removed_function_names'], [])

    def test_comments_and_strings_do_not_break_body_parser(self):
        text = HEADER + '''
// .visible .entry not_a_kernel() { }
.func unused() { .pragma "} {"; ret; }
.visible .entry kernel() { /* } */ ret; }
'''
        output, inventory = prune(text)
        self.assertEqual(inventory['removed_function_names'], ['unused'])
        self.assertIn('// .visible .entry not_a_kernel() { }', output)

    def test_unreachable_indirect_can_be_removed(self):
        text = HEADER + '.func dead() { call %r, (); ret; }\n.visible .entry kernel() { ret; }\n'
        _, inventory = prune(text)
        self.assertEqual(inventory['removed_function_names'], ['dead'])

    def test_reachable_unknown_calls_fail_closed(self):
        for call in ('call %r, ();', 'call.uni (%out), %r, (%arg), proto;',
                     'call missing, ();', 'call.bogus target, ();'):
            with self.subTest(call=call), self.assertRaises(ValueError):
                prune(HEADER + '.visible .entry kernel() { ' + call + ' ret; }\n')

    def test_bare_register_shadowing_function_is_not_a_direct_call(self):
        with self.assertRaises(ValueError):
            prune(HEADER + '.func target() { ret; }\n'
                  '.visible .entry kernel() { .reg .b64 target; call target, (); ret; }\n')

    def test_unknown_linkage_and_truncated_body_rejected(self):
        for text in ('.strange .func hidden() { ret; }\n.visible .entry kernel() { ret; }',
                     '.visible .entry kernel() { ret;'):
            with self.assertRaises(ValueError):
                prune(HEADER + text)


if __name__ == '__main__':
    unittest.main()

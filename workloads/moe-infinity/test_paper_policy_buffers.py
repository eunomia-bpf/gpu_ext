import ctypes
import unittest

import numpy as np

from paper_policy_buffers import ENTRY_DTYPE, RankEntry, packed_entries, packed_select


class BufferTests(unittest.TestCase):
    def test_exact_abi_and_all_float_bits_with_strided_input(self):
        bits = np.array([0, 0x8000000000000000, 0x7ff0000000000000,
            0xfff0000000000000, 0x7ff80000000000ab, 0xfff80000000000cd,
            0x3ff0000000000000, 0x0000000000000001], dtype=np.uint64)
        backing = np.repeat(bits, 2).view(np.float64)
        scores = backing[::2]
        identities = [2**64 - 1 - i for i in range(len(bits))]
        entries = packed_entries(identities, scores)
        self.assertEqual(ENTRY_DTYPE.itemsize, 24)
        self.assertEqual(ctypes.sizeof(RankEntry), 24)
        for i, value in enumerate(bits):
            pointer = entries.ctypes.data_as(ctypes.POINTER(RankEntry))
            self.assertEqual(pointer[i].score_bits, int(value))
            self.assertEqual(pointer[i].identity, identities[i])
            self.assertEqual(pointer[i].ordinal, i)
            self.assertEqual(pointer[i].reserved, 0)

    def test_no_sort_or_filter_in_bridge(self):
        entries = packed_entries([7, 8, 9, 10], np.array([-1., 3., 2., 3.]))
        np.testing.assert_array_equal(entries["identity"], [7, 8, 9, 10])
        np.testing.assert_array_equal(entries["score_bits"].view(np.float64), [-1., 3., 2., 3.])

    def test_failure_is_not_silently_replaced_by_native_selection(self):
        with self.assertRaisesRegex(RuntimeError, "no fallback"):
            packed_select(lambda *args: -1, [1], np.array([1.]))

    def test_empty_and_malformed_input(self):
        self.assertEqual(len(packed_entries([], np.array([]))), 0)
        with self.assertRaises(ValueError):
            packed_entries([1], np.array([1., 2.]))
        with self.assertRaises(ValueError):
            packed_entries([1], np.array([[1.]]))


if __name__ == "__main__":
    unittest.main()

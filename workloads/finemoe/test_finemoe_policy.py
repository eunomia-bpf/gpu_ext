"""Independent Python binary64 oracle vs native hardware FP and real uBPF JIT."""
import ctypes as C
from pathlib import Path
import random
import struct
import unittest


HERE = Path(__file__).resolve().parent


def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value32(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def bits64(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


class Input(C.Structure):
    _fields_ = [('count', C.c_uint32), ('top_k', C.c_uint32),
                ('threshold_bits', C.c_uint32), ('reserved', C.c_uint32),
                ('probability_bits', C.c_uint32 * 60)]


class Output(C.Structure):
    _fields_ = [('mask', C.c_uint64), ('cumulative_bits', C.c_uint64),
                ('selected', C.c_uint32), ('status', C.c_uint32)]


class Context(C.Structure):
    _fields_ = [('input', Input), ('output', Output)]


def oracle(probabilities, threshold, top_k):
    if not any(probabilities):
        return 0, 0, 0
    ordered = sorted(range(len(probabilities)), key=lambda i: (-value32(probabilities[i]), i))
    total, mask, count = 0., 0, 0
    for i in ordered:
        total += value32(probabilities[i])
        mask |= 1 << i
        count += 1
        if count >= top_k and total >= value32(threshold):
            break
    return mask, bits64(total), count


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(str(HERE / 'build/libfinemoe_policy.so'))
        cls.lib.finemoe_select_native.argtypes = [C.POINTER(Context)]
        cls.lib.finemoe_jit_open.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t]
        cls.lib.finemoe_jit_open.restype = C.c_void_p
        cls.lib.finemoe_select_bpf.argtypes = [C.c_void_p, C.POINTER(Context)]
        cls.lib.finemoe_jit_calls.argtypes = [C.c_void_p]
        cls.lib.finemoe_jit_calls.restype = C.c_uint64
        cls.lib.finemoe_jit_close.argtypes = [C.c_void_p]
        error = C.create_string_buffer(1024)
        cls.handle = cls.lib.finemoe_jit_open(str(HERE / 'build/finemoe_policy.bin').encode(), error, len(error))
        if not cls.handle:
            raise RuntimeError(error.value.decode())

    @classmethod
    def tearDownClass(cls):
        cls.lib.finemoe_jit_close(cls.handle)

    def compare(self, probabilities, threshold, top_k):
        expected = oracle(probabilities, threshold, top_k)
        inp = Input(len(probabilities), top_k, threshold, 0, (C.c_uint32 * 60)(*probabilities))
        for backend in ('native', 'bpf'):
            ctx = Context(inp, Output(123, 456, 7, 8))
            if backend == 'native':
                result = self.lib.finemoe_select_native(C.byref(ctx))
            else:
                before = self.lib.finemoe_jit_calls(self.handle)
                result = self.lib.finemoe_select_bpf(self.handle, C.byref(ctx))
                self.assertEqual(self.lib.finemoe_jit_calls(self.handle), before + 1)
            self.assertEqual(result, 0, backend)
            self.assertEqual(ctx.output.status, 0)
            self.assertEqual((ctx.output.mask, ctx.output.cumulative_bits, ctx.output.selected), expected, backend)

    def test_crossing_equality_ties_and_zero(self):
        for values, threshold, k in [([.6,.3,.1], .8, 1), ([.5,.25,.125,.125], .75, 1),
                                    ([.25]*4, 0., 2), ([0.]*4, 1., 4),
                                    ([1.,0.,0.,0.], 0., 4), ([1./60]*60, 1., 4)]:
            self.compare([bits32(v) for v in values], bits32(threshold), k)

    def test_subnormals_and_rounding_boundaries(self):
        cases = [[1, 2, 0x7fffff, 0x800000],
                 [bits32(.5), bits32(2**-54), bits32(2**-55)],
                 [bits32(1.), bits32(2**-53), bits32(2**-54)],
                 [0x3f7fffff]*60]
        for values in cases:
            self.compare(values, bits32(1.), len(values))

    def test_seeded_distributions_and_exponent_gaps(self):
        rng = random.Random(20260903)
        for _ in range(3000):
            count = rng.randint(1, 60)
            values = [rng.random() for _ in range(count)]
            total = sum(values)
            self.compare([bits32(v / total) for v in values], bits32(rng.random()), rng.randint(1, count))
            # top_k=count checks the complete software-vs-hardware sum.
            values = [rng.randint(0, 0x3f800000) for _ in range(count)]
            self.compare(values, bits32(1.), count)

    def test_invalid_inputs_clear_outputs_and_do_not_fallback(self):
        cases = [Input(0,1,0,0), Input(61,1,0,0), Input(1,0,0,0),
                 Input(1,2,0,0), Input(1,1,0x7f800000,0), Input(1,1,0,1)]
        for bad in (0x7fc00000,0x7f800000,0xbf000000,0x3f800001):
            cases.append(Input(1,1,0,0,(C.c_uint32*60)(bad)))
        for inp in cases:
            for backend in ('native', 'bpf'):
                ctx = Context(inp, Output(123,456,7,8))
                result = (self.lib.finemoe_select_native(C.byref(ctx)) if backend == 'native' else
                          self.lib.finemoe_select_bpf(self.handle,C.byref(ctx)))
                self.assertEqual(result,1)
                self.assertEqual((ctx.output.mask,ctx.output.selected,ctx.output.status),(0,0,1))
        error = C.create_string_buffer(128)
        self.assertFalse(self.lib.finemoe_jit_open(b'/nonexistent/finemoe-policy',error,len(error)))
        self.assertTrue(error.value)


if __name__ == '__main__':
    unittest.main()

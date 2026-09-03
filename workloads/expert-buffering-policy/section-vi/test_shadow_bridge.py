"""CPU-only bridge checks. Fake ABI faults are fixtures, never live evidence.

Run after the coordinator closes the timing window with make -f shadow.mk test-shadow. This does not
build the existing selector or touch the offloader, torch, or any GPU.
"""
import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from test_policy import CacheOracle, Context, HIT, INVALID, NO_VICTIM

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# Minimal independent ABI fixture. Only this fixture interprets bytecode-path
# strings as faults. The production bridge has no fault-injection switch.
FAKE = r'''
#include "policy.h"
#include <cstring>
static int mode, closed, native_calls;
static eb_u64 calls;
static eb_context before;
static bool jit_first;
extern "C" void *eb_jit_open(const char *path, char *, size_t) {
    if (!path || !std::strcmp(path, "open-failure")) return nullptr;
    mode = path[0] - '0'; calls = 0; closed = native_calls = 0; jit_first = false;
    return &calls;
}
extern "C" eb_u64 eb_select(eb_context *ctx) {
    ++native_calls;
    if (!jit_first || std::memcmp(ctx, &before, sizeof(before))) return EB_INVALID;
    jit_first = false;
    ctx->output = {ctx->input.batch_epoch, EB_HIT, EB_NO_VICTIM};
    return EB_HIT;
}
extern "C" int eb_jit_select(void *, eb_context *ctx) {
    before = *ctx; jit_first = true; calls += mode == 4 ? 2 : 1;
    ctx->output = {ctx->input.batch_epoch, EB_HIT, EB_NO_VICTIM};
    if (mode == 1) ctx->output.victim = 7;
    if (mode == 2) ++ctx->input.experts[EB_MAX_EXPERTS - 1].token_count;
    return mode == 3 ? EB_ADMIT : EB_HIT;
}
extern "C" eb_u64 eb_jit_calls(void *) { return calls; }
extern "C" void eb_jit_close(void *) { ++closed; }
extern "C" int eb_fake_closed() { return closed; }
extern "C" int eb_fake_native_calls() { return native_calls; }
'''


class ShadowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="eb-shadow-test-")
        cls.fake_path = Path(cls.temporary.name) / "fake.so"
        subprocess.run(["/usr/bin/g++-13", "-std=c++17", "-shared", "-fPIC",
                        "-Wall", "-Wextra", "-Werror", "-Wl,--build-id=none",
                        "-I", str(HERE), "-x", "c++", "-", "-o", str(cls.fake_path)],
                       input=FAKE, text=True, check=True, timeout=20)
        cls.fake = C.CDLL(str(cls.fake_path))
        cls.lib = C.CDLL(str(BUILD / "libeb_shadow.so"))
        cls.lib.eb_select.argtypes = [C.POINTER(Context)]
        cls.lib.eb_select.restype = C.c_uint64
        cls.lib.eb_jit_open.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t]
        cls.lib.eb_jit_open.restype = C.c_void_p
        cls.lib.eb_jit_select.argtypes = [C.c_void_p, C.POINTER(Context)]
        cls.lib.eb_jit_select.restype = C.c_int
        cls.lib.eb_jit_calls.argtypes = [C.c_void_p]
        cls.lib.eb_jit_calls.restype = C.c_uint64
        cls.lib.eb_jit_close.argtypes = [C.c_void_p]
        cls.lib.eb_jit_close.restype = None
        cls.lib.eb_shadow_snapshot.argtypes = [C.POINTER(C.c_uint64)] * 3
        cls.lib.eb_shadow_snapshot.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "EB_SECTION_VI_UNTIMED_SHADOW": "1",
            "EB_SECTION_VI_REAL_LIBRARY": str(self.fake_path)})
        self.environment.start()
        self.handle = None

    def tearDown(self):
        if self.handle:
            self.lib.eb_jit_close(self.handle)
        self.environment.stop()

    def open(self, path=b"0"):
        error = C.create_string_buffer(512)
        self.handle = self.lib.eb_jit_open(path, error, len(error))
        self.assertTrue(self.handle, error.value)

    def snapshot(self):
        values = [C.c_uint64() for _ in range(3)]
        self.assertEqual(self.lib.eb_shadow_snapshot(*map(C.byref, values)), 0)
        return tuple(v.value for v in values)

    def context(self):
        cache = CacheOracle(3, 2)
        cache.begin([1, 1, 1])
        ctx = cache.snapshot(0)
        ctx.output.batch_epoch, ctx.output.status, ctx.output.victim = 99, 77, 55
        return ctx

    def test_explicit_guard_and_absolute_real_library(self):
        error = C.create_string_buffer(512)
        for flag in ("", "0", "true", "01"):
            os.environ["EB_SECTION_VI_UNTIMED_SHADOW"] = flag
            self.assertFalse(self.lib.eb_jit_open(b"0", error, len(error)))
            self.assertIn(b"UNTIMED_SHADOW=1", error.value)
        os.environ["EB_SECTION_VI_UNTIMED_SHADOW"] = "1"
        for library in ("", "relative.so", "/no/such/selector.so",
                        str(BUILD / "libeb_shadow.so")):
            os.environ["EB_SECTION_VI_REAL_LIBRARY"] = library
            self.assertFalse(self.lib.eb_jit_open(b"0", error, len(error)))

    def test_jit_first_identical_before_and_close_retains_snapshot(self):
        self.open()
        ctx = self.context()
        self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(ctx)), HIT)
        self.assertEqual(ctx.output.victim, NO_VICTIM)
        self.assertEqual(self.fake.eb_fake_native_calls(), 1)
        self.assertEqual(self.snapshot(), (1, 0, 1))
        self.assertEqual(self.lib.eb_jit_calls(self.handle), 1)
        self.lib.eb_jit_close(self.handle)
        self.lib.eb_jit_close(self.handle)
        self.handle = None
        self.assertEqual(self.fake.eb_fake_closed(), 1)
        self.assertEqual(self.snapshot(), (1, 0, 1))
        self.open()
        self.assertEqual(self.snapshot(), (0, 0, 0))

    def test_mismatch_is_sticky_and_never_replaces_jit_context(self):
        for mode in range(1, 5):
            with self.subTest(fault=mode):
                self.open(str(mode).encode())
                ctx = self.context()
                self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(ctx)), -2)
                self.assertEqual(ctx.output.victim, 7 if mode == 1 else NO_VICTIM)
                if mode == 2:
                    self.assertEqual(ctx.input.experts[59].token_count, 1)
                expected = (1, 1, 2 if mode == 4 else 1)
                self.assertEqual(self.snapshot(), expected)
                self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(ctx)), -1)
                self.assertEqual(self.snapshot(), expected)
                self.lib.eb_jit_close(self.handle)
                self.handle = None

    def test_single_instance_nulls_guard_revocation_and_native_rejection(self):
        self.open()
        error = C.create_string_buffer(256)
        self.assertFalse(self.lib.eb_jit_open(b"0", error, len(error)))
        self.assertIn(b"one active", error.value)
        ctx = self.context()
        self.assertEqual(self.lib.eb_select(C.byref(ctx)), INVALID)
        self.assertEqual(self.fake.eb_fake_native_calls(), 0)
        self.assertEqual(self.lib.eb_jit_select(None, C.byref(ctx)), -1)
        self.assertEqual(self.lib.eb_jit_select(self.handle, None), -1)
        self.assertEqual(self.lib.eb_shadow_snapshot(None, None, None), -1)
        os.environ["EB_SECTION_VI_UNTIMED_SHADOW"] = "0"
        self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(ctx)), -1)
        self.assertEqual(self.snapshot(), (0, 0, 0))

    def test_real_open_failure_does_not_reserve_instance(self):
        error = C.create_string_buffer(256)
        self.assertFalse(self.lib.eb_jit_open(b"open-failure", error, len(error)))
        self.open()

    def test_real_ubpf_same_snapshot_sequence(self):
        os.environ["EB_SECTION_VI_REAL_LIBRARY"] = str(BUILD / "libeb_policy.so")
        self.open(str(BUILD / "eb_policy.bin").encode())
        cache = CacheOracle(3, 2)
        cache.begin([1, 1, 1])
        decisions = 0
        for incoming in (0, 1, 0, 2):
            ctx = cache.snapshot(incoming)
            expected = cache.decide(incoming)
            self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(ctx)), expected[0])
            self.assertEqual((ctx.output.status, ctx.output.victim), expected)
            cache.commit(ctx, expected)
            decisions += 1
        cache.locked = set(cache.resident)
        ctx = cache.snapshot(1)
        self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(ctx)), cache.decide(1)[0])
        decisions += 1
        self.assertEqual(self.snapshot(), (decisions, 0, decisions))
        print(f"shadow_cpu_real_jit_checks={decisions} live_gpu_evidence=false", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

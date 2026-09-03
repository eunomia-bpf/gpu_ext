"""Bounded CPU decisions: independent cache oracle versus native and real uBPF."""
import ctypes as C
from pathlib import Path
import random
import unittest

MAX_EXPERTS = 60
NO_VICTIM = 0xFFFFFFFF
RESIDENT, ELIGIBLE = 1, 2
HIT, ADMIT, EVICT, INVALID, BLOCKED = range(5)


class Entry(C.Structure):
    _fields_ = [("token_count", C.c_uint32), ("flags", C.c_uint32),
                ("admission", C.c_uint64)]


class Input(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("abi_version", "count", "capacity", "incoming", "layer_id", "device_id")]
    _fields_ += [("batch_epoch", C.c_uint64), ("experts", Entry * MAX_EXPERTS)]


class Output(C.Structure):
    _fields_ = [("batch_epoch", C.c_uint64), ("status", C.c_uint32),
                ("victim", C.c_uint32)]


class Context(C.Structure):
    _fields_ = [("input", Input), ("output", Output)]


class CacheOracle:
    """Independent state machine; no native/BPF result determines its choice."""
    def __init__(self, count, capacity, layer=0, device=0):
        self.count, self.capacity = count, capacity
        self.layer, self.device = layer, device
        self.resident = {}
        self.epoch = self.clock = 0
        self.counts = [0] * count
        self.locked = set()

    def begin(self, counts):
        assert len(counts) == self.count and all(x >= 0 for x in counts)
        self.counts = list(counts)
        self.epoch += 1

    def snapshot(self, incoming):
        ctx = Context()
        ctx.input = Input(1, self.count, self.capacity, incoming,
                          self.layer, self.device, self.epoch)
        for expert in range(self.count):
            present = expert in self.resident
            flags = RESIDENT | (0 if expert in self.locked else ELIGIBLE) if present else 0
            ctx.input.experts[expert] = Entry(
                self.counts[expert], flags, self.resident.get(expert, 0))
        return ctx

    def decide(self, incoming):
        if not self.counts[incoming]:
            return INVALID, NO_VICTIM
        if incoming in self.resident:
            return HIT, NO_VICTIM
        if len(self.resident) < self.capacity:
            return ADMIT, NO_VICTIM
        eligible = self.resident.keys() - self.locked
        if not eligible:
            return BLOCKED, NO_VICTIM
        # Declarative ordering, separate from the C implementation's scan.
        victim = min(eligible, key=lambda e:
                     (self.counts[e] > 0, -self.resident[e], e))
        return EVICT, victim

    def commit(self, snapshot, decision):
        incoming = snapshot.input.incoming
        if bytes(snapshot.input) != bytes(self.snapshot(incoming).input):
            raise ValueError("stale epoch, cohort, residency, or eligibility")
        if decision != self.decide(incoming):
            raise ValueError("not the oracle's decision")
        status, victim = decision
        if status in (ADMIT, EVICT):
            if status == EVICT:
                del self.resident[victim]
            self.clock += 1
            self.resident[incoming] = self.clock
        assert len(self.resident) <= self.capacity


class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert C.sizeof(Entry) == 16 and C.sizeof(Input) == 992
        assert C.sizeof(Context) == 1008
        build = Path(__file__).resolve().parent / "build"
        cls.lib = C.CDLL(str(build / "libeb_policy.so"))
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
        error = C.create_string_buffer(512)
        cls.handle = cls.lib.eb_jit_open(str(build / "eb_policy.bin").encode(), error, len(error))
        if not cls.handle:
            raise RuntimeError(error.value.decode())
        cls.decisions = 0

    @classmethod
    def tearDownClass(cls):
        actual = cls.lib.eb_jit_calls(cls.handle)
        cls.lib.eb_jit_close(cls.handle)
        assert actual == cls.decisions, (actual, cls.decisions)
        print(f"native_decisions={cls.decisions} real_ubpf_jit_decisions={actual}", flush=True)

    def compare(self, ctx, expected):
        native = Context.from_buffer_copy(ctx)
        bpf = Context.from_buffer_copy(ctx)
        self.assertEqual(self.lib.eb_select(C.byref(native)), expected[0])
        self.assertEqual(self.lib.eb_jit_select(self.handle, C.byref(bpf)), expected[0])
        type(self).decisions += 1
        for actual in (native, bpf):
            self.assertEqual(bytes(actual.input), bytes(ctx.input))
            self.assertEqual((actual.output.status, actual.output.victim), expected)
            self.assertEqual(actual.output.batch_epoch, ctx.input.batch_epoch)

    def step(self, cache, incoming):
        ctx = cache.snapshot(incoming)
        expected = cache.decide(incoming)
        self.compare(ctx, expected)
        cache.commit(ctx, expected)
        return expected

    def test_paper_example_and_hit_does_not_refresh_lifo(self):
        cache = CacheOracle(4, 2)
        cache.begin([1, 1, 1, 0])
        self.assertEqual(self.step(cache, 0)[0], ADMIT)
        self.assertEqual(self.step(cache, 1)[0], ADMIT)
        self.assertEqual(self.step(cache, 0)[0], HIT)
        self.assertEqual(cache.resident, {0: 1, 1: 2})
        # Paper expert IDs 1,2,3 map to slots 0,1,2: evict 2, preserve 1.
        self.assertEqual(self.step(cache, 2), (EVICT, 1))
        self.assertEqual(cache.resident, {0: 1, 2: 3})

    def test_inactive_first_then_lifo_within_class(self):
        cache = CacheOracle(5, 3)
        cache.resident = {0: 1, 1: 2, 2: 3}
        cache.clock = 3
        cache.begin([0, 0, 1, 1, 1])
        self.assertEqual(self.step(cache, 3), (EVICT, 1))
        self.assertEqual(self.step(cache, 4), (EVICT, 0))
        cache.begin([1, 1, 1, 1, 1])
        self.assertEqual(self.step(cache, 0), (EVICT, 4))

    def test_eligibility_and_bounded_no_victim(self):
        cache = CacheOracle(3, 2)
        cache.resident, cache.clock = {0: 1, 1: 2}, 2
        cache.begin([0, 1, 1])
        cache.locked = {0}
        self.assertEqual(self.step(cache, 2), (EVICT, 1))
        cache.begin([0, 1, 1])
        cache.locked = {0, 2}
        before = dict(cache.resident)
        self.assertEqual(self.step(cache, 1), (BLOCKED, NO_VICTIM))
        self.assertEqual(cache.resident, before)

    def test_epoch_cohort_and_changed_residency_rejected_by_host_oracle(self):
        cache = CacheOracle(3, 1, layer=7, device=0)
        cache.begin([1, 1, 1])
        self.step(cache, 0)
        snapshot = cache.snapshot(1)
        decision = cache.decide(1)
        self.compare(snapshot, decision)
        cache.begin([1, 1, 1])
        with self.assertRaises(ValueError):
            cache.commit(snapshot, decision)
        for change in ("layer", "device", "locked", "resident"):
            ctx = cache.snapshot(1)
            if change == "layer":
                cache.layer += 1
            elif change == "device":
                cache.device += 1
            elif change == "locked":
                cache.locked.add(0)
            else:
                cache.resident[0] += 1
            with self.assertRaises(ValueError):
                cache.commit(ctx, decision)

    def test_invalid_contexts(self):
        cache = CacheOracle(3, 2)
        cache.begin([1, 1, 1])
        changes = [("abi_version", 0), ("count", 0), ("count", 61),
                   ("capacity", 0), ("capacity", 4), ("incoming", 3),
                   ("batch_epoch", 0)]
        for name, value in changes:
            ctx = cache.snapshot(0)
            setattr(ctx.input, name, value)
            self.compare(ctx, (INVALID, NO_VICTIM))
        for entry in (Entry(1, 4, 0), Entry(1, ELIGIBLE, 0),
                      Entry(1, 0, 1), Entry(1, RESIDENT, 0)):
            ctx = cache.snapshot(0)
            ctx.input.experts[1] = entry
            self.compare(ctx, (INVALID, NO_VICTIM))
        cache.resident = {0: 1, 1: 2, 2: 3}
        self.compare(cache.snapshot(0), (INVALID, NO_VICTIM))
        cache.resident = {}
        cache.begin([0, 0, 0])
        self.compare(cache.snapshot(0), (INVALID, NO_VICTIM))

    def test_deterministic_tie_and_u64_order(self):
        cache = CacheOracle(4, 3)
        cache.begin([1, 1, 1, 1])
        cache.resident = {0: 2**63, 1: 2**64 - 1, 2: 2**64 - 1}
        self.compare(cache.snapshot(3), (EVICT, 1))

    def test_fixed_seed_snapshots(self):
        rng = random.Random(0x5EB)
        for _ in range(1200):
            count = rng.randrange(1, MAX_EXPERTS + 1)
            cache = CacheOracle(count, rng.randrange(1, count + 1), layer=rng.randrange(24))
            cache.begin([rng.randrange(9) for _ in range(count)])
            residents = rng.sample(range(count), rng.randrange(cache.capacity + 1))
            cache.resident = dict(zip(residents, rng.sample(range(1, 10001), len(residents))))
            cache.locked = {x for x in residents if rng.randrange(4) == 0}
            incoming = rng.randrange(count)
            self.compare(cache.snapshot(incoming), cache.decide(incoming))

    def test_fixed_seed_batch_sequences(self):
        rng = random.Random(60024)
        for capacity in (1, 2, 8, 16, 60):
            cache = CacheOracle(60, capacity)
            for _ in range(12):
                cache.begin([rng.randrange(5) if rng.randrange(3) == 0 else 0 for _ in range(60)])
                # Execution order is expert ID, never predicted heat or cache hit.
                for expert, count in enumerate(cache.counts):
                    if count:
                        self.step(cache, expert)

    def test_missing_program_is_error_not_native_fallback(self):
        error = C.create_string_buffer(256)
        handle = self.lib.eb_jit_open(None, error, len(error))
        self.assertFalse(handle)
        self.assertIn(b"missing BPF", error.value)
        ctx = Context()
        self.assertEqual(self.lib.eb_jit_select(None, C.byref(ctx)), -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

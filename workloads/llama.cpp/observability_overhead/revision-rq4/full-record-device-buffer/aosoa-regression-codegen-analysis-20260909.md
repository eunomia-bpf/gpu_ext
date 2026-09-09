# AoSoA regression: read-only codegen analysis, 2026-09-09

Bounded, read-only explanation of the measured grouped-SoA regression
([campaign](../raw/full-record-aosoa-20260909.8vVDkh/README.md), commit
b210e802: AoSoA 23249.143131 vs SoA 27763.563111 token/s median,
paired change -16.275%, all five blocks negative, observed range
-16.490% to -15.972%). No GPU run and no rebuild were involved in this
note; the analysis is against the measured snapshot bc7ea6e6 objects.
A follow-up opt-in `LAYOUT=soa-warp` variant was subsequently added to
this source directory per root's geometry reading. Its separate
[five-block measurement](../results-full-record-soa-warp-20260909.md)
is now complete; it does not isolate the causes of the AoSoA regression.

## Observations

- Post-client whole-arena drain is nearly identical (1501.485 ms AoSoA
  vs 1495.369 ms SoA), and every collector reports the same 23,068,672
  records, 524,288 active slots, zero overflow/out-of-range, and all
  timestamps nonzero. The BTF of both staged BPF objects confirms both
  bank value types are 335,675,408 bytes with 32 map entries
  (`frdb_value_aosoa` single `fields[256][512][10][32]` array vs
  `frdb_value_soa` ten 4,194,304-u64 planes). The equal storage volume
  and the measured post-drain medians do not directly explain the
  prefill metric, and they do not establish that allocation or
  per-event cache behavior is layout-invariant; they only show the
  two arms move the same number of bytes out of the same-size arena
  in nearly the same measured time.
- Disassembling the two existing kretprobe objects
  (`llvm-objdump-18 -d` on `full-record-device-buffer.bpf.o` in
  `full-record-aosoa-20260909.wXFOqg/.output-aosoa/` and
  `full-record-soa-20260908.fpd5QL/.output-soa/`): 148 vs 142 BPF
  instructions. The prologue, the per-thread coordinate mapping
  (thread_id computation), the out-of-range sink, the map lookup, the
  256-record capacity check, and the counter update run the same
  instruction sequence apart from register allocation (the remaining
  1-instruction difference); both layouts perform the same 10 field
  stores plus 1 counter store.
- The dominant per-event difference is the store-path base computation
  (at the BPF bytecode level; what the GPU backend generates for these
  BPF instructions is not visible in the object files):
  - SoA: 5 BPF instructions, no multiply opcode:
    `r3 = counter; r3 <<= 0x11` (x 2^17 = 131,072-byte record stride),
    `r3 |= slot*8` (safe OR), `r0 += r3`.
  - AoSoA: 10 BPF instructions including two 64-bit BPF MUL opcodes by
    non-power-of-two constants:
    `r4 = counter; r4 *= 0x140000` (x 1,310,720 = 5 x 2^18),
    `r0 += r4; r3 >>= 5` (group), `r3 *= 0xa00` (x 2,560 = 5 x 2^9),
    `r0 += r3; r7 &= 0x1f` (lane), `r7 <<= 3; r0 += r7`.
  SoA's record and plane strides are powers of two (2^17, 2^25) and
  appear as shift+OR in BPF bytecode; AoSoA's record and group strides
  carry a factor of 5 intrinsic to 80 bytes per record
  (10 fields x 8 bytes = 80 = 5 x 16) and appear as MUL. Whether the
  GPU BPF backend executes those MUL opcodes as native 64-bit
  multiplies or strength-reduces the constants is not established from
  these artifacts.
- Address pattern difference: SoA field addresses are a shared per-lane
  base plus per-field immediates 32 MiB apart (one contiguous 32 MiB
  plane each); AoSoA field addresses are a shared per-lane base plus
  256-byte strides inside the 2,560-byte group window, where one
  (group, field) slice is strided 1.25 MiB apart across consecutive
  record indices k.

## Inference

- The effect is systematic (all five blocks negative; range 0.52 points
  wide) and sits in the per-event writer hot path: both arms commit the
  identical 23,068,672 records, and the prefill metric by construction
  excludes the post-client drain. The coordinate-mapping code is the
  same BPF instruction sequence in both objects. Whether allocation or
  per-event cache behavior differs is not established by these
  artifacts; the equal storage volume and similar drain medians do not
  rule that out.
- The per-event difference is 6 BPF instructions, of which 2 are 64-bit
  BPF MUL opcodes where SoA has one 17-bit shift and one OR. No native
  GPU instruction or multiply count is claimed: how the GPU BPF
  backend lowers those opcodes (native multiply or strength-reduced
  constants) is not visible in the object files. A ~4% BPF
  instruction-count difference alone does not obviously account for
  ~16%, so if the regression is in the writer, the evidence points to
  the execution cost of those two MUL opcodes under the GPU BPF
  backend and/or the changed address pattern (1.25 MiB-strided
  group/field slices vs contiguous 32 MiB planes), or both; both are
  consistent with the measured direction, and neither is established by
  these artifacts alone.

## Not determinable from these artifacts

- The cost split between the BPF op-mix difference (two extra 64-bit
  BPF MUL opcodes plus about five extra BPF instructions per event,
  whatever native code the GPU backend emits for them) and the changed
  address pattern cannot be separated without GPU-side profiling, which
  is out of scope here. This analysis does not claim which dominates.

## Follow-up candidates

The power-of-two-stride variant for the grouped layout is not a clean
isolation: making the record and group byte strides powers of two
requires padding (the arena value grows beyond 335,675,408 bytes) or a
changed mapping, so retaining the 2,560-byte group window does not
retain the arena footprint. It is recorded here only as a rejected
direction, not a queued candidate.

A higher-value follow-up, identified from the measured pp512 rope
geometry (grid 2048x1x1, block 1x256x1), targets the storage
bijection: the default index is row-major over (block_x, thread_x)
with width 2048, while physical warp lanes differ in thread_y, so one
physical warp's SoA stores land on 16 KiB-strided slots spread across
four banks rather than adjacent slots. The opt-in `LAYOUT=soa-warp`
build (FRDB_SOA_LAYOUT + FRDB_WARP_CONTIGUOUS_INDEX) implements the
standard block-major/thread-major storage index instead, keeping the
SoA value layout, all ten fields, per-thread timestamps, counters,
capacity, and whole-bank drain unchanged. It is documented in the
README. Its frozen snapshot
(`raw/full-record-soa-warp-20260909.cjoIog/source`, staged at
`/home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-soa-warp-20260909.3mlEOo`)
built with exit 0; all 15 measurements in the five-block comparison
against frozen SoA and an uninstrumented baseline subsequently completed.
The median paired throughput improvement is 20.565%. This supports the
storage-index optimization on the measured geometry, but does not by itself
separate instruction lowering, map access and memory-transaction costs.

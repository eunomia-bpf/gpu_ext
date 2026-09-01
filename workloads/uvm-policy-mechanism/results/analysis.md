# Provisional analysis: same-policy UVM mechanism cost

Status: all planned rows complete; independent result review pending.

## Admission

- 15 of 15 paired blocks and 30 of 30 timed processes completed.
- Every row used 8 GiB, 64 KiB regions, and 131,072 unique demand addresses.
- All 3,932,160 expected observed values across the 30 processes matched
  exactly; every row reported zero mismatches.
- Every gpubpf loader recorded ready and detaching, exited zero, and left no
  attached memory struct_ops before the next module reload.
- No UVM event monitor or kprobe tracer ran during retained timing.
- The untimed preflight separately proved real migrations, zero prefetch
  migrations/bytes, zero drops, and complete matching wrapper/helper coverage.

## Primary result

The paired geometric mean of `gpubpf/native` kernel time is **1.03219**, or
**3.219% overhead**. A 100,000-resample paired percentile bootstrap gives a 95%
interval of **[2.247%, 4.202%] overhead**.

- Native median: 364.041 ms.
- gpubpf median: 374.402 ms.
- Mean paired difference: 11.705 ms, or 89.30 ns per unique demand region as a
  descriptive amortization.
- gpubpf was slower in 14 pairs and faster in one pair.
- Native-first blocks have a 1.03353 geometric mean ratio; gpubpf-first blocks
  have a 1.03065 ratio, so the alternating order does not explain the effect.
- The descriptive log-ratio slope is -0.000638 per block; no row was removed or
  repeated based on this trend.

The bootstrap used seed 20260831, fixed immediately before aggregate analysis.
The approved plan required a fixed seed but omitted its numeric value; this is
recorded as a minor protocol omission rather than described as preregistered.

## Interpretation

For the scoped CPU-resident, non-first-touch UVM fault path, expressing the
driver's existing no-prefetch decision through gpubpf adds a small but measurable
mechanism cost on this RTX 5090/610.43.02 stack. This is not a policy speedup:
the actual prefetch outcome and numerical outputs are held constant. The result
supports disclosing a generality tax for this hook while attributing larger
application improvements elsewhere to policy choices.

The derived median fault-span rates are 21.976 GiB/s for native and 21.367 GiB/s
for gpubpf. These are virtual fault-span rates, not payload bandwidth, because
the kernel reads one word per 64 KiB region.

Scope limits: this microbenchmark isolates the page-prefetch mechanism. It does
not establish equivalence for preferred-location first-touch, other hooks, or
application-level policy benefits, and it does not replace the requested SOTA
application baselines.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this analysis.

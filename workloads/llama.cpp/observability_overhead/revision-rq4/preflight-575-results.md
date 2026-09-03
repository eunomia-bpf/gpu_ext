# RTX 5090 Table 1 preflight — 2026-09-03

**Incomplete experiment: no performance cell started.** All three 575
preflight attempts are retained; these are not additional paper results.
The third attempt reached all seven real correctness configurations on
575.57.08 / Linux 6.15.11 / CUDA 12.9. Its exact-output and probe checks
rejected four configurations. All seven owned teardown/safety checks passed.

## Attempts and repairs

1. `raw/preflight-575-01`: NVBit build failed before GPU use because sudo's
   PATH lacked nvcc. No runtime or correctness conclusion follows.
2. `raw/preflight-575-02`: explicit CUDA 12.9 PATH builds NVBit and all three
   BPF tools. Native llama-cli exits 0 but produces empty stdout, correctly
   rejected. Its `--log-disable` suppresses generated tokens, which this
   CLI emits with `LOG()`, not just diagnostics.
3. `raw/preflight-575-03`: removed only that output-suppressing flag for all
   correctness arms. Prompt, seed 1797, temperature 0, eight-token limit and
   exact nonempty-output comparison are unchanged. A regression test covers
   that command; all 24 CPU tests pass. This attempt ends with exit 2 because
   real correctness/engagement remains incomplete.

## Third-attempt observations

| Configuration | Saved observation | Existing check |
| --- | --- | --- |
| Native | Exit 0; 47-byte normalized stdout | Pass |
| NVBit exit records | Exact native output; 901,120 nonzero-timestamp records, 220 selected launches | Pass |
| BPF exit records | Exit 0; 24,279-byte stdout including runtime diagnostics; only 16,384 collected records | Fail exact-output match; collection coverage also unresolved |
| NVBit thread histogram | Exact native output; 901,120 counts, 22,528 nonzero thread slots, 220 launches | Pass |
| BPF thread histogram | Full 1,048,576-entry / 8,388,608-byte readback; 720,896 counts and 22,528 nonzero slots; 24,278-byte stdout | Fail exact-output match; total is also 20% below NVBit and needs explanation |
| NVBit launch latency | Exact native output; 220 clock errors, zero valid samples | Fail clock/sample checks |
| BPF launch latency | Zero host timestamps, zero valid device samples, 220 queue underflows; 24,280-byte stdout | Fail output and engagement checks |

The full-width metadata/readback repair is now exercised on the real GPU,
including the zero-valued tail. This narrow success does not validate event
coverage, launch correlation, or the whole comparison. Successful process
exit and plausible generated tokens cannot replace the recorded checks.

## Required diagnosis before another performance attempt

- Send instrumentation diagnostics to their own supported stream without
  accepting arbitrary stdout differences or changing the exact text oracle.
- Establish whether exit-record loss is collection/drain loss or different
  instrumentation coverage. Explain the separate histogram count mismatch.
  Comparing different delivered work would not establish lower overhead.
- Repair and validate both host/device clock correlation and BPF host-stub
  engagement. Do not clamp negative deltas, drop failed samples or count an
  empty latency histogram as a low-overhead result.

A requested read-only OpenCode subagent is diagnosing these source/log paths.
Its hypotheses require confirmation; no fourth preflight or formal timing is
launched by this report. Keep the failed attempts distinct from any later
source/configuration revision, and rerun affected correctness paths before
timing. Logs and result records are tracked; per-attempt generated tool
binaries are retained locally, with their source changes/build recipe in
`gpubpf-observability.patch` and `runtime-575/`.

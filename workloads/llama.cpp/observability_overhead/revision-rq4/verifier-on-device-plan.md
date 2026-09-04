# Verifier-on device experiment plan

## Execution update — 2026-09-04

The semantic port is committed as bpftime `b266cf2`, and its separate
`build-table1-575-strict` completed all 459 build steps. Two fresh real-device
strict counter pairs pass on the ported runtime; see the
[result report](../../../bpftime-device-smoke/results-table1-port-strict-575-20260904.md).
This closes the build/G1 gate. The final
[A0 actual-object campaign](results-preflight-575-strict-a0-03-20260904.md)
also passes all five correctness cells and its complete pp32 preflight block;
all four gpubpf cells bind exactly one STRICT acceptance and the expected map
to the recorded target PID. A1 admission timing and S0 paired steady-state
timing remain open. The runner records and independently audits explicit
STRICT or NO_VERIFY treatment without relabelling the accepted verifier-OFF
result. A0-01 remains a harness false-negative and A0-02 an intermediate pass;
neither was overwritten or promoted into the final gate.
The separate [NO_VERIFY control preflight](results-preflight-575-noverify-a0-01-20260904.md)
also passes all five correctness cells and one pp32 block on the same
verifier-enabled binary. Every gpubpf target has exactly one explicit skip and
zero timing/admission/map/rejection records. This is a treatment prerequisite,
not a cross-campaign paired result.

## Question and current boundary

Measure two different costs without conflating them:

1. the one-time latency to admit an actual Table 1 device callback under the
   strict SIMT-aware verifier; and
2. any steady-state throughput difference after an admitted callback is
   attached.

The accepted RTX 5090 Table 1 subset remains valid **verifier-off** evidence.
It must not be relabelled. A separate strict/off campaign is needed only for a
claim about verifier cost; the full baseline/NVBit campaign need not be rerun
unless the paper replaces the existing rows with verifier-on rows.

## Build gate

Create a separate `bpftime-table1-575/build-table1-575-strict` build with
`ENABLE_EBPF_VERIFIER=ON`, CUDA attachment on, and LLVM JIT on. Do not replace
the accepted performance build. The final verifier path in `bpftime-r5` is
roughly 3.4 KLOC across about 15 production files and overlaps the Table 1
attachment repairs, so there is no safe single cherry-pick. Port the final
tree semantically, retaining the Table 1 registration lock, bootstrap logging,
target-only late replacement, lossless teardown, and newer map-size handling.

Before timing, require all scoped verifier and attach tests plus the existing
two real strict positive/negative device pairs. The positive must admit and
execute 32,768 callbacks; the lane-varying negative must be rejected before
policy-entry insertion and bootstrap and retain zero callbacks. Generic
Frida/CUDA interception already exists before this admission point and is not
described as verifier-gated.

## Experimental cells

- **A0, actual-object admission:** feed the compact `kernelretsnoop` and
  `threadhist` BPF objects and their real map descriptors through strict
  admission. A rejected object is inconclusive and receives no performance
  cell.
- **A1, one-time cost:** for each admitted object, run at least ten fresh
  processes in randomized `STRICT`/`NO_VERIFY` pairs. Time only the verifier
  call, from entry to the explicit admitted/skipped record. Report paired
  latency separately from application throughput.
- **S0, steady state:** one pp32 correctness block followed by ten randomized
  pp512 `{control, STRICT, NO_VERIFY}` blocks per object. Start throughput only
  after admission, attachment, and warmup.

Every cell retains the existing normalized-output, event-count, map-readback,
drop/error, GPU-safety, process, shared-memory, and lease gates. Strict and off
use the same verifier-enabled binary, object, workload, and configuration;
only the mode changes. Do not compare new strict numbers to old off numbers as
a paired result. `launchlate` remains outside this campaign.

## Claim rule

Report the strict admission latency directly. For steady state, report the
paired strict-minus-off effect and interval. Only call it equivalent if the
entire predeclared 95% interval lies within a practical equivalence margin;
otherwise say that no clear difference was resolved. Do not infer zero runtime
cost merely because verification is intended to happen once.

## Independent reviews

- A source-diff audit found that a focused semantic port is required; initial
  verifier and attach commits omit later conservative fixes and are not safe
  to cherry-pick alone.
- Strict deny-all OpenCode review session
  `ses_f93f1c313ffeHh31TEuApc4GtH`, using the locally configured
  `spark-gateway/qwen3.8-flash-next-nvfp4-220k` model, independently selected
  the same separation between admission latency and steady-state cost. It also
  rejected retroactive verifier-on labelling and cross-session paired claims.

This document records completed G1/A0 gates and the reviewed A1/S0 execution
plan; it is not an A1 admission-cost or S0 steady-state result.

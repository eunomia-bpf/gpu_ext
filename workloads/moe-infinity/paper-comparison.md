# Same-frontend activation-aware policy comparison

The earlier 1.63/5.94/11.64 token/s diagnostic compared generic page-level
stride/LFU in llama.cpp, native UVM in llama.cpp, and the MoE-Infinity expert
engine. It is retained as a negative, confounded diagnostic, not a reproduction
of MoE-Infinity's activation-aware algorithm or a measurement of its BPF cost.

The new comparison uses the same repaired GPT-OSS-120B MoE frontend for all
three configurations:

| Configuration | Decision policy | Execution |
| --- | --- | --- |
| `native-off` | Current upstream GPT-OSS demand/cache path, no paper activation policy | Original frontend/native dispatcher |
| `paper-native` | Paper-v3 EAMC matching, proximity-aware prefetch ordering and scored eviction | Native selectors |
| `paper-bpf` | Identical paper-v3 decisions and shared float64 features | Three actual host uBPF JIT selectors |

This is an explicit paper-algorithm port to this frontend and hardware, not a
reproduction of the authors' original end-to-end platform. The algorithm and
paper-unspecified conventions are described in
[activation-aware-port.md](activation-aware-port.md). Host JIT execution is
not presented as execution of all three selectors inside kernel UVM hooks.

`run_paper_comparison.py` fixes five complete paired blocks. Every block uses
one shared shuffled order of the eight frozen 512-input/64-output prompts and
a different seeded permutation of the three configurations. Each cell launches
a fresh server, starts with an empty EAMC, uses the same excluded prompt-0
warmup, 0.75 device budget, 128 KV blocks and CPU 0–7 affinity. There is no
additional sleep cooldown: full model initialization and the identical warmup
occur before every measured cell. GPU telemetry records temperature, clocks,
power and hardware throttling. Heavy compilation must remain stopped during
timing; pre-cell, per-request and post-cell checks reject detected overlap.

All timed cells disable the native/BPF shadow oracle. Separately retained
three-mode canaries check real finite expert numerics, exact same-frontend
outputs, full SSE lifecycles, independent engine/token accounting and policy
engagement. Timing still requires every measured output to match its frozen
golden, exactly 64 token frames plus DONE per request, and independent engine
and serving metrics to agree on 512 generated tokens per cell.

The primary throughput is 512 tokens divided by the full eight-request wall
window, including the final prefetch drain. The runner also reports request-only
throughput, the drain tail, median/max request latency and time to first
nonempty text; that last metric is not claimed to be first model-token latency.
Actual policy calls and completed prefetch bytes must increase in the measured
window, not just during warmup. Exact selector equivalence is established by
the separate shadow-enabled correctness runs, not inferred from disabled
mismatch counters in timed runs.

No cell is successful until server teardown, GPU/driver cleanup and log checks
pass. After all writers stop, every owned raw artifact and the cell directory
are synchronized to storage before the successful result is published; this
flush is outside the measured window. A successful JSON summary alone cannot
prove that earlier buffered raw writes survived a reboot. Partial SSE responses,
failed cells and failed block attempts remain on
disk. A resumed campaign must use the same ordinary source revisions and file
inventories and re-audit successful blocks against their raw responses and
counter evidence. Only complete, unique three-configuration blocks contribute
to paired geometric throughput ratios and block-bootstrap confidence intervals.
One block is explicitly preliminary; completion requires five valid blocks.

The runner can release the shared GPU/struct-ops leases after a whole paired
block, allowing the coordinator to run GPreempt or XSched before resuming the
remaining MoE blocks without overlapping GPU timing:

```sh
.venv/bin/python -B run_paper_comparison.py \
  --output raw/paper-v3-575/timing-849ea75d-01 \
  --driver-stage /opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11 \
  --max-new-blocks 1
```

Rerunning the same command audits prior successful blocks and runs the next
one. Omitting `--max-new-blocks` finishes all remaining blocks. This convenience
does not reduce the fixed five-block requirement.

## Removing avoidable Python-to-BPF packing overhead

The initial correct port allocated a Python `ctypes` structure for every rank
candidate on every routed layer. `paper_policy_buffers.py` instead batch-copies
the unchanged 24-byte candidate ABI, preserving score bits, input ordinals,
identities and reserved zeros. It does not filter or sort candidates in the host;
all selection still occurs in the real BPF program, with the same output checks
and no fallback. The synchronous call retains ownership of its backing array.

The CPU-only diagnostic `raw/paper-v3-575/buffers-cpu-01.json` compares the
retained legacy bridge, packed bridge and native rank on the same 4,096 inputs.
For 300 calls each on CPU 8, observed times were 2.356, 0.571 and 0.730 ms per
call. This single-order microbenchmark identifies avoidable interface overhead;
it is not an end-to-end GPU performance result or a statistically controlled
native/BPF performance comparison. All 140 real-JIT boundary cases matched the
legacy and native decisions exactly. Separate layout tests cover noncontiguous
input, signed zero, infinities, NaN payloads, stable ordinals and pointer lifetime.
The packed path must pass a new real shadow-enabled SSE canary before timing.

# Predictive-prefetch factorial preflight 02

Date: 2026-09-03

Status: **all four correctness and engagement arms passed**. This is a
preflight with one measured request per arm, not a performance result.

## Scope and admission

The serialized run used an RTX 5090 at the fixed 400 W limit, NVIDIA
575.57.08, Linux 6.15.11, and the declared protected driver stage
`/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11`. Each arm started a
fresh 120B MoE server under the same 75%-of-HBM expert-cache budget and disabled
the temporary overload slot.

The first randomized block supplied the preflight order:

1. `bpf-prefetch-on`
2. `bpf-prefetch-off`
3. `native-prefetch-off`
4. `native-prefetch-on`

Each arm completed one independently checked 64-token SSE response matching
the frozen same-frontend golden. All servers exited 0, all cleanup lists were
empty, and the before/after safety and RM-warning gates passed.

## Mechanism engagement

The two prefetch-off arms issued and completed zero speculative copies. The
prefetch-on arms exercised real copies and first-use hits:

| Arm | completed prefetches | first-use hits | wasted | unused at drain |
| --- | ---: | ---: | ---: | ---: |
| BPF, prefetch on | 2,012 | 1,195 | 773 | 44 |
| Native, prefetch on | 1,964 | 1,163 | 768 | 33 |

Both rows conserve exactly as `completed = hits + wasted + unused at drain`.
The BPF-on cell made 3,780 measured eviction-policy calls (1,768 demand and
2,012 prefetch); BPF-off made 2,675 demand calls and zero prefetch calls. Native
arms made no BPF eviction calls. All four arms recorded zero temporary-slot
use, and no selector mismatch was observed.

## Independent audit and use

After every producer and telemetry process closed, a separate CPU-only call to
`validate_preflight` reread the raw SSE streams, goldens, launch recipes,
activation snapshots, engine metrics, JIT shutdown counters, telemetry,
cleanup, and safety records. It accepted all four cells.

The single-cell throughput values are diagnostic only. This directory is
eligible solely as the mandatory correctness/engagement prerequisite for a new
five-block, 20-cell factorial timing campaign; it must not be pooled into that
campaign or cited as comparative performance evidence.

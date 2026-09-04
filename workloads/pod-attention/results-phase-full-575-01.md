# POD device-policy phase decomposition on RTX 5090

## Result

The formal campaign completed all 15 fresh-process cells: five randomized
matched blocks across the original inline implementation, the same CUDA launch
adapter without BPF, and the BPF-controlled adapter. Each cell used the fixed
Llama-3-8B, batch-32 POD operator shape, 10 warmups, and 100 retained samples.
All numerical, exact-work, attachment, runtime, safety, telemetry, and cleanup
gates passed on driver 575.57.08.

The same-path operator result is a small but measurable BPF cost:

| Median of five cell means | Original inline | CUDA adapter | BPF adapter |
| --- | ---: | ---: | ---: |
| CUDA-event operator latency (ms) | 3.4430 | 3.4567 | 3.5120 |
| Synchronized host-wall operator latency (ms) | 3.4501 | 3.4638 | 3.5203 |

Across paired blocks, CUDA-event latency is 1.01777x for BPF versus the CUDA
adapter (a 1.78% increase; 95% interval 1.64--1.92%) and 1.02177x versus the
original inline implementation (2.18%; 2.07--2.30%). The CUDA adapter alone is
1.00393x the inline implementation (0.39%; 0.28--0.50%). Synchronized host-wall
latency for BPF versus the CUDA adapter is 1.01809x (1.81%; 1.66--1.96%). These
are pointwise whole-block bootstrap intervals, not equivalence tests.

## Cold-path finding

The current device-policy injection path has a much larger fresh-process cost:

| Median phase (ms) | Original inline | CUDA adapter | BPF adapter |
| --- | ---: | ---: | ---: |
| Before the first Python module statement | 19.8 | 21.4 | 271,224.9 |
| Complete 100-sample measurement loop | 699.0 | 703.0 | 1,667.9 |
| Whole cell including setup and cleanup | 7,509.6 | 7,310.7 | 280,667.4 |

The BPF whole-cell ratio is 38.30x the CUDA adapter (95% interval
37.60--38.94x), and 96.5--96.7% of each BPF cell occurs before the first Python
module statement. The private loader itself reports ready in a stable median
201.2 ms. The retained phase boundary cannot split the remaining pre-Python
time among dynamic loading, agent initialization, PTX transformation, and JIT
work, so the 271-second interval must not be called generic attach latency.

The 100-sample loop is 2.372x for BPF versus the CUDA adapter (2.369--2.375x),
but this interval includes correctness checks and a full decision audit after
every timed operator. It is a harness/audit-loop cost, not operator latency or
an end-to-end serving result. The operator-only result is the approximately
1.8% paired cost above.

## What this supports

For this frozen POD shape, the BPF mechanism reproduces the same device policy
and exact work with a 1.8% same-path operator-latency cost. It does not improve
the policy's performance, and the current cold path is unsuitable for frequent
fresh-process deployment. This complements the broader 250-cell POD operator
study: that study covers ten shapes and five arms, while this follow-up isolates
one shape to locate mechanism cost.

The result does not establish generic attachment latency, full POD serving
performance, strict-verifier admission, or constant overhead for other
handlers and kernels.

## Evidence

- Formal raw records and machine-readable analysis:
  `raw/phase-full-575-01/`
- Required three-arm preflight:
  `raw/phase-preflight-575-02/`
- Offline analyzer: `analyze_phase_study.py`
- OpenCode read-only review: `opencode-phase-analysis-review-01/`

# Transition validator Phase A results

Date: 2026-08-31

Disposition: `PASS` for the isolated candidate-header test only. This is not
yet production transition evidence.

## Scope

Phase A added the candidate production header
`kernel-open/common/inc/nv-gpu-transition-validator.h` on the driver
`test-sched` branch. The host test directly includes that header and covers:

- scheduler presence, native-minimum rejection, explicit interleave `LOW=0`,
  independent field results, two-snapshot identity/phase, repeat, and conflict;
- distinct initial and iterator action admission, action×region routing,
  absolute half-open region bounds, original `u64` endpoint rejection, and
  checked relative-to-absolute translation; and
- PMM callback-local attempt latching, invalid ordering sequences, snapshot
  identity/generation/source checks, and access/activate routing effects.

No running module, GPU, or BPF program was used. Live-preflight count: zero.

## Execution

The focused target and its containing target both compiled with
`-std=c11 -Wall -Wextra -Werror` and returned:

```text
PASS scheduler-presence-minimum
PASS interleave-low-range
PASS scheduler-identity-phase
PASS scheduler-repeat-conflict
PASS scheduler-independent-fields
PASS prefetch-action
PASS prefetch-action-region-routing
PASS prefetch-region-width
PASS prefetch-translation
PASS pmm-attempt-latching
PASS pmm-rejected-attempt-sequences
PASS pmm-snapshot-routing
PASS all: 12 cases, 145 assertions
```

The registration array contains those 12 unique names exactly once. A failed
assertion terminates the target with failure, so an empty or silently skipped
selector cannot produce the recorded result.

## Fresh review

The first fresh review reproduced the earlier 9-case/100-assertion run but
returned `BLOCK`: initial and iterator actions were not separated, three raw
width cases had invalid controls, scheduler independent-field behavior was
incomplete, and PMM invalid sequences were not paired with both actions.

After repair, an independent fresh rerun reproduced both targets at 12 cases
and 145 assertions and returned `PASS`. It confirmed that only the candidate is
out of range in width tests, all four PMM invalid sequences preserve state
under DEFAULT and BYPASS, and the missing scheduler/action cases are present.

## Boundary and next gate

The header currently has only the host test as a consumer. Therefore the result
is accurately described as an **isolated candidate-header test**, not as
production-shared validation and not as driver safety evidence.

Phase B must include this same header from the scheduler and UVM production
paths, add kernel-native PMM tests and real BPF verifier-load fixtures, and
re-run the host tests and module builds. Only that later evidence can establish
that production consumers use the tested definitions.

# Hummingbird host/device comparison — in progress

2026-09-08. Five blocks and four implementation arms use the existing
Hummingbird client, frozen idle-policy profile, 60-second periodic 100 Hz
foreground arrivals, and continuous ResNet-152 background requests.
No model/profile rebuild or new frontend is needed. This follows the
existing host-policy study, not a reproduction of another paper.

| Arm | Host decision | Device mapping |
|---|---|---|
| original_inline | Original C idle policy | Original inline coordinate mapping |
| inline_bpfhost | Matching BPF idle policy | Original inline mapping |
| native_adapter | Matching BPF idle policy | Callable native mapping adapter |
| bpf_device | Matching BPF idle policy | eBPF-derived callable mapping |

The callable native arm is an adapter control, not the unchanged original
implementation. Original no-policy baseline results remain in the existing
host study; they are not relabeled as contemporaneous controls here.

## First block completed

| Arm | BE requests completed | BE goodput, requests/s | HP response p99, ms |
|---|---:|---:|---:|
| original_inline | 7953 | 132.533333 | 1.892354 |
| inline_bpfhost | 7954 | 132.550000 | 1.913771 |
| native_adapter | 7743 | 129.033333 | 1.902283 |
| bpf_device, retained earlier run | 7718 | 128.616667 | 1.876349 |

All four clients exit zero and complete all 6000 foreground requests within
the measurement window. Each background worker completes one final request
after the window; goodput counts only completions inside it. Response p99
is the nearest-rank 99th percentile of completion minus scheduled arrival
from `GPREEMPT_LOAD_STUDY`, not the six-stage service-time sum.

The retained BPF cell is
`../raw-first-real-20260908.08lUpN/full-bpf-compatible-core.log`
(20:48:56--20:49:56 PDT). The three fresh controls run at 21:21--21:25 PDT
in the order recorded by `run-compatible-driver.sh`. Thus block 0 is not
fully adjacent or randomized and spans a module reload. It is retained to
avoid repeating completed work; final analysis must also exclude block 0.
This first block alone establishes neither equivalence nor stable overhead.

The three-control lifecycle exits zero, restores the previous stock core
and saved GDS UVM, and reattaches GDS/KV loaders 4107386/4107388. Their
logs report `attached`, both services are active, and GPU memory returns
to 1 MiB. Explicit PIDs in the script describe that invocation only.

## Remaining four blocks

Root starts the 16 missing cells at 21:27 PDT using `run-remaining.sh` and
the already-working client. Four-arm order is shuffled within each block
using Python `random.Random(20260908)`; the exact expanded sequence is in
the script. Both experiment locks remain held for this batch, including
compatible-core setup and restoration. These cells are **in progress**;
their completion and final performance are not established by this record.

Large cubins remain in the existing temporary build directory, not Git.
Completed raw logs are preserved verbatim. The two earlier ioctl startup
failures and the original host-only/pipeline results remain unchanged.

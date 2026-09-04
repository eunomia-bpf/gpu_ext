# Source audit: warp-leader device-hook implementation

Status: **no claim-matched implementation or runnable entry was found in the
bounded local source inventory**. This is a source result, not a GPU result.

## Acceptance criteria

A source tree counted as the Fig. 15(a) warp-aggregation implementation only
if one connected hook path visibly performs all of the following:

1. obtains the active-lane mask and elects an active leader;
2. gathers or reduces lane-local hook inputs with a warp collective;
3. invokes the JIT-compiled scalar eBPF handler only in that leader;
4. broadcasts the handler's result or decision to the participating lanes;
5. provides a build/run entry and an externally checkable callback-count and
   broadcast oracle.

A CUDA kernel that happens to use a shuffle, a verifier model that assumes
uniform execution, or a loop that serializes already-running lanes does not
meet these criteria.

## Bounded inventory

The audit searched current source and every locally reachable Git ref in:

- `gpu_ext`;
- `bpftime`, `bpftime-r5`, and `bpftime-table1-575`;
- the other local bpftime worktrees/clones named `bpftime-pr*`,
  `bpftime-fuzzy-backtracer`, `bpftime-preload-safety.*`, and
  `eunomia.dev/bpftime`;
- the source-bearing portions of `bpftime-verifier`, the SPIR-V build tree,
  the 575/610 driver trees, and `jax-xla-mapping`.

The search covered CUDA/C/C++, PTX, LLVM IR, BPF sources, patch files, build
entries, filenames, and all-ref Git diffs. Terms included CUDA and PTX shuffle,
ballot/vote, active-mask, lane election, lane-zero guards, warp leader, warp
aggregation, and the proposed helper IDs 520/521. Build products, vendored
libraries, model caches, result directories, and unrelated application kernels
were excluded after classification. No network source was fetched.

## Runtime evidence

The selected RTX 5090 runtime is scalar per thread:

- `bpftime-table1-575/attach/nv_attach_impl/pass/ptxpass_kprobe_entry/main.cpp`
  lines 53--80 replace an existing `call` or `call.uni` target stub directly
  with the compiled eBPF function name. Lines 83--95 insert an unconditional
  ordinary call in the fallback path. There is no mask, leader predicate, or
  result broadcast.
- `.../ptxpass_kretprobe/main.cpp` lines 36--64 insert an ordinary call before
  each matching `ret`/`exit`, retaining only the original exit predicate.
- `.../trampoline/default_trampoline.cu` lines 140--179 use `__activemask()`
  inside a 32-iteration host-helper RPC loop. Each active lane takes its own
  turn; this serializes per-lane requests and does not deduplicate a handler
  invocation. Lines 554--569 merely expose warp and lane IDs.
- The generated `trampoline_ptx.h` contains active-mask and lane-ID
  instructions but no shuffle or ballot/vote instruction. No runtime source
  defines helper 520 or 521.

The strict SIMT verifier does not supply the missing execution mechanism.
Current `bpftime-verifier/src/gpu/gpu_platform.cpp` registers device helpers
only through ID 511. An older verifier-only commit, `9a753e1`, introduced
`bpf_warp_ballot_sync_placeholder` and `bpf_warp_shfl_sync_placeholder` in its
analysis model and explicitly described them as future infrastructure; it did
not add CUDA helper definitions, a PTX wrapper, or a run entry. The current
table1 verifier entered independently in `b266cf2` and omits those placeholders.

All locally reachable bpftime refs produce the same result: the only code-history
match for shuffle/ballot or warp-aggregation terminology is the verifier-only
placeholder commit. No deleted/replaced hook implementation is reachable from
the local refs.

## Near misses in `gpu_ext`

| Source | What exists | Why it is not Fig. 15(a) |
|---|---|---|
| `docs/experiment/test-verify/examples/05_map_bandwidth_contention.cu:144` | A standalone CUDA function reduces a synthetic metric with `__shfl_down_sync`; lane 0 writes a raw device array. `make run5` is a nominal entry. | It simulates BPF maps, does not load/JIT/attach eBPF, never broadcasts a policy result, has no callback-count oracle, and allocates about 20 GiB. It is a verifier-motivation demo, not the gpubpf hook. |
| `microbench/clc_bench/clc_policy_framework.cuh:128` | Thread 0 evaluates a compile-time C++ CLC policy and shares `go` through CTA shared memory and barriers. | This is block-level persistent-kernel scheduling, not a warp-level dynamically attached eBPF handler, and it uses no shuffle broadcast. |
| `microbench/clc_bench/README.md` | Example snippets contain `__shfl_sync`. | The snippets describe CLC result sharing and do not occur in the compiled implementation above. |
| `bpftime/.../default_trampoline.cu:140` | Active lanes are enumerated around the host-helper bridge. | Every active lane still issues its own request; it is serialization, not leader execution plus broadcast. |
| `jax-xla-mapping` and workload CUDA sources | Many ordinary application reductions use shuffle intrinsics. | None connects to the bpftime attach pass, eBPF JIT, or gpubpf hook ABI. |

The local paper-material directory contains the eGPU PDF but no eGPU source
checkout or pinned build/run artifact. Therefore there is also no local
official comparator entry to pair with a future warp arm.

## Gap and minimum valid recovery

There is no existing minimum run command: the mechanism itself is absent.
Running the synthetic CUDA example or relabeling the current per-thread
microbenchmark would not test the paper claim.

A valid recovery must first add or recover a PTX/CUDA wrapper that implements
the five acceptance criteria above. Before timing, one untimed test must prove
exactly one handler side effect per active warp per launch and must make every
participating lane consume the leader's broadcast decision. The transformed
PTX must retain the election, single guarded handler call, reconvergence, and
shuffle broadcast. Only after that engagement test passes can a randomized,
paired comparison use the same CUDA binary and handler for:

1. the current scalar per-thread path;
2. the recovered warp-leader path; and
3. a separately pinned official eGPU implementation, if available.

Until both missing source paths exist, Fig. 15(a), its 60--80% reduction text,
and any universal once-per-warp performance statement remain unsupported. The
current per-thread measurements, SIMT-verifier evidence, fixed-work trampoline
study, and map-tier experiment answer different questions and cannot replace
this source requirement.

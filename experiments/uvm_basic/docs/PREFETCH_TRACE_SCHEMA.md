# Prefetch Decision Trace Schema

Evidence class: `GPU_EXT_PREFETCH_DECISION_TRACE`.

`extension/prefetch_trace` emits one `CALLBACK` row at entry to `uvm_bpf_call_gpu_page_prefetch()` and one `DECISION` row at the end of `compute_prefetch_region()`. A BPF map allocates `call_id` only while the trace is attached; the module itself does not maintain a counter.

## Region Semantics

| Region | Meaning | Producer |
|---|---|---|
| `max_candidate_[first,outer)` | Maximum range UVM permits for this decision | UVM input |
| `policy_result_[first,outer)` | Region written by the initial policy callback before tree traversal or clamping | Policy output snapshot |
| `final_effective_[first,outer)` | Region returned after action handling, optional tree traversal, offset adjustment, and candidate-bound clamping | UVM final result |

`outer` is exclusive. `final_pages` is `max(0, final_effective_outer - final_effective_first)` in UVM page-index units. Converting it to bytes requires the page geometry of the exact module build; the CSV intentionally does not hard-code 4096 bytes.

## Fields

| Field | Unit | Source | Class | May be zero | Interpretation |
|---|---|---|---|---|---|
| `time_ms` | ms | trace process | derived | yes | Time since first captured event |
| `timestamp_ns` | monotonic ns | `bpf_ktime_get_ns()` | trace | no | Aligns decisions with workload phase windows |
| `event_type` | enum text | trace probe | trace | no | `CALLBACK` or `DECISION` |
| `call_id` | run-local integer | attached BPF maps | correlation | yes | Zero means entry/final correlation was unavailable |
| `cpu` | CPU id | BPF helper | context | yes | CPU executing the UVM path |
| `current_pid` | pid | BPF helper | context | yes | Current kernel task, not necessarily allocation owner |
| `fault_pid` | pid | cached VA-block state | context | yes | Existing UVM fault-authorized PID field |
| `owner_tgid` | tgid | VA-space `mm->owner` | attribution | yes | Preferred target-process attribution when available |
| `va_start`, `va_end` | user VA | VA block | identity | yes | Managed VA-block range; not a kernel pointer |
| `page_index` | UVM page index | `compute_prefetch_region()` | input | yes | Triggering page within the candidate context |
| `faulted_first`, `faulted_outer` | UVM page index | cached fault region | input | yes | Faulted interval |
| `max_candidate_first`, `max_candidate_outer` | UVM page index | max region argument | input | yes | Allowed interval before policy execution |
| `action`, `action_name` | enum | policy wrapper return | policy output | yes | Actual action and decoded name |
| `policy_result_first`, `policy_result_outer` | UVM page index | callback-result snapshot | policy output | yes | Region immediately after initial callback |
| `final_effective_first`, `final_effective_outer` | UVM page index | return value | final result | yes | Region UVM uses after action processing and clamp |
| `final_pages` | pages | final interval | final result | yes | Zero is a valid empty decision |
| `tree_offset`, `leaf_count`, `level_count`, `pages_accessed` | count/index | bitmap tree | callback input | yes | Populated on callback rows; decision rows may be zero |

## Actions

The current source defines these exact values in `uvm_bpf_struct_ops.h`:

| Value | Name | Semantics |
|---:|---|---|
| 0 | `DEFAULT` | Ignore policy-region selection and execute original UVM bitmap-tree selection |
| 1 | `BYPASS` | Skip UVM tree selection and use the policy result, subject to final offset/clamp logic |
| 2 | `ENTER_LOOP` | Traverse the tree and invoke the iterative policy hook before final clamp |

Any other value is exported as `UNKNOWN`; it is never silently mapped to a known action. A callback is not a page fault, and a decision can cover more than one page.

## Probe Locations

- NVIDIA-origin file with minimal gpu_ext call points: `kernel-open/nvidia-uvm/uvm_perf_prefetch.c`, `compute_prefetch_region()` lines 103-179 in the current working tree.
- gpu_ext-added wrapper implementation: `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c`, `uvm_bpf_trace_gpu_page_prefetch_decision()` near line 424.
- Attached BPF trace and run-local call correlation: `extension/prefetch_trace.bpf.c`.

When no BPF trace is attached, the wrapper performs only a noinline call and compiler barrier. It allocates no memory, takes no lock, increments no counter, and logs nothing. The 10-run 256 MiB measurement was 243.496 ms versus the prior 240.731 ms custom no-policy mean, or +1.149%. With the decision trace attached the mean was 285.711 ms, +17.337% over the enhanced untraced module.

## Runtime Validation

All representative Stage 3A runs had equal callback and decision counts, complete `call_id` matches, zero callback-without-decision rows, and zero unknown actions.

| Policy | Callback/decision count | Action | Policy equals final | Final pages |
|---|---:|---|---:|---|
| custom no-policy | 17,621 | DEFAULT | 0 | median 64, p95 256, max 512 |
| prefetch_none | 393,216 | BYPASS | 393,216 | always 0 |
| prefetch_always_max | 768 | BYPASS | 768 | always 512 |
| prefetch_adaptive_sequential | 12,349 | BYPASS | 12,349 | mean 187.85, max 220 |

The 768 `always_max` decisions each selected the full 512-page candidate in that representative run. They are not 768 distinct hardware faults or necessarily 768 unique regions; CPU-first-touch and GPU-demand activity share the isolated trace window. No `ENTER_LOOP` action occurred in this sequential vector-add matrix.

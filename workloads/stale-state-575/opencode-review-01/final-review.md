READY AT DECLARED BOUNDARY

All attached sources maintain the declared CPU/source boundary and satisfy every criterion:

1. **No implied live execution**: `run_study.py live` raises `ValidationError` before any path, lease, process, or device access. `dry_run_plan` returns `live_executable: False` and `writes_output: False`.

2. **Dry-run is side-effect-free**: It only prints a frozen plan; `test_dry_run_has_no_filesystem_or_process_side_effects` enforces this with mocked filesystem/subprocess.

3. **Preflight precedes formal campaign**: `validate_campaign` requires an excluded complete seven-cell preflight (`PREFLIGHT_BLOCKS=1`), validates its exact order via `_validate_manifest`, and rejects self-referential preflight paths. Mechanism and information cost comparisons are paired fairly within blocks.

4. **No synthesized/proxy counters**: `_validate_uvm` rejects any `dropped_*` > 0; `_validate_truth` enforces exact 15-record phase truth; `_validate_policy` requires every decision to join a host-truth interval; numerical mismatches, foreign PIDs, and cleanup errors all raise `ValidationError`.

5. **All decision ages retained**: CPU relay preflight confirms distinguishable medians (~0 ms, ~100 ms, ~1000 ms); wrong-phase decisions are independently counted from raw timestamps.

6. **Valid negative rows retained**: Documentation and `validate_campaign` both preserve negative results; `negative_results_retained` is reported and all 12 information-cost rows can be negative without affecting validity.

7. **No source/ABI issues**: `make test-offline` passes (C policy assertions + 13 Python tests); `make stale_state_workload` compiles with nvcc under C++17; all `_Static_assert` ABI checks pass.

## Follow-up after the final README update

READY AT DECLARED BOUNDARY

No inconsistencies found. The updated README's new claims all trace to the corresponding enforcement in the source:

- **"exact shared-publisher, legal-region, driver-effect fields"**: `protocol.py:_validate_policy` enforces `publisher == "shared_driver_snapshot"`, `effect_source == "driver_diagnostic"`, and `maximum_first <= fault_page < maximum_outer <= REGIONS`, with dense-prefetch output matching the full legal maximum and sparse-discard output matching the empty region.
- **"analysis revalidates all seven preflight cells before accepting the 21 formal cells"**: `run_study.py:analyze` calls `protocol.validate_campaign`, which calls `validate_preflight(preflight_path)`—validating all 7 preflight cells via `validate_cell` for each—before proceeding to validate the 21 formal campaign cells. The preflight path must be absolute and must not self-reference.
- **`make test-offline`** still passes C assertions plus 13 Python tests; the `analyze` subcommand exists and delegates to `validate_campaign`.

The declared boundary holds.

## Final source follow-up after inherited-UVM-fd change

READY AT DECLARED BOUNDARY

No regressions found across the changed files:

**Compile**: `uvm_event_monitor.c` still includes all required headers; the new `fcntl(F_GETFD)` validation is well-formed; unused `<sys/syscall.h>` is harmless. Both `protocol.py` and `test_offline.py` changes are syntactically consistent.

**Ownership**: The coordinator now owns pidfd duplication and passes an inherited fd to the monitor, matching the established runner pattern. The monitor closes only its inherited copy in cleanup (`close((int)inherited_fd)`), not the coordinator's copy or the `/dev/nvidia-uvm-tools` fd.

**Dry-run**: `dry_run_plan` still returns all false flags (`launches_processes`, `executes_gpu_work`, etc.) and `live_executable: False`. The `uvm_monitor_command` now documents `--uvm-fd`/`--target-pid` instead of `--pid`/`--target-fd`, which is a plan-text update, not an execution change.

**Evidence-integrity**: `_validate_uvm` now validates the ready record (`target_pid` match, `queue_entries == 65536`, `entry_bytes == 72`, non-negative `uvm_fd`), and `test_offline.py::make_cell` generates matching ready records with those fields. The `validate_cell` to `_validate_uvm` call chain passes `execution["target_pid"]` correctly. All 13 Python tests and C assertions remain consistent.

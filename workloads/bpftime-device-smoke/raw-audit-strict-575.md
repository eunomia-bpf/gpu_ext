# Independent strict-device evidence audit

2026-09-03. **No blocking mismatch found** in the narrow claim made by
[the result report](results-strict-575-20260903.md). This audit read the four
completed cells' original logs/results, the retained failed cell, build/test
logs, parser changes, and R5 revision `ea9907d`. It ran only the existing
CPU-only Python tests and read-only log assertions: no rebuild, BPF attachment,
GPU workload, driver operation, or main-runtime edit.

| Raw cell | Independently checked evidence |
| --- | --- |
| [02 positive](raw/575-r5-strict-02/positive/result.json) | STRICT admission of 13 actual instructions and map 1502; eight patched vector launches; 4096 thread counters each equal eight, sum 32768 |
| [02 negative](raw/575-r5-strict-02/negative/result.json) | Actual 15-instruction negative rejected for lane-varying branch at instruction 1; `hook_created=0`; propagated initialization failure; no recorded policy pass or patched launch; fresh all-zero counters |
| [03 positive](raw/575-r5-strict-03/positive/result.json) | Same complete admission, launch and 32768-counter evidence in a fresh process/private segment |
| [03 negative](raw/575-r5-strict-03/negative/result.json) | Same explicit rejection and fresh all-zero observation in a fresh process/private segment |

All four cells have exactly one native and one instrumented numerical record,
each checking all 32768 vector values with zero mismatches. Every probe reports
4096 allocated map slots and a **32768-byte** userspace readback, not the former
8192-byte prefix. Native application success is not used as the rejection
oracle. The runner samples the negative map again after the target has exited;
both negative cells retain that later zero snapshot.

The four private names are distinct, their saved identities are present,
removal is recorded and their paths are absent at audit time. Every cell has
empty owned-survivor state. All eight pre/post snapshots show 400 W, UVM
references zero, no compute clients or struct_ops maps/links, and no Xids or
abnormal kernel records. Cleanup stops the target group before the probe and
checks survivors even after a group leader exits; replacement/unknown segments
are not deleted. These are observed bounded smoke results, not proof against
arbitrary concurrent external interference.

## Repairs and retained failures

- R5 `ea9907d` changes only two files relative to `b4b0ba8`: seven insertions,
  one deletion. The production change stores the selected allocation count in
  `runtime/src/handler/map_handler.cpp:1002`; userspace copy sizing at `:69–78`
  then uses the same count. It does not change verifier rules or thresholds.
- The corrected C++ fixture samples `is_enabled()` before rejection and
  requires that state to remain unchanged, plus no hook ID 1. This matches
  `enabled{true}` in `nv_attach_impl.hpp`; the strict return still precedes
  `allocate_id()`, `hook_entries`, and late bootstrap in
  `nv_attach_impl.cpp:240–303`. General CUDA interception/context setup may
  already exist, as the report explicitly distinguishes.
- The Python parser now matches the exact 15-character syscall program name
  `cuda__count_ret`, while the ELF symbol remains `cuda__count_return`. Tests
  reject the full ELF name, shorter prefix, and another program name.
- [Run 01](raw/575-r5-strict-01/positive/result.json) remains failed, with no
  negative directory. Its raw target log actually admitted the truncated name,
  but the old parser did not recognize it. Its probe allocated 4096 slots while
  advertising only 8192 readback bytes, exposing just 1024 counters / 8192
  callbacks. The later complete 32768 requirement was not reduced.
- The initial C++ `REQUIRE_FALSE(is_enabled())` failure remains in
  `strict-counter-test.log`; the corrected log records six assertions passing.
  Public verifier tests record 5 cases / 28 assertions, and the separate
  mode-control fixture records five assertions. The latter intentionally tests
  WARNING/DISABLED behavior for a **different** fixture and is not evidence of
  bypass in either strict raw pair.

The inspected R5 CMake cache has verifier, CUDA attach, and LLVM JIT all ON;
raw targets identify R5-built agent/pass paths. Configure/build logs agree on
CUDA 12.9, GCC 13.3, LLVM 15 and the 277-step build, followed by scoped rebuilds.
Existing ODR/header build warnings and the runtime's duplicate-VM-registration
warning remain visible; this audit does not certify those unrelated paths.
The separate performance build still has `ENABLE_EBPF_VERIFIER=OFF`.

Audit rerun: `taskset -c 17 python3 -B -m unittest -v test_runner` from this
directory passed **12 tests**. Independent read-only assertions also matched
all four raw verdict/numeric/readback/cleanup records and retained 01 as failed.

Evidence detail: the negative `result.json` verifier summary omits the separate
continuation line containing `(mode=STRICT, hook_created=0)`. That historical
field means no policy entry was allocated; generic Frida/CUDA interception was
already installed. The ported runtime uses the less ambiguous
`policy_entry_created=0`. The old line is present at line 126 of both full
`instrumented.log` files, which the historical runner validates.

Claim boundary remains unchanged: these are two successful strict counter
positive/negative pairs, not general SIMT soundness, POD pointer/ticket ABI
validation, performance results, or native scheduler-init/prefetch coverage.

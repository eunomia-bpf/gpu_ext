# Device-verifier scaling offline validation

Date: 2026-09-04. No verifier admission, GPU command, GPU device access, lease
operation, preflight, or formal timing run was performed in this stage.

## Isolated build

- Source tree: `/home/yunwei37/workspace/gpu/bpftime-table1-575`
- Source Git revision embedded in the probe:
  `39d099198938c122e67372d02eaaabd3aaf86436`
- Build tree:
  `/home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build`
- Configuration: `Release`, CMake 3.28.3, GCC 13.3.0, Ninja, at most two
  parallel build jobs
- Built target: `verifier_scaling_probe` plus only its verifier-library
  dependencies
- Probe metadata after the clean incremental build: 1,661,480 bytes; modified
  2026-09-04 10:07:47.931113904 -0700
- Existing `build-table1-575-strict` artifacts were not inputs and were not
  modified.

The first incremental metadata rebuild exposed incorrect shell quoting in
compile definitions. Ninja then linked the previous object, and the live
description test correctly rejected that stale executable because it lacked
the required `Release` metadata. The implementation now generates a configured
C++ header instead. A subsequent configure/build compiled both metadata values
into the probe and all live description checks passed. No failed executable was
used for an API call or timing measurement.

## Offline checks

The final command ran Python bytecode compilation, twelve `unittest` methods,
all ten live `--describe` arms, the explicit rejection of a non-frozen size,
and a dry reconstruction of the 200-cell/20-block schedule.

- Result: 12/12 test methods passed, no skip in the probe-supplied run.
- Analyzer fixtures: one complete 200-cell run and one two-cell preflight.
- Fail-closed mutations: 18 cases covering run status, dirty verifier source,
  cpufreq drift, schedule drift, missing cell/warmup, start/end revision drift,
  executable replacement, stderr, argv, timeout, non-positive timing, CPU
  drift, rejected admission, wrong structural branch counts, and JSON
  boolean/integer confusion; three additional checks cover non-object
  top-level results, execution records, and probe metadata.
- Noise behavior: a major-fault mutation keeps every row and changes only the
  hypothesis classification to inconclusive.
- Dry schedule: exactly 200 cells, 20 blocks, and every block contains each of
  the ten frozen arms once.

`--describe` constructs the complete C++ instruction vector, validates its
prefix/body/exit and all branch targets, and emits structure metadata. It does
not invoke `verify_gpu_program` and does not read either experiment clock.

The final rebuild occurred after mechanical C++ formatting and compiled the
current configured metadata header. Python compilation, shell lint, C++ format
validation, all tests, and the dry schedule check then passed in one command.

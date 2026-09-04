# Device-verifier admission scaling result

The frozen CPU-only campaign completed without retry. The real preflight
accepted both endpoint programs and is retained only as a dependency check.
The full run accepted all 200 programs in 20 fixed-seed randomized complete
blocks. The independent analyzer reopened every raw stdout, stderr, and
execution record and reports `complete: true`, `run_status: valid`, and no
errors.

An independent read-only result review then reconstructed the matrix and
statistics directly from the raw files. It found no invalidating blocker and
returned:

- **Run status:** valid.
- **Tested hypothesis:** contradicted.
- **Research value:** supporting.
- **Paper impact:** a mechanism/program-shape boundary and additional RQ4
  evidence, not a direct challenge to the paper's central thesis.
- **Next paper decision:** report acceptance through 4,096 instructions and the
  measured superlinear linear-family boundary; do not claim universally
  near-linear verifier scaling.

## Result

| Instructions | Linear median [95% CI], ms | Diamonds median [95% CI], ms | Diamonds / linear [95% CI] |
|---:|---:|---:|---:|
| 16 | 0.880 [0.871, 0.908] | 1.982 [1.929, 1.988] | 2.210 [2.173, 2.240] |
| 64 | 3.366 [3.266, 3.412] | 7.838 [7.724, 7.926] | 2.323 [2.274, 2.392] |
| 256 | 18.375 [17.988, 18.616] | 32.412 [31.999, 32.572] | 1.757 [1.729, 1.814] |
| 1,024 | 156.075 [155.554, 157.338] | 135.116 [134.031, 135.792] | 0.863 [0.859, 0.872] |
| 4,096 | 1,899.001 [1,895.921, 1,907.625] | 572.025 [569.692, 575.173] | 0.301 [0.300, 0.302] |

The log-log Theil--Sen exponent is 1.3841 [1.3813, 1.3898] for the linear
family and 1.0255 [1.0215, 1.0315] for the uniform-diamond family. The frozen
rule classifies the tested hypothesis as **contradicted** because the linear
family's lower interval endpoint exceeds 1.25. Dense uniform branches cost
more at 16--256 instructions, but the relation crosses over by 1,024
instructions; at 4,096 instructions the diamond program takes 30.1% of the
linear program's admission time. Across the 20 paired blocks, the 4,096-arm
ratio ranges from 0.296 to 0.306, so the reversal is not caused by a single
measured outlier.

This does not show that adding branches speeds verification. Exact instruction
count is matched, so the diamond constructor necessarily replaces half of the
linear ALU operations with branches, and the direct API interval combines
PREVAIL, uniformity, and SIMT passes. The result establishes a program-shape
boundary: admission cost is not explained by instruction count or CFG density
alone. Per-pass instrumentation would be needed to attribute the crossover to
abstract-state propagation, joins, or a particular verifier stage.
The bootstrap interval describes run-to-run uncertainty over these 20 blocks;
it is not an asymptotic complexity proof.

## Validity and scope

- All ten arms have 20 samples; no program was rejected and no cell timed out.
- There were zero major faults. One of 200 cells had wall time above 1.25 times
  process CPU time, below the frozen 10% veto, so `noise_veto` is false.
- CPU affinity remained exactly CPU 23. The `intel_pstate` governor/EPP tuple,
  bpftime source revision/status, and probe path/size/mtime were unchanged at
  both run boundaries.
- The build was `Release` with GCC 13.3.0 on an Intel Core Ultra 9 285K and
  Linux 6.15.11. The probe used bpftime Git revision
  `39d099198938c122e67372d02eaaabd3aaf86436`.
- `CUDA_VISIBLE_DEVICES` was empty, `LD_PRELOAD` was unset, and the harness
  invoked only the CPU verifier API. No GPU execution or device measurement is
  part of this experiment.

This is supporting RQ4 evidence about one-time direct `verify_gpu_program`
admission for accepted synthetic 16--4,096-instruction programs. It is not a
soundness result, GPU/device overhead, attach/JIT/bootstrap latency,
cross-vendor evidence, or a measurement of the 65,536-instruction runtime
boundary. The paper may report the measured boundary, but must not claim
approximately linear admission scaling for all accepted policies.

## Evidence

- Dependency preflight: `raw/preflight-01/`
- Full run: `raw/scaling-575-01/`
- Independent analysis: `raw/scaling-575-01/analysis.json`
- Frozen plan and commands: `plan.md` and `README.md`

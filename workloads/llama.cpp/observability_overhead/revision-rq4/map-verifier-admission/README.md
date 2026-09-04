# Map-program strict verifier admission

This dependency experiment asks one narrow question: how does the current
bpftime GPU verifier classify each of the seven real programs compiled from
`microbench/fig15-device/map_probe.bpf.c`?

The probe uses Prevail's own ELF/BTF reader to apply the object's map
relocations and retain the four map descriptors. Because all seven functions
share one ELF section, it then uses the real `STT_FUNC` offsets and sizes to
slice that relocated instruction stream. Each slice, its program name, and the
complete descriptor table go directly to `verify_gpu_program`. Helper calls
remain the immediates from the ELF and are resolved by the current GPU helper
table.

The ELF reader assigns offline pseudo-fds while preserving map type, key size,
value size, and capacity. Runtime shared-memory fd allocation is therefore out
of scope. This is CPU-only admission evidence: it does not load, attach, JIT,
or execute a policy, and it is not evidence of GPU execution safety.

Build and run without exposing a GPU:

```bash
./build_isolated.sh
make -C ../../../../../microbench/fig15-device .output/map-probe.bpf.o
env CUDA_VISIBLE_DEVICES= \
  /home/yunwei37/workspace/gpu/bpftime-map-verifier-admission-build/map_verifier_admission \
  --object ../../../../../microbench/fig15-device/.output/map-probe.bpf.o
```

The output is one JSON record containing the object/section inventory, map
descriptors, per-program instruction/helper/map inventory, admission result,
and exact rejection text. A minimal accepted program, an unknown-helper
program, a varying-branch program, and an unsupported-GPU-map descriptor are
the positive and negative controls.

Run the source-only tests, or additionally exercise the built probe and real
object:

```bash
python3 -m unittest -v test_map_verifier_admission.py
MAP_VERIFIER_ADMISSION_PROBE=/home/yunwei37/workspace/gpu/bpftime-map-verifier-admission-build/map_verifier_admission \
MAP_VERIFIER_ADMISSION_OBJECT=/home/yunwei37/workspace/gpu/gpu_ext/microbench/fig15-device/.output/map-probe.bpf.o \
CUDA_VISIBLE_DEVICES= \
  python3 -m unittest -v test_map_verifier_admission.py
```

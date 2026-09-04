# STRICT device preflight A0-01 (retained failed attempt)

This fresh RTX 5090 attempt used driver 575.57.08, the verifier-enabled CUDA/LLVM runtime, `BPFTIME_VERIFIER_LEVEL=STRICT`, and the real Table 1 `kernelretsnoop` and `threadhist` BPF objects. The correctness workloads and both instrumentation implementations ran successfully, but the campaign correctly stopped before all timing cells because its verifier-evidence gate returned false.

The failure was in the experiment runner, not in device policy admission. The original gate scanned only `agent.log` and matched a bare target symbol. Runtime admission records are emitted by the target process into `llama_cli.log` and name the attachment as `kretprobe/<symbol>`:

- `kernelretsnoop`: STRICT accepted, 60 BPF instructions, one verified map; 720,896 expected records, 720,896 nonzero timestamps, complete launch/coordinate/multiplicity oracle, and zero drops.
- `threadhist`: STRICT accepted, 13 BPF instructions, one verified map; 720,896 samples, 22,528 nonzero threads, and a complete 1,048,576-entry (8 MiB) readback.
- Native and instrumented application outputs matched the exact 47-byte correctness oracle in all five configurations.
- All recorded cleanup and GPU safety gates passed. No timing cell started, so this attempt contains no performance result and is not relabeled as a successful preflight.

The runner was fixed in the following commit to scan the control and target-process logs, require the exact program and `kretprobe/<symbol>` attachment, deduplicate mirrored records, report scanned and matched sources, reject mixed evidence, and fail closed when evidence is missing. A fresh campaign uses a new output directory; this directory remains immutable failure evidence.

Retained evidence includes the admission record, result and summaries, all correctness logs and safety records, final BPF objects, and build logs. Large generated host binaries and copied dependency build products are deliberately excluded.

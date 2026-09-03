# LMCache 575 execution runbook

Updated 2026-09-03 after the first traced preflight failed before serving and
the tiny Triton diagnostic supported the explicit 12.9 compiler repair. The
`preflight-02` has been launched; it and the subsequent commands below are not
completed results. Root remains the sole GPU launcher. Follow the current addendum in [plan-v2.md](plan-v2.md),
not the closed 610 command interface: traced disk preflight, then **three
untraced correctness cells**, then 30 untraced formal cells. No BPF arm.

## Environment and launch boundary

Work from `/home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk` and use
`./current-venv/bin/python`, not system Python or another workload's venv.
The executable resolves to `/usr/bin/python3.12`; `pyvenv.cfg` records Python
3.12.3 with system packages disabled, and `current-venv/bin/vllm` names this
exact venv in its shebang.

| Component | Installed evidence, inspected without importing |
| --- | --- |
| torch | `torch-2.13.0+cu129.dist-info/METADATA`; static `torch/version.py` says CUDA 12.9 |
| vLLM | `vllm-0.27.1+cu129.dist-info/METADATA`; requires torch 2.13.0; the official cu129 wheel/index is recorded in `artifacts-current.json` |
| LMCache | `lmcache-0.5.4.dist-info/METADATA`; `direct_url.json` names the present local `dist-current-py312/lmcache-0.5.4-cp312-cp312-linux_x86_64.whl` |
| CUDA packages | Installed metadata: toolkit 12.9.1, runtime 12.9.79, NVRTC/nvJitLink 12.9.86, bindings 12.9.7; torch explicitly requires toolkit 12.9.1 |
| Triton assembler on 575 | All three arms explicitly use `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-12.9/bin/ptxas`, version 12.9.86; the default Blackwell assembler is 13.1.80. |

Metadata is under `current-venv/lib/python3.12/site-packages`; versions agree
with [current-requirements.txt](current-requirements.txt). Both frozen wheel
files, `strace`, and the fixed `uv` executable exist. LMCache source revision
`3e11b8ed` is recorded in the manifest, not freshly Git-checked by this audit.
The loaded module version files both say 575.57.08. Ancillary nvdisasm 13.3.73
and NVML Python bindings 13.610.43 are not CUDA-runtime/driver requirements by
themselves. The prior single-prefix success was on 610. The actual 575
diagnostic in `raw/triton-575-diagnostic-01` failed with the default assembler
and passed 4,096 exact outputs with 12.9; see [compatibility-575.md](compatibility-575.md).
This is not full-model or storage evidence. Keep the fixed 0.98 memory budget
and `VLLM_USE_DEEP_GEMM=0`; complete-workload startup with the repair remains
to be tested.

The pin is inside `server_environment`, including `recompute`, not merely in
the launcher environment. Admission records the accessible 12.9.86 binary's
file inventory and version. `environment.json` saves the full server
environment before startup, and `result.json` repeats it after success;
revalidation requires consistent values. Older 610 runs retain their original
unpinned behavior and are never mixed with the 575 comparison.

Wait for the current GPU campaign and let the runner acquire both existing
leases. Launch with CPUs **8–16 available**: the worker/strace gets 8–15 and
telemetry/kernel monitoring gets 16. Do not pin the launcher to CPU 17 or
8–15. No ambient `LD_PRELOAD`, `LD_AUDIT`, or CUDA injection may be present.
The current shared lock files are root-owned; use `sudo -n` for the actual
`run-cell` commands below, without changing their permissions. Offline
validation only needs read access to the saved artifacts.
Set `HF_HOME` explicitly for that root launcher: the prepared model/tokenizer
cache is under the user's existing cache, not root's default cache. Worker
processes already receive this same cache location from the frozen runner.
The runner checks 400 W, exclusivity, driver, port 18080, the frozen ext4 NVMe
mount and at least 100 GiB free. This audit did not perform that admission.

`inspect` is not a lightweight metadata command: it imports LMCache/vLLM and
reads the dataset. `validate-cell`, `compare-outputs`, and `analyze` regenerate
the pinned prompts/tokenizer too. Run these only in the scheduled window;
this preparation imported no CUDA package and read no model or large corpus.

## Preflight and correctness: four separate startups

Run separately and stop on any failure. Preserve the failed
`raw/storage-575-preflight-01/disk`; use the new `preflight-02` directory below,
and never overwrite an existing attempt.

```bash
sudo -n env HF_HOME=/home/yunwei37/.cache/huggingface taskset -c 8-16 ./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --prefix-limit 8 --config lmcache_disk \
  --output raw/storage-575-preflight-02/disk --trace
taskset -c 17 ./current-venv/bin/python -B run_lmcache_disk.py validate-cell \
  raw/storage-575-preflight-02/disk --require-trace

sudo -n env HF_HOME=/home/yunwei37/.cache/huggingface taskset -c 8-16 ./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --prefix-limit 8 --config recompute \
  --output raw/storage-575-correctness-01/recompute
sudo -n env HF_HOME=/home/yunwei37/.cache/huggingface taskset -c 8-16 ./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --prefix-limit 8 --config lmcache_cpu \
  --output raw/storage-575-correctness-01/lmcache_cpu
sudo -n env HF_HOME=/home/yunwei37/.cache/huggingface taskset -c 8-16 ./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --prefix-limit 8 --config lmcache_disk \
  --output raw/storage-575-correctness-01/lmcache_disk
taskset -c 17 ./current-venv/bin/python -B run_lmcache_disk.py compare-outputs \
  raw/storage-575-correctness-01/recompute \
  raw/storage-575-correctness-01/lmcache_cpu \
  raw/storage-575-correctness-01/lmcache_disk
```

The trace must establish successful O_DIRECT writes and reads for all 48
24-MiB cache objects, without buffered `.pt` opens. All eight prefixes need
exact 1536-token stores/retrievals in cache arms, zero external cold hits,
HTTP 200 and 16 output tokens per request. `compare-outputs` fully revalidates
each correctness cell and compares exact cold/warm text. Retain
`execution.json`, `environment.json`, `server.log`, `result.json`, telemetry,
kernel log and preflight traces. None of these four cells enters formal timing.

## Formal sequence and analysis

There is **no existing batch entrypoint**: only `run-cell`, `validate-cell`,
`compare-outputs`, and `analyze` are relevant. The old `run`/`preflight`/`smoke`
subcommands in revision-1 prose no longer exist. Root invokes the existing
one-cell runner sequentially, following [schedule.json](schedule.json).

For every table entry, use output
`raw/storage-575-full-01/attempt-AA/position-P-CONFIG`; `P` is zero-based and
`CONFIG` is the full name in that column. The first exact command is:

```bash
sudo -n env HF_HOME=/home/yunwei37/.cache/huggingface taskset -c 8-16 ./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --prefix-limit 8 --config lmcache_cpu \
  --output raw/storage-575-full-01/attempt-00/position-0-lmcache_cpu
```

| AA | Position 0 | Position 1 | Position 2 |
| --- | --- | --- | --- |
| 00 | lmcache_cpu | recompute | lmcache_disk |
| 01 | recompute | lmcache_disk | lmcache_cpu |
| 02 | lmcache_disk | lmcache_cpu | recompute |
| 03 | recompute | lmcache_disk | lmcache_cpu |
| 04 | lmcache_cpu | recompute | lmcache_disk |
| 05 | lmcache_disk | lmcache_cpu | recompute |
| 06 | lmcache_cpu | lmcache_disk | recompute |
| 07 | lmcache_disk | recompute | lmcache_cpu |
| 08 | recompute | lmcache_cpu | lmcache_disk |
| 09 | lmcache_cpu | recompute | lmcache_disk |

These are 30 cells if all ten blocks succeed. Keep one driver and boot, do not
trace formal cells, and do not shuffle/select favorable results. Technical
failures stay in place with an ordinary nonempty `attempt-AA/failure.md`;
the runner does not write that note automatically. Stop and let root account
for failures before using the remaining scheduled attempts 10–14. Analysis
also requires preserved launch observations for every attempted cell, a
contiguous attempt history, balanced completed positions, and no attempt
after the tenth complete block; a failure note alone cannot replace missing
`environment.json`.

```bash
taskset -c 17 ./current-venv/bin/python -B run_lmcache_disk.py analyze \
  raw/storage-575-full-01
```

This writes `raw/storage-575-full-01/analysis.json` after full revalidation:
10 paired blocks, warm TTFT, sequential request/output-token rates, and
fixed-seed 95% intervals. Fewer valid blocks or engagement failures are not
a completed performance result.

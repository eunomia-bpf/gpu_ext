# disk_uvm_perf

Source checkpoint: not yet built or run. No new performance numbers are
claimed; driver loading and measurements await the active GPU task's handoff.

Small real-performance client for the disk-backed managed-UVM mechanism in
the nvidia-uvm driver built from `gpu_ext-kernel-575-gds` (branch
`revision/gpu-storage-decision-575`). Exercises the REGISTER/OFFLOAD/QUERY ABI
(raw ioctls 84/85/86 on Linux) with real CUDA managed memory and O_DIRECT
disk I/O.

## Build

    make NVCC=/usr/local/cuda-12.9/bin/nvcc CFLAGS="-std=c++14 -O2 -arch=sm_120"

## Run

    ./disk_uvm_perf [--size 256MiB] [--backing-file PATH] [--device 0]
                    [--durability] [--poll-us 200]

- `--size`: managed range, a multiple of the 2 MiB UVM VA block size
  (default 256 MiB).
- `--backing-file`: local file that receives the offloaded data
  (default `disk_uvm_backing.bin`). Opened with O_RDWR|O_CREAT|O_DIRECT
  (+O_DSYNC with `--durability`); the driver's offload worker does
  kernel_read/kernel_write on this exact fd. O_DIRECT-unsupported paths fail
  hard by design; the actual kernel open flags are recorded from
  /proc/self/fdinfo in the raw output.
- `--device`: CUDA device index (default 0).
- `--poll-us`: QUERY poll interval while waiting for offload (default 200).

The client uses the CUDA context's own /dev/nvidia-uvm fd in this process and
never opens /dev/nvidia-uvm itself; run it in a dedicated process with a
single CUDA context.

## Arms and output

All numbers are printed to stdout and saved under `raw/`; the backing file is
preserved on disk after the run.

- baseline_gpu: full GPU read, data GPU-resident (no backing file involved).
- register: attach + seal the whole managed range read-only.
- offload1: write the whole range to the backing file (direct I/O) and
  release the in-memory copies.
- cpu_restore: first-touch host read at the same managed VA; CPU faults
  hydrate chunks from the backing file.
- offload2_release: no file writes (spans already on disk); releases the CPU
  copies restored above.
- gpu_restore: first-touch GPU read; the GPU fault restores via CPU-first
  hydration from the file, then a CPU->GPU copy.
- steady_gpu: GPU read with the data resident again.

Every read arm traverses the full buffer and prints a few individually
sampled words next to the value the deterministic pattern defines at that
index (no aggregate fingerprint). `t_complete_ns` is measured from the
offload t0; `bytes_expected` is the requested byte count, not a measured
transport counter.

## Error semantics (QUERY)

pending=0 is published only after the offload's reclamation outcome is
resolved. Error pages cover both offload I/O failures and failed release of
the in-memory copies (the span keeps its resident copy and is reported
not-on-disk; a later offload rewrites it and retries).

## Exit codes

- 0: all arms completed, sampled words matched, no error pages.
- 1: infrastructure failure (register/query/offload ioctl, allocation).
- 2: adverse result - a sampled word mismatched or error pages were reported
  (raw numbers preserved in `raw/`).
- 3: blocker - no CUDA-context UVM fd, unaligned allocation, or a loaded
  module predating the disk-backing ioctls.

# Matched NVBit observability adapters

These adapters implement the three RQ4 observability tasks on the official
NVBit 1.8 core. They deliberately instrument only the exact mangled kernel in
`OBS_TARGET_SYMBOL` and do not enable related functions.

- `kernelretsnoop`: inject before every `EXIT`, emit one record per logical
  thread through NVBit's device-to-host channel, and count records with nonzero
  `%globaltimer` timestamps on the host.
- `threadhist`: inject before every `EXIT`, atomically update the full
  configured logical-thread array, and report its nonzero entries and total at
  context termination.
- `launchlate`: pass the host CUDA launch-callback timestamp through NVBit's
  per-launch argument and inject one device-entry sample for block/thread zero.
  This is the closest native NVBit counterpart to gpubpf's exact host-stub
  uprobe plus device-entry probe; the different host hook locations are kept
  explicit in the experiment plan.

Build against the pinned release extracted under `revision-rq4/deps`:

```bash
make -C nvbit_adapters/observability \
  CXX=g++ \
  NVBIT_ROOT="$PWD/deps/nvbit_release_x86_64" \
  ARCH=sm_120
```

The revision runner sets `LD_PRELOAD`, `OBS_MODE`, `OBS_TARGET_SYMBOL`, and
`OBS_GPU_THREAD_COUNT`. A run is invalid unless the selected kernel launches
and the task-specific sample checks pass.

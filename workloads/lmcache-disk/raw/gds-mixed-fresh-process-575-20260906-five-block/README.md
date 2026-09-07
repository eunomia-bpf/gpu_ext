# Fresh-process mixed cuFile comparison

Five cyclically rotated repetitions of FIFO, native and BPF. Each measurement
runs in a separate Python process to release the CUDA context and 4096 MiB
staging pool. Each uses 64 demand reads and 96 background writes of 24 MiB,
with the runner's unchanged 2 ms read / 4 ms write arrival spacing and
controlled pressure 801 permille / slack 10 ms.

Source: the main gpu_ext checkout's run_gds_mixed_backend.py from commit
10cc867d. The implementation and policy are unchanged. Invocation imports
that runner, sets CONFIGS to the single selected configuration, and invokes
main with --blocks 1 --reads 64 --writes 96 --gds-buffer-size-mib 4096.
The current-venv Python executable belongs to gpu_ext-lmcache-gds-control.

Outer directories retain repetition and order. Each inner campaign has one
measurement and consequently internally reports block=0, position=0.
Orders: FIFO/native/BPF; native/BPF/FIFO; BPF/FIFO/native; FIFO/native/BPF;
native/BPF/FIFO. No failed measurement is retried or silently replaced.

The existing offer_s field records actual call dispatch, not the scheduled
arrival. Report these measurements as call-to-completion latency, not full
scheduled-arrival latency. The timing-field refinement remains a separate
local-model task. The backend performs real cuFile compatibility-mode I/O;
this does not establish hardware NVMe-to-GPU P2P. All earlier records remain.

Collection completed 2026-09-06: all 15 processes returned zero, all 960 reads
and 1440 writes completed, with no cleanup errors. See the top-level
summary.json and results-575-gds-mixed-fresh-process-20260906.md two levels up
for complete performance values, paired changes, and interpretation.

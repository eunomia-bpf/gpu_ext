# Mixed cuFile burst follow-up

This follow-up keeps the preceding fresh-process experiment's 64 demand reads,
96 background writes, 24 MiB objects, 4096 MiB pool, FIFO/native/BPF policies,
and five cyclic rotations. It changes only both requested inter-arrival
intervals to zero: --read-stagger-ms 0 --write-stagger-ms 0. This asks whether
the existing one-shot 10 ms write deferral helps a short demand-read burst,
as opposed to the prior 128 ms read / 384 ms write offered stream. No winning
configuration is selected from the preceding repetitions.

Run invocation and source are identical to the preceding collection: main
gpu_ext run_gds_mixed_backend.py from commit 10cc867d, imported by the
development worktree's current-venv Python, CONFIGS set to the single selected
configuration, --blocks 1 --reads 64 --writes 96 --gds-buffer-size-mib 4096,
plus the two zero-spacing options above. Each measurement has its own process.
All 15 measurements are retained, without retry or exclusion.

These are call-to-completion durations: offer_s is dispatch, not scheduled
arrival. Threads are launched by the unchanged common executor, so zero
spacing does not imply physically simultaneous CUDA calls. cuFile uses the
compatibility path, not demonstrated hardware NVMe-to-GPU P2P. The same
controlled pressure 801/slack 10 ms inputs apply, not measured live pressure.

Orders: FIFO/native/BPF; native/BPF/FIFO; BPF/FIFO/native; FIFO/native/BPF;
native/BPF/FIFO. Outer directory names carry repetition and position; inner
single-measurement campaigns each label their block and position as zero.

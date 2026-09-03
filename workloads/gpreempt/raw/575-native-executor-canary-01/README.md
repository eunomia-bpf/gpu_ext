This is an executor correctness canary, not a comparative timing cell.

The original config A ran for 60 seconds using the exported VGG19/ResNet152
models on loaded driver e3bb2938. Each task completed 6,000 timed requests and
110 original warmup/standalone-calibration requests; every output contained
1,000 checked FP32 values, with maximum absolute error zero. The runner retained
the original six-stage latency samples and telemetry, but concurrent CPU builds
were allowed. Do not reuse these samples as a contention-free performance block
or as performance of a subsequently loaded driver. No BPF policy was attached.

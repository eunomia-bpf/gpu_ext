# STRICT device preflight A0-02 (intermediate passing attempt)

This fresh RTX 5090 / driver 575.57.08 run used the verifier-enabled CUDA/LLVM runtime and the real `kernelretsnoop` and `threadhist` Table 1 objects. All five correctness cells and the single randomized pp32 timing block passed the runner active at launch (`0a036a3`). No cell was rejected or retried, and all process, shared-memory, GPU-safety, and restoration gates passed.

The stricter target-bound parser added after this process started was replayed read-only over all four gpubpf target logs. Each log independently passes: its admission record is bound to the PID in the corresponding execution record, contains exactly one STRICT acceptance and exactly one expected map, and contains no skip, reject, foreign-PID, unexpected, or unparsed verifier record. The observed objects were:

- `kernelretsnoop`: 60 instructions; type 1527/key 4/value 32; max entries 256 for correctness and 44 for timing.
- `threadhist`: 13 instructions; type 1502/key 4/value 8/max entries 1 in both cells.

The one-block pp32 rates were baseline 7,099.88, gpubpf exit recording 132.33, matched NVBit exit recording 134.91, gpubpf histogram 5,331.84, and matched NVBit histogram 6,662.24 token/s. These are preflight diagnostics, not paper performance estimates; the paper-facing rows remain the existing ten-block pp512 campaign.

Because this process began before the final PID/map/cardinality gate was installed, it is retained as an intermediate passing attempt rather than used as the final A0 gate. A new output directory is used for the final run.

# Disk-UVM restore bring-up, 2026-09-08

Driver c8e2831d and the sm_120 client were built successfully. The client
needed the single CUDA error-enum cast fix in main 47646ec3. Failed and
successful build logs are retained. Backing payloads and compiled modules
are not repository artifacts.

The first live attempt in `repeat-01/client.log` exited 3 before performance
timing because its FD finder rejected multiple CUDA-owned UVM descriptors.
This is not a disk performance result. `lifecycle.log` records restoration
of the saved original UVM and the GDS/KV loaders with RESTORATION_OK=1.
The first privileged flock-open attempt failed before any module mutation;
the working command acquires both experiment leases as the ordinary user
and runs the lifecycle script as a privileged child.

The next `fd-selection-retry.e5NtEs/` attempt completed five processes after
repairing selection of the descriptor owning the managed allocation. Its
GPU traversal was compiled away, so those GPU timings are not full-buffer
restore measurements. After the one-line volatile-source fix, the fresh
`full-read-retry.4Nw61Y/` run completed all five full-read repetitions and
restored the original UVM/loaders. See the workload report
`results-disk-uvm-restore-20260908.md` for results and scope. All original
attempts and output remain.

Scope: explicit registered read-only file-backed UVM allocation, real
O_DIRECT file I/O and fault-time CPU staging. This is not established
NVMe-to-GPU P2P or an end-to-end LMCache result, and does not replace the
existing LMCache FIFO/native/BPF performance comparisons.

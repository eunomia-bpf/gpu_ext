# Disk-UVM GPU promotion follow-up

Driver `dea1fefc` is built and pushed. It adds default-off per-range GPU
promotion to the existing disk-backed UVM primitive: fault-time disk hydration
uses CPU staging, followed by the normal copy to the faulting GPU. This is
not direct NVMe-to-GPU P2P and is not a measured end-to-end LMCache policy.

Root integrated the local Qwen patch and its early-return repair. The first
build selected only nvidia-uvm and failed modpost on NVIDIA core symbols;
`build-driver.log` retains that attempt. Retrying the established full
`make -j2 modules` command succeeded (`build-driver-retry.log`). No module
from either build has been loaded at this checkpoint. Large modules and
backing files stay outside Git.

The existing client opt-in is still being implemented. The next run preserves
the earlier 256 MiB, seven-stage full-read order and five fresh processes,
with GPU promotion enabled. Prior CPU-first results remain in
`../disk-uvm-restore-20260908.cxYKb4/full-read-retry.4Nw61Y/`; they are not
repeated or replaced. Compare stage times as separate campaigns, not as
randomized paired samples. No new result or HBM bandwidth claim is made yet.

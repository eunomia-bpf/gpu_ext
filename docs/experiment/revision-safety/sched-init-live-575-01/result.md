# Scheduler-init lifecycle attempt 01

Status: **admission rejected; no module or service mutation occurred**.

The lifecycle coordinator acquired both existing read-only leases, then rejected
the admitted NVIDIA core while parsing the C-form BTF declaration for
`bpf_nv_gpu_preempt_tsg`. The parser expected one exact textual rendering of
the function signature. The failure occurred before candidate staging,
service stop, module removal, native execution, or matrix execution.

The retained `lifecycle.json` reports `destructive_started: false`,
`candidate_insert_started: false`, and an empty `candidate_loaded` list. The
immediate post-attempt audit observed Linux 6.15.11, NVIDIA 575.57.08, 15 MiB
GPU memory use, 0% GPU utilization, a 400 W power limit, active GDM and NVIDIA
persistence services, UVM reference count zero, and no attached struct_ops.

This attempt is not an experiment result and contributes zero completed cells.
The next attempt must make the BTF signature parser accept bpftool's equivalent
C rendering without weakening the raw-BTF type and declaration-tag checks.

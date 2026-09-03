# Unprivileged live-BTF check

This check was run by the primary agent as UID 1000, without `sudo`, before any
module or service mutation:

```text
/sys/kernel/btf/nvidia mode=444 owner=root group=root size=125993
/sys/kernel/btf/nvidia_uvm mode=444 owner=root group=root size=268940
bpftool btf dump file /sys/kernel/btf/nvidia_uvm format raw: exit 0
```

The full read-only coordinator preflight also called both `capture_core()` and
`live_uvm_interface(diagnostic=False)` as UID 1000 and completed successfully.
Its machine-readable result is attached separately as
`root-readonly-preflight.json`.

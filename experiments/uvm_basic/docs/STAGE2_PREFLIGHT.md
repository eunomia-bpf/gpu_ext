# gpu_ext Stage 2 Preflight

Status: `READY_FOR_ROOT_TRACE_ATTACH`.

- Distribution `nvidia_uvm` loaded: `False`
- Custom `nvidia_uvm` loaded: `True`
- Custom hook symbols visible: `True`
- Loaded module candidate: `/lib/modules/6.15.11-gpuext-gpuext/updates/dkms/nvidia-uvm.ko`
- Loaded runtime srcversion: `2A011BD52759A63796A0B00`
- Custom module: `/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko`
- Custom version/kernel match: `True`
- Loaded/custom dependency: `nvidia` / `nvidia`
- All five extension binaries ready: `True`

- UVM-only static ABI status: `SUPPORTED_BY_STATIC_MODVERSION_EVIDENCE`
- Module-switch recommendation: `TRY_UVM_ONLY_SWITCH_FIRST`

No root operation, module switch, BPF attach, or policy execution was performed by this preflight.

The privileged `bpftool prog show` and `bpftool struct_ops list` checks are provided only in `scripts/SAFE_GPU_EXT_COMMANDS.sh`.

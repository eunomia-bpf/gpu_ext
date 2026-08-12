# UVM-only Module Compatibility

Status: `SUPPORTED_BY_STATIC_MODVERSION_EVIDENCE`.

Recommendation: `TRY_UVM_ONLY_SWITCH_FIRST`.

- Shared UVM required symbols: 322; CRC mismatches: 0
- Shared `nvUvmInterface*` symbols: 76; CRC mismatches: 0
- Shared distribution/custom `nvidia.ko` exports: 93; CRC mismatches: 0
- Custom-only required kernel symbols: 5; all kernel CRCs match: True
- Custom-only `nvidia.ko` exports required by custom UVM: 0

This supports trying the UVM-only switch before a full-stack switch. It is not runtime load proof; unresolved-symbol and kernel-log checks remain mandatory during the manual maintenance window.

The manual UVM-only action checks only active CUDA/UVM users and the `nvidia_uvm` use count. It intentionally keeps the loaded `nvidia`, `nvidia_modeset`, and `nvidia_drm` stack in place; full-stack switching retains stricter display and all-device-user checks.

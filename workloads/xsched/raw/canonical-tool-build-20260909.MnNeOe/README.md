# Canonical XSched host-tool reconciliation build — 2026-09-09

The canonical `level2/tool/xsched_guard_tool.cu` now registers each launch's
context value, including zero/disarm, after instrumentation and enables the
instrumented function on every launch. The native-C/BPF decision and shared
device actuator are unchanged. A direct source diff against the tested
`xsched_guard_tool_callback.cu` contains only the latter's seven-line candidate
comment; executable source is identical.

Root compiled the frozen canonical source through the existing
`level2-build/Makefile` `guard-tool` target. Both shared experiment leases
covered compilation. The target used the unchanged XG5 carrier object from
`../level2-xg5-build-20260909.yiBj3W/`; `-o` tells make to reuse that object,
not rebuild the device/exporter pipeline. Build exit status was zero at
08:58:25 PDT. The local shared library is 3,153,248 bytes. Existing unused
function warnings remain in `build.log`.

Published here: the exact source input in `source/tool/`, `build.sh` and
`build.log`. Compiled objects/libraries are regenerable local outputs and
are not committed. The script is the original machine's build record, not a
portable all-dependencies installer; the referenced CUDA, NVBit and XG5
carrier must exist or be rebuilt from their recorded inputs.

No library was installed, no driver was changed, and no GPU performance cell
was rerun. The [completed five-block comparison](../level2-device-policy-pair-20260909.buBjns/README.md)
used the previously built callback-reference library, not this new library.
This is a successful canonical host-component rebuild, not new performance
evidence or reproduction of the original cuXtra artifact on sm_120.

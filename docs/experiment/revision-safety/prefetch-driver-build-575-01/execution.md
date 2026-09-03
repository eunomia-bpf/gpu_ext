# Prefetch diagnostic driver build 575-01

Result: **source implementation, compilation, unit tests, BTF, and static
function-entry inspection passed. The module has not been loaded and no live
fallback control has run.**

## Exact source and scope

The source is sibling repository `gpu_ext-kernel-575`, branch `test-sched`,
revision `0c109956`. The commit changes only:

- `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h`;
- `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c`;
- `kernel-open/nvidia-uvm/uvm_perf_prefetch.c`.

It adds a driver-filled scalar context, a void/const-pointer noinline hook,
SELECTED and FINISHED calls, and a native-loop iteration counter. The context
has no kernel pointer or address-derived field. Policy dispatch, validator
calls, effect selection, branch conditions, translations, and the return value
remain unchanged. The diagnostic adds functional overhead and is barred from
performance measurements.

The source repository was otherwise clean except for two pre-existing generated
unit-test binaries listed in [`source-status.txt`](source-status.txt). They were
not committed.

## Build and tests

The exact build command was:

```sh
taskset -c 0-7 make modules -j8 \
  KERNEL_UNAME=6.15.11-061511-generic CC=/usr/bin/gcc-14
```

[`build.log`](build.log) records exit 0, all five modules linked, and BTF
generation for `nvidia-uvm.ko`. Its only warnings are the existing modpost
`MODULE_DESCRIPTION()` warnings for four NVIDIA modules. No compile, link, or
BTF error is present. [`artifact-inventory.txt`](artifact-inventory.txt)
records paths, sizes, and modification times; [`module-info.txt`](module-info.txt)
records version 575.57.08 and the 6.15.11 module ABI without content-derived
identifiers.

`taskset -c 19 make -C kernel-open/tests test` also exited 0. The retained
[`transition-tests.log`](transition-tests.log) reports 12 cases and 145
assertions passing, including prefetch action, action/region routing, region
width, and translation.

## Static admission evidence

[`hook-btf.txt`](hook-btf.txt) records an 88-byte, 14-field
`uvm_bpf_prefetch_diagnostic_ctx` and the function prototype returning void
with one const-context pointer. The listed fields are copied action, request,
bounds, output, phase, validation/effect, and native traversal values; no
address field is present.

[`hook-disassembly.txt`](hook-disassembly.txt) shows a retained global
`uvm_bpf_prefetch_diagnostic` body beginning with a real `__fentry__` call.
The built module also has two calls from `compute_prefetch_region`, one after
effect selection and one before the existing single return.

These facts make live fentry admission plausible but do not prove it. The next
step is to replace the failed structure-return observer with the new diagnostic
observer, pass its offline gates, stage this exact module, and attempt a real
load/attach before releasing any target.

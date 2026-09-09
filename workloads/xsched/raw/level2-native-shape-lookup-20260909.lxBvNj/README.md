# Native Level-2 layout lookup candidate: build succeeded, workload failed

## Execution result — 2026-09-09 04:05 PDT

After LMCache released both leases, the isolated install build completed.
The six-process native cell failed before BE execution reached the running
event: be1 exited 1 and be2/be3/be4 exited -11 (SIGSEGV). The runner exited 1.
Raw process JSON and failure.json are retained under cells/block-01-native_blob;
run.log includes the completed build and runner traceback. No native Level-2
performance result is available from this candidate. The exact new crash
location has not yet been localized; it must not be inferred from the previous
701/4 failures. The source remains efb36b2c, with no in-run modifications.

## Original queued record

Root queued run.sh behind the current LMCache campaign using both shared
GPU/struct-ops leases. It has not yet built or run at this start record.
The local model's candidate is published as efb36b2c; its isolated source
is hal-native-20260908.i8na51/source and is frozen for this execution.

The candidate first looks up the exported registry helpers through
RTLD_DEFAULT, retains the named-handle fallback, and adds a loaded
parameter-shape fallback when kernel-name lookup misses. It preserves
the prior initialized allocation layout and default-visible exports.
The reported symbol/load-order and name-mismatch explanations remain
hypotheses for the actual runtime failure, not established causes simply
because the code was changed or CPU specimens passed.

run.sh builds and installs the isolated native HAL, then executes one
existing six-process native-blob cell with the original workload
dimensions and metadata flags. It does not rebuild the tool HAL or
shared bpftime runtime and does not change the driver. Success requires
the actual native workload to run; no completed Level-2 reproduction or
performance improvement is claimed by this queued record.

The previous 701/4 failures, host constructor SIGBUS and failed tool
observations remain unchanged. No manuscript is edited.

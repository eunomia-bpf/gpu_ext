# Device-trampoline preflight attempt 02

Status: **stopped before any arm was accepted**.

The native application again completed its correctness workload, but its arm
was rejected because UVM reference count stayed at 4 for the enlarged
120-second post-run window. The runner's outer finalizer then stopped the
campaign-wide `nvidia-smi` telemetry process and immediately recorded UVM
reference count zero, with no cleanup error.

A separate idle-GPU diagnostic reproduced the lifecycle exactly: the same
continuous `nvidia-smi` query held UVM reference count 4, and stopping that
exact process returned it to zero. Thus attempt 01 was not a slow CUDA-context
release, and lengthening the timeout cannot solve the gate conflict. The next
revision must make telemetry lifetime compatible with the per-arm zero-UVM
gate while retaining all samples and safety checks. This attempt contains no
valid trampoline performance record and must not be pooled with later runs.

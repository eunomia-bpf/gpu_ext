# Preserved historical Python prototype

`policy.py` (6248 bytes) and `test_policy.py` (10266 bytes) were found untracked during the
2026-09-08 workspace cleanup. They are preserved on this historical branch,
not integrated into the active native/BPF LMCache storage implementation.

The Python request layout and policy precedence must not be assumed to match
the currently installed driver ABI. The Makefile does not build or invoke this
prototype, and no formal performance result is attributed to it. This cleanup
does not certify its behavior or introduce another experiment. Original files
are retained without changes; their tests were not run as part of archival.

The active implementation and results remain on the gpu_ext master branch.

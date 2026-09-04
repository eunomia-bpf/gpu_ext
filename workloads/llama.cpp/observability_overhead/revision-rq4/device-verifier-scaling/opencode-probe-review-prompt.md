Review the attached device-verifier probe and isolated build read-only. Do not
invoke tools, shell, edits, web, or subagents.

Check concrete correctness blockers only: frozen constructors and structure
checks; legal helper/branches; `--describe` making zero verifier/clock calls;
accept-only and timed modes making exactly one real `verify_gpu_program` call;
construction outside the timed interval; affinity and diagnostic correctness;
JSON/exit behavior; Release/source-revision provenance; and strict separation
from existing bpftime build trees and GPUs. The plan was already source-reviewed.

End with exactly `VERDICT: PASS` or `VERDICT: FAIL`.

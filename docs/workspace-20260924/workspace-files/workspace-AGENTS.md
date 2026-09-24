# Workspace instructions

- Never generate, refresh, compare, or record file/content hashes, checksums,
  fingerprints, or digests, and never use them as experiment, implementation,
  review, publication, provenance, or resume gates.
- Use normal Git commits, upstream source revisions, explicit file inventories
  and sizes, patch-application checks, build inspection, tests, and real
  correctness or engagement runs instead.
- Git commit IDs are ordinary commit and push bookkeeping and are not
  prohibited.
- Do not modify vendored third-party code merely because that dependency uses
  internal integrity checks. Do not invoke or refresh those checks as project
  evidence.
- Do not stop a local OpenCode session merely because it has not emitted output
  or written files recently, and do not impose an artificial short timeout on
  it. Let it continue until the assigned task completes or reports an actual
  error; terminate it only with concrete evidence that it is irrecoverably
  stuck.
- Run at most three local OpenCode sessions concurrently. When three-way
  parallelism is useful, prefer one session each using Qwen 27B, Qwen Next, and
  GLM Next instead of duplicating one model.
- The root agent plans and coordinates work, reviews results, runs experiments,
  and commits and pushes completed changes. It may directly perform very simple,
  clearly bounded edits or commands; delegate non-trivial project-code
  development to the local models through OpenCode.

# OpenCode endpoint lifecycle review request

- Date: 2026-09-04 (America/Vancouver)
- Model: `spark-gateway/qwen3.8-27b-nvfp4-200k`
- Mode: `--pure`, attached-file review, no tools, no GPU execution
- Permissions: all denied; write/edit/bash/webfetch/task disabled

Review the attached endpoint module lifecycle wrapper, launchlate runner lease
handoff, CPU-only tests, and operator note. Return `PASS` or `BLOCKER`, then at
most five concrete findings. Focus on unconditional restoration of the exact
known-good four-module stack after any post-removal failure or interrupt,
partial service/label recovery, correct completion publication after lease
release, endpoint source/export and 200/200 output gates, and the fixed
preflight-to-full child campaign while the endpoint module is still loaded.
Check that the child cannot become an arbitrary callback, receives both held
read-only lease descriptors without reacquisition, retains argv/stdout/stderr/
return code and independent-analysis results, and that a requested child is
required for wrapper completion. Do not treat intentionally pending GPU
execution as a code defect. Do not suggest weakening a safety or evidence gate.

# Read-only SPIR-V failure-closure audit

Act as an independent systems-artifact reviewer. Inspect every attached text
file. Do not use tools, edit files, run commands, or assume facts absent from
the attachments.

Review the retained RTX 5090 / driver 575.57.08 attempt-01 failure and the
runner/analyzer repair. Check only blockers that could make these conclusions
false:

1. attempt 01 generated a structurally validator-accepted SPIR-V module, but
   `clCreateProgramWithIL` returned `-59` before program build, kernel creation,
   or kernel execution;
2. host generation/validation is never reported as GPU execution;
3. future attempts query and retain `CL_DEVICE_IL_VERSION`,
   `CL_DEVICE_ILS_WITH_VERSION`, and the extension inventory before starting
   the demo, require SPIR-V to be explicitly advertised, and fail closed with
   no PTX/CPU fallback;
4. the analyzer independently distinguishes complete success, a new
   pre-execution unsupported-capability result, and the retained legacy
   `-59` result; and
5. attempt 01 remains a negative capability boundary, not evidence for a
   gpubpf attach backend, cross-vendor portability, or performance.

The Khronos contract supplied in the report is that OpenCL 3.0 need not support
SPIR-V, empty IL queries mean no IL-program support, and
`clCreateProgramWithIL` returns `CL_INVALID_OPERATION` (`-59`) when no device in
the context supports IL programs. Do not request broader experiments or
cosmetic changes. Identify concise blocking defects only.

End with exactly one line:

`VERDICT: PASS`

or

`VERDICT: FAIL`

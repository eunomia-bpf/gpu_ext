# OpenCode/Qwen review record

OpenCode 1.18.27 was invoked in `--pure`, deny-all mode with the local
`spark-gateway/qwen3.8-27b-nvfp4-200k` model. The C++ probe, CMake file, and
Python test were attached for a bounded read-only correctness review.

The corrected invocation produced no event, text, session finding, or verdict
before its 300-second timeout. The earlier invocation put the message after the
file options, so OpenCode interpreted that message as a file path and exited;
it likewise produced no review. Neither attempt is counted as a PASS, and no
further retry was made. Confidence in the result instead rests on the isolated
build, direct real-ELF run, explicit controls, and the four passing tests.

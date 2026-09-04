# Review outcome

Accepted actions:

- preserve the failed `full-575-01` campaign rather than selecting its seven
  passed cells;
- report the BPF object path, libbpf error, saved `errno`, and error text;
- capture shared-segment identity only while the live probe demonstrably holds
  the exact object;
- keep replacement and unknown-object cleanup fail closed; and
- retain the no-within-run-retry rule.

Local verification after the patch: the sm_120/C/BPF build passed, all 15 CPU
tests passed with resource warnings promoted to errors, the dry-run passed,
and `git diff --check` passed.

The review does not turn the failed campaign into evidence and does not claim a
root cause that the old diagnostics cannot establish.

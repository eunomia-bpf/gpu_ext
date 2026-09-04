# STRICT uniform-map preflight 575-01: analyzer-record failure

The real six-arm RTX 5090 process sequence completed and the online runner
accepted its execution, strict-admission, map-effect, attach, and cleanup
checks. Independent replay then stopped before accepting any campaign result:
the runner wrote the observed return code into the execution TSV header where
the fixed column name `returncode` was required.

This is a raw-record serialization defect. No timing or strict-execution claim
is taken from attempt 01, and its files remain unchanged under
`raw/strict-uniform-map-preflight-575-01/`. The header writer and a source
contract test were repaired before starting a fresh complete preflight in a
new directory. The map workload, correctness oracle, strict gate, schedule,
timed region, and analysis were not changed.

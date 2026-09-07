# Retained startup failure; no performance samples

All 25 child invocations returned 2 before storage setup/measurement.
The root launcher created each cell directory to open its log there, while
`run_cell` deliberately creates that directory with `exist_ok=False`.
Each result records the resulting FileExistsError, empty requests and no
performance metrics. The policy-budget implementation was not the cause.

The repaired launcher writes startup logs in the parent block directory,
lets the existing runner create the cell, and moves the log after exit.
The unchanged 25-cell matrix proceeds in the sibling
`gds-write-budget-575-20260907-five-block-02` directory. No successful cell
was retried or overwritten; all failed outputs are retained here.

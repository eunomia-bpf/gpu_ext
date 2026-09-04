Review the attached runner, independent analyzer, and tests read-only. Do not
invoke tools, shell, edits, web, or subagents.

Check concrete correctness blockers only: exact 200-cell schedule; fresh
processes; environment/affinity/source gates; timeout/no-retry and incremental
failure retention; raw record binding; independent schedule/shape replay;
median, paired-ratio, block-bootstrap, Theil--Sen, noise-veto and verdict logic;
and whether malformed or missing evidence can pass or crash analysis. Confirm
preflight cannot be promoted to a result. The plan was already source-reviewed.

End with exactly `VERDICT: PASS` or `VERDICT: FAIL`.

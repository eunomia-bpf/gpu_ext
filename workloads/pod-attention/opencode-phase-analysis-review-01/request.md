# POD phase-analysis read-only review request

Review `analyze_phase_study.py` and `test_analyze_phase_study.py` without
editing files or running GPU, sudo, module, or Git operations. Check that:

1. incomplete or mismatched preflight/formal records fail closed;
2. all 15 fresh-process cells and five matched blocks are revalidated;
3. phase summaries, operator output, safety state, and telemetry are checked
   against retained raw records;
4. paired ratios and uncertainty preserve whole-block pairing; and
5. `pre_python_main` and `steady_samples` are not overstated as generic attach
   latency or end-to-end serving performance.

Return `READY` only if no source-level blocker remains.

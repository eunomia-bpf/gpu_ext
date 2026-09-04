READY — the blocker is closed. `run_study.py:314-322` now captures
`segment_identity` inside the readiness poll while the loader is alive, the
post-loop capture (`323-324`) only fires on the ready-string path, and the
`finally` segment block (`372-383`) runs independently of the loader-close
guard and refuses removal when `identity is None` instead of reconstructing
it. Tests cover the previously-missing paths: loader-dies-after-creating-
segment (`test_loader_early_failure...`), loader-close failure still removing
its captured segment (`test_loader_close_failure...`), and preservation of a
not-yet-identified replacement (`test_cleanup_never_adopts...`). First-launch
timestamp stability is now asserted in `test_launch_bridge.cpp`.

Only the optional matrix/wrong-mode-mutation CPU coverage note remains; it is
not a reservation blocker.

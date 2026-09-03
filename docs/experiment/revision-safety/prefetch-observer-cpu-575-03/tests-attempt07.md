# Offline test attempt 07

The first regression-test edit placed the second missing-tail assertion after
the next test method, causing an `IndentationError` during collection. No test
and no GPU action ran. The assertion was moved back into its lifecycle test
before the suite was rerun.

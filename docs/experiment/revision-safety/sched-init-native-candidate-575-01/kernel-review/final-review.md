**READY**

The placement gate now closes the only prior blocker. Verified against the
actual source:

- `validator` (373) -> `validated_phase` (392) -> first emit (395): VALIDATED
  is strictly after `nv_gpu_transition_validate_scheduler`.
- Each NATIVE_RETURN is ordered setter -> status capture -> phase -> emit ->
  `NV_ASSERT_OK_OR_GOTO` (404--409 and 418--424), with `bPolicy*` committed
  only afterward.
- Six emit sites and three CONSTRUCTOR_RETURN phase sites pin the failure,
  lock-reacquire early-return, and success paths. The former two disable the
  active flag; the latter reaches the terminal return.
- The reserve-failure ordering covers both the lock-failure early return and
  the successful lock-retry path back through `failed:`.
- The hook remains one `noinline void`, barrier-only definition.
- The semantic source test is phony and part of the CPU `test` target.

No remaining fixes.

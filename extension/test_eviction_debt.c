/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
/* Offline state-machine tests for the migration-debt eviction policy;
 * no BPF/GPU interaction. */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t s32;
typedef int64_t s64;

#define DEBT_INLINE static inline
#include "eviction_debt_model.h"

static unsigned int tests_run;

static void test_activate_samples_warm_flag(void)
{
	struct debt_chunk_state warm, cold;

	debt_activate(&warm, 1234, 1);
	assert(warm.owner_pid == 1234);
	assert(warm.debt == 0);
	assert(warm.accesses == 0);
	assert(warm.disk_durable == 1);

	debt_activate(&cold, 5678, 0);
	assert(cold.owner_pid == 5678);
	assert(cold.disk_durable == 0);
	tests_run++;
}

static void test_prepare_marks_before_cap(void)
{
	struct debt_chunk_state state;
	u64 delta;
	int i;

	debt_activate(&state, 1, 0);
	for (i = 0; i < 3; i++) {
		assert(debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK);
		assert(delta == 1);
		assert(state.debt == (u8)(i + 1));
	}
	tests_run++;
}

static void test_prepare_caps_and_flags_low_reuse(void)
{
	struct debt_chunk_state durable, plain;
	u64 delta;

	debt_activate(&durable, 1, 1);
	debt_activate(&plain, 2, 0);

	while (debt_prepare(&durable, 4, &delta) == DEBT_PREPARE_MARK)
		;
	assert(durable.debt == 4);
	/* Terminal VICTIM call leaves the ledger unchanged. */
	assert(delta == 0);

	while (debt_prepare(&plain, 4, &delta) == DEBT_PREPARE_MARK)
		;
	assert(plain.debt == 4);

	/* At the cap the chunk is low-reuse; no state change. */
	assert(debt_prepare(&durable, 4, &delta) == DEBT_PREPARE_VICTIM);
	assert(delta == 0 && durable.debt == 4);
	assert(debt_prepare(&plain, 4, &delta) == DEBT_PREPARE_PENDING);
	assert(delta == 0 && plain.debt == 4);
	tests_run++;
}

static void test_reuse_clears_debt_and_saves(void)
{
	struct debt_chunk_state state;
	u64 delta;

	debt_activate(&state, 1, 0);
	assert(debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK);
	assert(debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK);
	assert(state.debt == 2);

	/* A later observed reuse reduces/clears the debt and saves. */
	assert(debt_access(&state, 4, &delta) == DEBT_ACCESS_SAVE);
	assert(delta == 2);
	assert(state.debt == 0);
	assert(state.accesses == 1);

	/* After the save the chunk can be marked at risk again. */
	assert(debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK);
	assert(state.debt == 1);
	tests_run++;
}

static void test_reuse_keeps_chunk_without_debt(void)
{
	struct debt_chunk_state state;
	u64 delta;

	debt_activate(&state, 1, 0);
	assert(debt_access(&state, 4, &delta) == DEBT_ACCESS_KEEP);
	assert(delta == 0 && state.debt == 0 && state.accesses == 1);
	assert(debt_access(&state, 4, &delta) == DEBT_ACCESS_KEEP);
	assert(state.accesses == 2);
	tests_run++;
}

static void test_reuse_holds_durable_low_reuse_candidate(void)
{
	struct debt_chunk_state state;
	u64 delta;

	debt_activate(&state, 1, 1);
	while (debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK)
		;
	assert(state.debt == 4);

	/* Reuse of a low-reuse disk-durable chunk does not save it. */
	assert(debt_access(&state, 4, &delta) == DEBT_ACCESS_HOLD);
	assert(delta == 4 && state.debt == 0 && state.accesses == 1);

	/* It is at risk again from the next candidate observation. */
	assert(debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK);
	tests_run++;
}

static void test_reuse_saves_non_durable_low_reuse_candidate(void)
{
	struct debt_chunk_state state;
	u64 delta;

	debt_activate(&state, 1, 0);
	while (debt_prepare(&state, 4, &delta) == DEBT_PREPARE_MARK)
		;
	assert(state.debt == 4);

	assert(debt_access(&state, 4, &delta) == DEBT_ACCESS_SAVE);
	assert(delta == 4 && state.debt == 0);
	tests_run++;
}

static void test_cleanup_releases_remaining_debt(void)
{
	struct debt_chunk_state state;
	u64 delta;

	debt_activate(&state, 1, 1);
	assert(debt_cleanup_delta(&state) == 0);
	while (debt_prepare(&state, 3, &delta) == DEBT_PREPARE_MARK)
		;
	assert(debt_cleanup_delta(&state) == 3);
	tests_run++;
}

static void test_default_debt_max(void)
{
	struct debt_chunk_state state;
	u64 delta;
	int marks = 0;

	assert(debt_effective_max(0) == DEBT_DEFAULT_MAX);
	assert(debt_effective_max(9) == 9);
	assert(debt_effective_max(300) == 255);

	debt_activate(&state, 1, 0);
	while (debt_prepare(&state, 0, &delta) == DEBT_PREPARE_MARK)
		marks++;
	assert(marks == DEBT_DEFAULT_MAX);
	assert(state.debt == DEBT_DEFAULT_MAX);
	tests_run++;
}

static void test_prefetch_gate(void)
{
	assert(!debt_suppress_prefetch(100, 0));  /* gate disabled */
	assert(!debt_suppress_prefetch(0, 8));
	assert(!debt_suppress_prefetch(7, 8));
	assert(debt_suppress_prefetch(8, 8));
	assert(debt_suppress_prefetch(9, 8));
	assert(!debt_suppress_prefetch(0, 1));
	assert(debt_suppress_prefetch(1, 1));
	tests_run++;
}

static void test_pressure_ledger_closes(void)
{
	/* Three chunks with debts 3, 2, 1: pressure must equal the sum. */
	struct debt_chunk_state a, b, c;
	u64 delta, pressure = 0;

	debt_activate(&a, 1, 0);
	debt_activate(&b, 2, 1);
	debt_activate(&c, 3, 0);

	while (a.debt < 3)
		pressure += (delta = 1, debt_prepare(&a, 8, &delta), delta);
	while (b.debt < 2)
		pressure += (delta = 1, debt_prepare(&b, 8, &delta), delta);
	while (c.debt < 1)
		pressure += (delta = 1, debt_prepare(&c, 8, &delta), delta);
	assert(pressure == 6);

	assert(debt_access(&b, 8, &delta) == DEBT_ACCESS_SAVE);
	pressure -= delta;
	assert(debt_access(&a, 8, &delta) == DEBT_ACCESS_SAVE);
	pressure -= delta;
	assert(pressure == 1);

	/* Chunk c is dropped as a victim at the head. */
	assert(debt_prepare(&c, 8, &delta) == DEBT_PREPARE_MARK);
	pressure += delta;
	assert(pressure == 2);
	pressure -= debt_cleanup_delta(&c);
	assert(pressure == 0);
	tests_run++;
}

int main(void)
{
	test_activate_samples_warm_flag();
	test_prepare_marks_before_cap();
	test_prepare_caps_and_flags_low_reuse();
	test_reuse_clears_debt_and_saves();
	test_reuse_keeps_chunk_without_debt();
	test_reuse_holds_durable_low_reuse_candidate();
	test_reuse_saves_non_durable_low_reuse_candidate();
	test_cleanup_releases_remaining_debt();
	test_default_debt_max();
	test_prefetch_gate();
	test_pressure_ledger_closes();

	printf("eviction debt policy tests: %u passed\n", tests_run);
	return 0;
}

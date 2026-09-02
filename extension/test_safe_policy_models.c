/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
/* Offline unit tests for the policy state machines; no BPF/GPU interaction. */

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

#define SAFE_POLICY_INLINE static inline
#include "safe_policy_models.h"

static unsigned int tests_run;

static void test_twoq_two_hit_promotion(void)
{
	struct safe_twoq_state state = {};

	assert(safe_twoq_observe(&state, 10, 2, 1) ==
	       SAFE_TWOQ_MOVE_HEAD);
	assert(state.segment == SAFE_TWOQ_PROBATION);
	assert(state.observations == 1);
	assert(safe_twoq_observe(&state, 12, 2, 0) ==
	       SAFE_TWOQ_MOVE_TAIL);
	assert(state.segment == SAFE_TWOQ_PROTECTED);
	assert(state.observations == 2);
	assert(safe_twoq_observe(&state, 12, 2, 0) ==
	       SAFE_TWOQ_MOVE_TAIL);
	tests_run++;
}

static void test_twoq_same_generation_access_does_not_promote(void)
{
	struct safe_twoq_state state = {};

	/* activate at generation 30 admits the chunk to probation. */
	assert(safe_twoq_observe(&state, 30, 2, 1) ==
	       SAFE_TWOQ_MOVE_HEAD);
	/* The first access in that USED episode sees generation 30 too. */
	assert(safe_twoq_observe(&state, 30, 2, 0) ==
	       SAFE_TWOQ_MOVE_HEAD);
	assert(state.segment == SAFE_TWOQ_PROBATION);
	assert(state.observations == 1);
	/* A later list-state/residency episode advances generation and promotes. */
	assert(safe_twoq_observe(&state, 32, 2, 0) ==
	       SAFE_TWOQ_MOVE_TAIL);
	assert(state.segment == SAFE_TWOQ_PROTECTED);
	assert(state.observations == 2);
	tests_run++;
}

static void test_twoq_configurable_admission(void)
{
	struct safe_twoq_state state = {};

	assert(safe_twoq_observe(&state, 4, 3, 1) ==
	       SAFE_TWOQ_MOVE_HEAD);
	assert(safe_twoq_observe(&state, 6, 3, 0) ==
	       SAFE_TWOQ_MOVE_HEAD);
	assert(state.segment == SAFE_TWOQ_PROBATION);
	assert(safe_twoq_observe(&state, 6, 3, 0) ==
	       SAFE_TWOQ_MOVE_HEAD);
	assert(state.observations == 2);
	assert(safe_twoq_observe(&state, 8, 3, 0) ==
	       SAFE_TWOQ_MOVE_TAIL);
	assert(state.segment == SAFE_TWOQ_PROTECTED);
	tests_run++;
}

static void test_twoq_generation_reset(void)
{
	struct safe_twoq_state state = {};

	assert(!safe_twoq_generation_is_new(20, 22, 2));
	assert(safe_twoq_generation_is_new(20, 23, 2));
	assert(safe_twoq_generation_is_new(20, 19, 2));

	safe_twoq_observe(&state, 20, 2, 1);
	safe_twoq_observe(&state, 22, 2, 0);
	assert(state.segment == SAFE_TWOQ_PROTECTED);
	assert(safe_twoq_observe(&state, 23, 2, 1) ==
	       SAFE_TWOQ_MOVE_HEAD);
	assert(state.segment == SAFE_TWOQ_PROBATION);
	assert(state.observations == 1);
	tests_run++;
}

static void test_delta_learning_and_adaptation(void)
{
	struct safe_delta_transition transition = {};
	s32 successor = 0;

	assert(safe_delta_learn(&transition, 4, 0) == SAFE_DELTA_NEW);
	assert(transition.successor == 4 && transition.confidence == 1);
	assert(!safe_delta_predict(&transition, 2, &successor));
	assert(safe_delta_learn(&transition, 4, 1) == SAFE_DELTA_MATCH);
	assert(safe_delta_predict(&transition, 2, &successor));
	assert(successor == 4);

	assert(safe_delta_learn(&transition, -2, 1) == SAFE_DELTA_DECAY);
	assert(transition.successor == 4 && transition.confidence == 1);
	assert(safe_delta_learn(&transition, -2, 1) == SAFE_DELTA_REPLACE);
	assert(transition.successor == -2 && transition.confidence == 1);
	assert(safe_delta_learn(&transition, -2, 1) == SAFE_DELTA_MATCH);
	assert(safe_delta_predict(&transition, 2, &successor));
	assert(successor == -2);
	tests_run++;
}

static void test_delta_alternating_markov_chain(void)
{
	/* Models page deltas +2,+4,+2,+4,... with one state per predecessor. */
	struct safe_delta_transition after_two = {};
	struct safe_delta_transition after_four = {};
	s32 successor = 0;

	assert(safe_delta_learn(&after_two, 4, 0) == SAFE_DELTA_NEW);
	assert(safe_delta_learn(&after_four, 2, 0) == SAFE_DELTA_NEW);
	assert(safe_delta_learn(&after_two, 4, 1) == SAFE_DELTA_MATCH);
	assert(safe_delta_learn(&after_four, 2, 1) == SAFE_DELTA_MATCH);

	assert(safe_delta_predict(&after_two, 2, &successor));
	assert(successor == 4);
	assert(safe_delta_predict(&after_four, 2, &successor));
	assert(successor == 2);
	tests_run++;
}

static void test_delta_block_isolation(void)
{
	struct safe_delta_transition block_a = {};
	struct safe_delta_transition block_b = {};
	s32 successor_a = 0;
	s32 successor_b = 0;

	safe_delta_learn(&block_a, 8, 0);
	safe_delta_learn(&block_a, 8, 1);
	safe_delta_learn(&block_b, -3, 0);
	safe_delta_learn(&block_b, -3, 1);

	assert(safe_delta_predict(&block_a, 2, &successor_a));
	assert(safe_delta_predict(&block_b, 2, &successor_b));
	assert(successor_a == 8);
	assert(successor_b == -3);
	tests_run++;
}

static void test_delta_region_bounds(void)
{
	u32 first = 0;
	u32 outer = 0;

	assert(safe_delta_region(10, 4, 3, 0, 20, &first, &outer));
	assert(first == 14 && outer == 17);
	assert(safe_delta_region(10, -4, 3, 0, 20, &first, &outer));
	assert(first == 4 && outer == 7);
	assert(safe_delta_region(1, -1, 4, 0, 20, &first, &outer));
	assert(first == 0 && outer == 1);
	assert(!safe_delta_region(19, 2, 2, 0, 20, &first, &outer));
	assert(!safe_delta_region(10, 0, 2, 0, 20, &first, &outer));
	tests_run++;
}

int main(void)
{
	test_twoq_two_hit_promotion();
	test_twoq_same_generation_access_does_not_promote();
	test_twoq_configurable_admission();
	test_twoq_generation_reset();
	test_delta_learning_and_adaptation();
	test_delta_alternating_markov_chain();
	test_delta_block_isolation();
	test_delta_region_bounds();

	printf("safe policy model tests: %u passed\n", tests_run);
	return 0;
}

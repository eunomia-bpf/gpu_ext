/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef GPU_EXT_SAFE_POLICY_MODELS_H
#define GPU_EXT_SAFE_POLICY_MODELS_H

/*
 * Pure policy state machines shared by the BPF implementations and the
 * offline userspace tests.  The including translation unit provides the
 * fixed-width aliases (u8/u16/u32/u64/s32/s64).
 */
#ifndef SAFE_POLICY_INLINE
#define SAFE_POLICY_INLINE static __always_inline
#endif

enum safe_twoq_segment {
	SAFE_TWOQ_EMPTY = 0,
	SAFE_TWOQ_PROBATION = 1,
	SAFE_TWOQ_PROTECTED = 2,
};

enum safe_twoq_action {
	SAFE_TWOQ_KEEP = 0,
	SAFE_TWOQ_MOVE_HEAD = 1,
	SAFE_TWOQ_MOVE_TAIL = 2,
};

struct safe_twoq_state {
	u64 generation;
	u32 observations;
	u8 segment;
	u8 reserved[3];
};

SAFE_POLICY_INLINE u32 safe_twoq_promotion_threshold(u32 configured)
{
	/* A two-queue admission policy must see at least two references. */
	return configured < 2 ? 2 : configured;
}

SAFE_POLICY_INLINE int safe_twoq_generation_is_new(u64 previous,
						    u64 current,
						    u64 maximum_gap)
{
	if (current < previous)
		return 1;
	return current - previous > maximum_gap;
}

SAFE_POLICY_INLINE enum safe_twoq_action
safe_twoq_observe(struct safe_twoq_state *state,
		  u64 generation,
		  u32 promote_after,
		  int new_identity)
{
	promote_after = safe_twoq_promotion_threshold(promote_after);

	if (new_identity || state->segment == SAFE_TWOQ_EMPTY) {
		state->generation = generation;
		state->observations = 1;
		state->segment = SAFE_TWOQ_PROBATION;
		return SAFE_TWOQ_MOVE_HEAD;
	}

	/*
	 * activate is followed by access in the same USED-list residency
	 * episode. list_generation does not change for a move within the same
	 * list, so do not misclassify that callback pair as two references.
	 */
	if (state->generation == generation) {
		if (state->segment == SAFE_TWOQ_PROTECTED)
			return SAFE_TWOQ_MOVE_TAIL;
		return SAFE_TWOQ_MOVE_HEAD;
	}

	state->generation = generation;
	if (state->segment == SAFE_TWOQ_PROTECTED)
		return SAFE_TWOQ_MOVE_TAIL;

	if (state->observations != ~(u32)0)
		state->observations++;
	if (state->observations >= promote_after) {
		state->segment = SAFE_TWOQ_PROTECTED;
		return SAFE_TWOQ_MOVE_TAIL;
	}

	return SAFE_TWOQ_MOVE_HEAD;
}

enum safe_delta_update {
	SAFE_DELTA_NEW = 1,
	SAFE_DELTA_MATCH = 2,
	SAFE_DELTA_DECAY = 3,
	SAFE_DELTA_REPLACE = 4,
};

struct safe_delta_transition {
	s32 successor;
	u16 confidence;
	u16 changes;
};

SAFE_POLICY_INLINE enum safe_delta_update
safe_delta_learn(struct safe_delta_transition *transition,
		 s32 successor,
		 int initialized)
{
	if (!initialized) {
		transition->successor = successor;
		transition->confidence = 1;
		transition->changes = 0;
		return SAFE_DELTA_NEW;
	}

	if (transition->successor == successor) {
		if (transition->confidence != (u16)~0U)
			transition->confidence++;
		return SAFE_DELTA_MATCH;
	}

	if (transition->changes != (u16)~0U)
		transition->changes++;
	if (transition->confidence > 1) {
		transition->confidence--;
		return SAFE_DELTA_DECAY;
	}

	transition->successor = successor;
	transition->confidence = 1;
	return SAFE_DELTA_REPLACE;
}

SAFE_POLICY_INLINE int
safe_delta_predict(const struct safe_delta_transition *transition,
		   u32 confidence_threshold,
		   s32 *successor)
{
	if (confidence_threshold == 0)
		confidence_threshold = 1;
	if (transition->confidence < confidence_threshold)
		return 0;

	*successor = transition->successor;
	return transition->successor != 0;
}

/*
 * Anchor a contiguous prefetch window at the predicted page.  Positive
 * deltas extend forward; negative deltas extend backward.  The result is
 * always clamped to the callback's absolute [maximum_first, maximum_outer)
 * region.
 */
SAFE_POLICY_INLINE int
safe_delta_region(u32 page,
		  s32 predicted_delta,
		  u32 prefetch_pages,
		  u32 maximum_first,
		  u32 maximum_outer,
		  u32 *first,
		  u32 *outer)
{
	s64 predicted;
	s64 requested_first;
	s64 requested_outer;

	if (predicted_delta == 0 || maximum_first >= maximum_outer)
		return 0;
	if (prefetch_pages == 0)
		prefetch_pages = 1;

	predicted = (s64)page + (s64)predicted_delta;
	if (predicted_delta > 0) {
		requested_first = predicted;
		requested_outer = predicted + (s64)prefetch_pages;
	}
	else {
		requested_first = predicted - (s64)prefetch_pages + 1;
		requested_outer = predicted + 1;
	}

	if (requested_first < (s64)maximum_first)
		requested_first = (s64)maximum_first;
	if (requested_outer > (s64)maximum_outer)
		requested_outer = (s64)maximum_outer;
	if (requested_first >= requested_outer)
		return 0;

	*first = (u32)requested_first;
	*outer = (u32)requested_outer;
	return 1;
}

#endif /* GPU_EXT_SAFE_POLICY_MODELS_H */

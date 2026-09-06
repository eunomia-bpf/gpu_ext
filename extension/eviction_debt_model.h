/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Migration-debt eviction policy state machine (LMCache/gpubpf prototype).
 *
 * Pure, shared by the BPF policy (eviction_debt.bpf.c) and the offline
 * userspace tests (test_eviction_debt.c).  The including translation
 * unit provides the fixed-width aliases (u8/u32/u64).
 *
 * Semantics:
 * - debt_kv_range_*: single-KV-pool range tracking.  The uprobe/
 *   uretprobe pair on uvm_kv_malloc records the largest successful
 *   allocation as {start, end, tgid}; membership is checked per owner
 *   tgid at activation time.
 * - debt_prepare: one eviction-candidate observation for a chunk walked
 *   at the head of the USED list.  Below the debt cap it increments the
 *   chunk's at-risk debt signal; at/above the cap the chunk is a
 *   low-reuse candidate (disk-durable ones are preferred victims).
 * - debt_access: one observed reuse.  Reuse reduces/clears the debt.
 *   A cleared sub-cap chunk is saved (moved to tail); a low-reuse
 *   disk-durable chunk is held as a preferred eviction candidate.
 * - debt_suppress_prefetch: aggregate debt pressure gate.
 */
#ifndef GPU_EXT_EVICTION_DEBT_MODEL_H
#define GPU_EXT_EVICTION_DEBT_MODEL_H

#ifndef DEBT_INLINE
#define DEBT_INLINE static __always_inline
#endif

/* debt_config map keys (BPF_MAP_TYPE_ARRAY, u32 key, u64 value). */
enum debt_config_key {
	DEBT_CONFIG_DISK_DURABLE = 0, /* warm-phase disk-durable flag, 0/1 */
	DEBT_CONFIG_DEBT_MAX = 1,      /* debt cap; 0 selects DEBT_DEFAULT_MAX */
	DEBT_CONFIG_PRESSURE_THRESHOLD = 2, /* 0 disables prefetch suppression */
};

#define DEBT_DEFAULT_MAX 4

/* Per-chunk state, BPF hash map value (chunk_ptr -> state). */
struct debt_chunk_state {
	u32 owner_pid;
	u8  debt;          /* at-risk signal: candidate hits without an effective save */
	u8  accesses;      /* observed reuse count, saturated */
	u8  is_kv;         /* activated inside the recorded KV pool range */
	u8  disk_durable;  /* warm-phase flag sampled at activation (KV only) */
};

_Static_assert(sizeof(struct debt_chunk_state) == 8,
	       "debt chunk state ABI");

/*
 * Single-KV-pool range: the largest successful uvm_kv_malloc allocation,
 * recorded by the uprobe/uretprobe pair.  [start, end) with end exclusive.
 */
struct debt_kv_range {
	u64 start;
	u64 end;
	u32 tgid;
	u32 _pad;
};

/* Non-empty [start, end) containing va. */
DEBT_INLINE int debt_kv_range_contains(const struct debt_kv_range *range,
				       u64 va)
{
	return range && range->start != 0 && range->end > range->start &&
	       va >= range->start && va < range->end;
}

/* Membership scoped to the owner: va inside the range recorded for tgid. */
DEBT_INLINE int debt_kv_range_matches(const struct debt_kv_range *range,
				      u32 tgid, u64 va)
{
	return range && tgid != 0 && range->tgid == tgid &&
	       debt_kv_range_contains(range, va);
}

/*
 * Largest-wins record policy: a successful allocation replaces the
 * recorded range only when strictly larger.  Rejects empty and
 * wrapping (start + size overflow) allocations.
 */
DEBT_INLINE int debt_kv_range_replace(const struct debt_kv_range *cur,
				      u64 new_start, u64 new_size)
{
	u64 new_end;

	if (!new_start || !new_size)
		return 0;
	new_end = new_start + new_size;
	if (new_end <= new_start)
		return 0;
	if (cur && cur->end > cur->start && cur->end - cur->start >= new_size)
		return 0;
	return 1;
}

DEBT_INLINE u64 debt_effective_max(u64 configured)
{
	if (configured == 0)
		return DEBT_DEFAULT_MAX;
	if (configured > 255)
		return 255;
	return configured;
}

/*
 * Activation record.  The warm-phase disk_durable flag is sampled only
 * for KV chunks: a chunk outside the recorded KV pool range can never
 * be marked disk-durable, regardless of the flag.
 */
DEBT_INLINE void debt_activate(struct debt_chunk_state *state,
			       u32 owner_pid, int is_kv, int disk_durable)
{
	state->owner_pid = owner_pid;
	state->debt = 0;
	state->accesses = 0;
	state->is_kv = is_kv ? 1 : 0;
	state->disk_durable = (is_kv && disk_durable) ? 1 : 0;
}

enum debt_prepare_action {
	DEBT_PREPARE_MARK = 0,    /* debt incremented, chunk is now at risk */
	DEBT_PREPARE_VICTIM = 1,  /* low-reuse disk-durable: preferred eviction candidate */
	DEBT_PREPARE_PENDING = 2, /* low-reuse, not durable: next reuse saves the chunk */
};

/*
 * One eviction-candidate observation.
 *
 * - debt < debt_max: debt += 1, *pressure_delta = 1, returns MARK.
 * - debt >= debt_max: low-reuse candidate; no state change,
 *   *pressure_delta = 0, returns VICTIM for disk-durable chunks
 *   and PENDING otherwise.
 */
DEBT_INLINE enum debt_prepare_action
debt_prepare(struct debt_chunk_state *state, u64 debt_max, u64 *pressure_delta)
{
	debt_max = debt_effective_max(debt_max);

	if (state->debt < (u8)debt_max) {
		state->debt++;
		*pressure_delta = 1;
		return DEBT_PREPARE_MARK;
	}

	*pressure_delta = 0;
	if (state->disk_durable)
		return DEBT_PREPARE_VICTIM;
	return DEBT_PREPARE_PENDING;
}

enum debt_access_action {
	DEBT_ACCESS_KEEP = 0, /* no debt: keep list position */
	DEBT_ACCESS_SAVE = 1, /* reuse cleared debt: move to tail (second chance) */
	DEBT_ACCESS_HOLD = 2, /* low-reuse disk-durable: no save, preferred victim */
};

/*
 * One observed reuse.
 *
 * - debt == 0: KEEP, *pressure_delta = 0.
 * - 0 < debt < debt_max: reuse clears the debt; SAVE,
 *   *pressure_delta = cleared debt.
 * - debt >= debt_max: low-reuse candidate; the debt is cleared either
 *   way.  Disk-durable chunks return HOLD (they remain preferred
 *   eviction candidates until re-capped); the rest return SAVE.
 */
DEBT_INLINE enum debt_access_action
debt_access(struct debt_chunk_state *state, u64 debt_max, u64 *pressure_delta)
{
	u8 old_debt;

	debt_max = debt_effective_max(debt_max);
	if (state->accesses != 255)
		state->accesses++;

	old_debt = state->debt;
	state->debt = 0;
	*pressure_delta = old_debt;

	if (old_debt == 0)
		return DEBT_ACCESS_KEEP;
	if (old_debt >= (u8)debt_max && state->disk_durable)
		return DEBT_ACCESS_HOLD;
	return DEBT_ACCESS_SAVE;
}

/*
 * Debt released when a tracked chunk leaves the pool without a reuse
 * (victim dropped at gpu_evict_prepare).  The caller removes the state.
 */
DEBT_INLINE u64 debt_cleanup_delta(const struct debt_chunk_state *state)
{
	return state->debt;
}

/*
 * Speculative prefetch gate: suppress prefetch when aggregate migration
 * debt pressure reaches the configured threshold.  Threshold 0 disables
 * the gate.
 */
DEBT_INLINE int debt_suppress_prefetch(u64 pressure, u64 threshold)
{
	if (threshold == 0)
		return 0;
	return pressure >= threshold;
}

#endif /* GPU_EXT_EVICTION_DEBT_MODEL_H */

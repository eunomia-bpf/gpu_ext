/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Migration-debt eviction policy state machine (LMCache/gpubpf prototype).
 *
 * Pure, shared by the BPF policy (eviction_debt.bpf.c) and the offline
 * userspace tests (test_eviction_debt.c).  The including translation
 * unit provides the fixed-width aliases (u8/u32/u64).
 *
 * Semantics:
 * - debt_kv_entry_* and debt_kv_table_*: bounded KV pool range table.
 *   The uprobe/uretprobe pair on uvm_kv_malloc records each successful
 *   allocation as a {start, end, owner_tgid, active} entry in the
 *   DEBT_KV_RANGE_MAX-bounded table; an entry uprobe on uvm_kv_free
 *   retires the matching entry.  Activation marks a chunk is_kv only
 *   when its va_block start lies inside a live entry recorded by the
 *   same owner tgid.
 * - debt_prepare: one eviction-candidate observation for a chunk walked
 *   at the head of the USED list.  The debt signal is KV-scoped:
 *   non-KV chunks are ignored (no debt, no pressure, no reorder);
 *   disk-durable KV chunks are immediate preferred victims (no
 *   debt-cap wait); other KV chunks accumulate the at-risk debt signal
 *   below the cap and are low-reuse candidates at or above it.
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
 * Bounded KV pool range table.  The uprobe/uretprobe pair on
 * uvm_kv_malloc records each successful allocation as one entry
 * {start, end, owner_tgid, active}; an entry uprobe on uvm_kv_free
 * retires (deactivates) the matching entry.  The BPF policy keeps the
 * entries in a DEBT_KV_RANGE_MAX-slot ARRAY map; the in-memory table
 * below has the same entry layout and drives the bounded scans, so
 * every scan is verifier-bounded by DEBT_KV_RANGE_MAX.  [start, end)
 * with end exclusive.  Documented limitation: when all slots are live,
 * further successful allocations are not recorded, and without the
 * free uprobe an entry lives for the process run lifetime.
 */
#define DEBT_KV_RANGE_MAX 64

struct debt_kv_entry {
	u64 start;
	u64 end;
	u32 owner_tgid;
	u32 active;
};

_Static_assert(sizeof(struct debt_kv_entry) == 24, "kv entry ABI");

/* Bounded in-memory view of the range table (CPU tests). */
struct debt_kv_table {
	struct debt_kv_entry entries[DEBT_KV_RANGE_MAX];
};

/* Live slot: marked active with sane bounds. */
DEBT_INLINE int debt_kv_entry_valid(const struct debt_kv_entry *e)
{
	return e && e->active && e->start != 0 && e->end > e->start;
}

/*
 * Membership scoped to the owner: the entry is live, recorded for the
 * same owner tgid (nonzero), and va lies inside [start, end).
 */
DEBT_INLINE int debt_kv_entry_contains(const struct debt_kv_entry *e,
				       u32 owner_tgid, u64 va)
{
	return debt_kv_entry_valid(e) && owner_tgid != 0 &&
	       e->owner_tgid == owner_tgid &&
	       va >= e->start && va < e->end;
}

/*
 * Bounded table scan: is va inside any live entry recorded by owner
 * tgid?  The loop is bounded by DEBT_KV_RANGE_MAX regardless of the
 * entry contents, so inactive or corrupt slots can never change the
 * answer.
 */
DEBT_INLINE int debt_kv_table_contains(const struct debt_kv_table *tab,
				       u32 owner_tgid, u64 va)
{
	u32 i;

	if (!tab)
		return 0;
	for (i = 0; i < DEBT_KV_RANGE_MAX; i++)
		if (debt_kv_entry_contains(&tab->entries[i], owner_tgid, va))
			return 1;
	return 0;
}

/*
 * Slot selection for a new successful allocation (start, size):
 * - invalid input (NULL table, zero start/size/owner, or a wrapping
 *   start + size): -1.
 * - a live entry with the same [start, end) and owner_tgid: that slot
 *   (idempotent re-record).
 * - otherwise the first inactive slot, or -1 when all slots are live.
 */
DEBT_INLINE int debt_kv_table_slot_for(const struct debt_kv_table *tab,
				       u64 start, u64 size, u32 owner_tgid)
{
	u32 i;
	u64 end;
	int free_slot = -1;

	if (!tab || !start || !size || owner_tgid == 0)
		return -1;
	end = start + size;
	if (end <= start)
		return -1;
	for (i = 0; i < DEBT_KV_RANGE_MAX; i++) {
		const struct debt_kv_entry *e = &tab->entries[i];

		if (e->active && e->start == start && e->end == end &&
		    e->owner_tgid == owner_tgid)
			return (int)i;
		if (!e->active && free_slot < 0)
			free_slot = (int)i;
	}
	return free_slot;
}

/*
 * Slot retirement for a free of (ptr, size): the first live entry
 * recorded by the same owner tgid whose bounds match
 * [ptr, ptr+size), or -1 when there is no such entry.  The caller
 * deactivates the slot.
 */
DEBT_INLINE int debt_kv_table_retire_slot(const struct debt_kv_table *tab,
					  u32 owner_tgid, u64 ptr, u64 size)
{
	u32 i;
	u64 end;

	if (!tab || !ptr || !size || owner_tgid == 0)
		return -1;
	end = ptr + size;
	if (end <= ptr)
		return -1;
	for (i = 0; i < DEBT_KV_RANGE_MAX; i++) {
		const struct debt_kv_entry *e = &tab->entries[i];

		if (e->active && e->owner_tgid == owner_tgid &&
		    e->start == ptr && e->end == end)
			return (int)i;
	}
	return -1;
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
	DEBT_PREPARE_VICTIM = 1,  /* disk-durable KV: immediate preferred eviction candidate */
	DEBT_PREPARE_PENDING = 2, /* low-reuse KV, not durable: next reuse saves the chunk */
	DEBT_PREPARE_IGNORE = 3,  /* non-KV chunk: no debt, no pressure, left to native eviction */
};

/*
 * One eviction-candidate observation.  The debt signal is KV-scoped:
 * non-KV chunks never increment debt, reorder, or contribute aggregate
 * pressure.
 *
 * - non-KV: no state change, *pressure_delta = 0, returns IGNORE.
 * - disk_durable (KV): immediate victim; no state change,
 *   *pressure_delta = 0.  The chunk is left for native eviction
 *   without waiting for the debt cap.
 * - debt < debt_max: debt += 1, *pressure_delta = 1, returns MARK.
 * - debt >= debt_max: low-reuse candidate; no state change,
 *   *pressure_delta = 0, returns PENDING.
 */
DEBT_INLINE enum debt_prepare_action
debt_prepare(struct debt_chunk_state *state, u64 debt_max, u64 *pressure_delta)
{
	*pressure_delta = 0;

	if (!state->is_kv)
		return DEBT_PREPARE_IGNORE;

	if (state->disk_durable)
		return DEBT_PREPARE_VICTIM;

	debt_max = debt_effective_max(debt_max);
	if (state->debt < (u8)debt_max) {
		state->debt++;
		*pressure_delta = 1;
		return DEBT_PREPARE_MARK;
	}
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
 *   eviction candidates; retroactively marked chunks can reach this
 *   branch before their next evict_prepare observation); the rest
 *   return SAVE.
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

/*
 * Recoverability-ordering ranges (MVP).  Semantic KV-pool ranges tagged
 * with recovery metadata, so eviction candidates can be ordered by how
 * costly it is to bring the range back.  Pure and pointer-bounded like
 * the bounded KV range-table helpers above; all names are kept separate
 * from the debt_kv_* allocator-traced range API.
 *
 * Classification (first match wins):
 * - STALE:          record is NULL, retired, invalid (bounds or
 *                   out-of-range semantic fields), or its generation
 *                   differs from the table generation.
 * - PROTECT:        lifecycle pinned or active, recovery class loss, or
 *                   a nonzero deadline inside the protect window
 *                   [now, now + protect_window].  A deadline at or
 *                   before now is conservatively still protected; a
 *                   window that would wrap the u64 timeline protects
 *                   every remaining deadline.
 * - CHEAP_ELIGIBLE: inactive local-disk range whose recovery is cheap
 *                   and carries no deadline protection.
 * - DEFAULT:        everything else, including queries that no live
 *                   record covers for the given owner.
 */
enum debt_rcov_lifecycle {
	DEBT_RCOV_LIFECYCLE_INACTIVE = 0,
	DEBT_RCOV_LIFECYCLE_ACTIVE = 1,
	DEBT_RCOV_LIFECYCLE_PINNED = 2,
	DEBT_RCOV_LIFECYCLE_MAX = 3,
};

enum debt_rcov_tier {
	DEBT_RCOV_TIER_NONE = 0,
	DEBT_RCOV_TIER_LOCAL_DISK = 1,
	DEBT_RCOV_TIER_MAX = 2,
};

enum debt_rcov_recovery {
	DEBT_RCOV_RECOVERY_CHEAP = 0,
	DEBT_RCOV_RECOVERY_SLOW = 1,
	DEBT_RCOV_RECOVERY_LOSS = 2,
	DEBT_RCOV_RECOVERY_MAX = 3,
};

enum debt_rcov_class {
	DEBT_RCOV_CLASS_STALE = 0,
	DEBT_RCOV_CLASS_PROTECT = 1,
	DEBT_RCOV_CLASS_CHEAP_ELIGIBLE = 2,
	DEBT_RCOV_CLASS_DEFAULT = 3,
};

/* Bounded number of semantic ranges per table. */
#define DEBT_RCOV_MAX 16

/* Generation-tagged semantic range record.  [start, end), end exclusive. */
struct debt_rcov_record {
	u64 start;
	u64 end;
	u64 deadline;    /* recovery deadline; 0 = unknown */
	u64 last_update; /* last time the record content changed */
	u32 tgid;        /* owner tgid that recorded the range */
	u32 generation;  /* table generation the record was written in */
	u8  lifecycle;   /* enum debt_rcov_lifecycle */
	u8  tier;        /* enum debt_rcov_tier */
	u8  recovery;    /* enum debt_rcov_recovery */
	u8  retired;     /* 1: taken out of service by the table owner */
};

_Static_assert(sizeof(struct debt_rcov_record) == 48, "rcov record ABI");

/* Bounded table of semantic ranges. */
struct debt_rcov_table {
	u32 generation; /* current table generation */
	u32 count;      /* live slots; find() additionally bounds by MAX */
	struct debt_rcov_record ranges[DEBT_RCOV_MAX];
};

/*
 * Record validity: sane bounds (nonzero start, end > start) and
 * in-range semantic fields.  A record that fails is STALE material for
 * debt_rcov_classify().
 */
DEBT_INLINE int debt_rcov_record_valid(const struct debt_rcov_record *rec)
{
	return rec && rec->start != 0 && rec->end > rec->start &&
	       rec->lifecycle < DEBT_RCOV_LIFECYCLE_MAX &&
	       rec->tier < DEBT_RCOV_TIER_MAX &&
	       rec->recovery < DEBT_RCOV_RECOVERY_MAX;
}

/*
 * Bounded, owner-scoped membership: the owner recorded the range
 * (nonzero owner, tgid match) and va lies inside [start, end).
 * Invalid records contain nothing.
 */
DEBT_INLINE int debt_rcov_contains(const struct debt_rcov_record *rec,
				   u32 owner_tgid, u64 va)
{
	return rec && debt_rcov_record_valid(rec) && owner_tgid != 0 &&
	       rec->tgid == owner_tgid &&
	       va >= rec->start && va < rec->end;
}

/*
 * Deadline-window check, overflow-safe.  deadline == 0 (unknown) is
 * never deadline-protected.  A deadline at or before now is
 * conservatively still in-window.  When now + protect_window would wrap
 * the u64 timeline, the window is treated as covering the whole
 * remaining timeline (protect) instead of comparing against the wrapped
 * value.
 */
DEBT_INLINE int debt_rcov_deadline_in_window(u64 now, u64 protect_window,
					     u64 deadline)
{
	if (deadline == 0)
		return 0;
	if (deadline < now)
		return 1;
	if (now > ~(u64)0 - protect_window)
		return 1;
	return deadline - now <= protect_window;
}

/*
 * Pure classifier for one semantic range record at query time (see the
 * section comment for the STALE/PROTECT/CHEAP_ELIGIBLE/DEFAULT rules).
 * table_gen is the generation the record must match; (owner_tgid, va)
 * is the chunk being classified.
 */
DEBT_INLINE enum debt_rcov_class
debt_rcov_classify(const struct debt_rcov_record *rec, u32 table_gen,
		   u32 owner_tgid, u64 va, u64 now, u64 protect_window)
{
	if (!rec || rec->retired || rec->generation != table_gen ||
	    !debt_rcov_record_valid(rec))
		return DEBT_RCOV_CLASS_STALE;
	if (!debt_rcov_contains(rec, owner_tgid, va))
		return DEBT_RCOV_CLASS_DEFAULT;
	if (rec->lifecycle == DEBT_RCOV_LIFECYCLE_PINNED ||
	    rec->lifecycle == DEBT_RCOV_LIFECYCLE_ACTIVE ||
	    rec->recovery == DEBT_RCOV_RECOVERY_LOSS ||
	    debt_rcov_deadline_in_window(now, protect_window, rec->deadline))
		return DEBT_RCOV_CLASS_PROTECT;
	if (rec->tier == DEBT_RCOV_TIER_LOCAL_DISK &&
	    rec->recovery == DEBT_RCOV_RECOVERY_CHEAP &&
	    rec->lifecycle == DEBT_RCOV_LIFECYCLE_INACTIVE)
		return DEBT_RCOV_CLASS_CHEAP_ELIGIBLE;
	return DEBT_RCOV_CLASS_DEFAULT;
}

/*
 * Bounded spatial lookup: first record in the table containing
 * (owner, va).  The loop is bounded by DEBT_RCOV_MAX regardless of
 * tab->count, so a corrupt count cannot run past the array.  Generation
 * and retired checks are the classifier's job; returns NULL when no
 * record covers the query.
 */
DEBT_INLINE const struct debt_rcov_record *
debt_rcov_table_find(const struct debt_rcov_table *tab,
		     u32 owner_tgid, u64 va)
{
	u32 i;

	if (!tab)
		return NULL;
	for (i = 0; i < tab->count && i < DEBT_RCOV_MAX; i++)
		if (debt_rcov_contains(&tab->ranges[i], owner_tgid, va))
			return &tab->ranges[i];
	return NULL;
}

#endif /* GPU_EXT_EVICTION_DEBT_MODEL_H */

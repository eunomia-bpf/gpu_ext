#ifndef EDRVM_EVICTION_DEBT_RANGE_MODEL_H
#define EDRVM_EVICTION_DEBT_RANGE_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t edrvm_flags_t;

#define EDRVM_FLAG_STALE   ((edrvm_flags_t)1u << 0)
#define EDRVM_FLAG_PROTECT ((edrvm_flags_t)1u << 1)
#define EDRVM_FLAG_MARK    ((edrvm_flags_t)1u << 2)
#define EDRVM_FLAG_TOUCHED ((edrvm_flags_t)1u << 3)
#define EDRVM_FLAG_MASK    ((edrvm_flags_t)0xFu)

#define EDRVM_DEADLINE_UNKNOWN UINT64_MAX

struct edrvm_entry {
    uint64_t generation;
    uint64_t recovery_rank;
    uint64_t deadline;
    uint64_t last_touch;
    uint64_t debt;
    edrvm_flags_t flags;
    void *pointer;
};

struct edrvm_table {
    struct edrvm_entry *slots;
    uint32_t cap;
    uint64_t generation;
};

static inline struct edrvm_entry *edrvm_entry_at(struct edrvm_table *t, uint32_t index)
{
    if (t == NULL || t->slots == NULL || index >= t->cap)
        return NULL;
    return &t->slots[index];
}

static inline void edrvm_table_init(struct edrvm_table *t, struct edrvm_entry *slots,
                                    uint32_t cap, uint64_t generation)
{
    if (t == NULL)
        return;
    t->slots = slots;
    t->cap = cap;
    t->generation = generation;
}

static inline void edrvm_entry_reset(struct edrvm_entry *e, uint64_t generation,
                                     uint64_t recovery_rank, uint64_t deadline,
                                     void *pointer)
{
    if (e == NULL)
        return;
    e->generation = generation;
    e->recovery_rank = recovery_rank;
    e->deadline = deadline;
    e->last_touch = 0;
    e->debt = 0;
    e->flags = 0;
    e->pointer = pointer;
}

static inline int edrvm_generation_matches(const struct edrvm_table *t,
                                           const struct edrvm_entry *e)
{
    return t != NULL && e != NULL && e->generation == t->generation;
}

static inline void edrvm_advance_generation(struct edrvm_table *t)
{
    uint32_t i;
    if (t == NULL)
        return;
    t->generation += 1;
    if (t->slots == NULL)
        return;
    for (i = 0; i < t->cap; i++)
        t->slots[i].flags |= EDRVM_FLAG_STALE;
}

static inline int edrvm_has_flag(const struct edrvm_entry *e, edrvm_flags_t flag)
{
    return e != NULL && (e->flags & flag) != 0;
}

static inline void edrvm_set_flag(struct edrvm_entry *e, edrvm_flags_t flag)
{
    if (e != NULL)
        e->flags |= (edrvm_flags_t)(flag & EDRVM_FLAG_MASK);
}

static inline void edrvm_clear_flag(struct edrvm_entry *e, edrvm_flags_t flag)
{
    if (e != NULL)
        e->flags &= (edrvm_flags_t)~(flag & EDRVM_FLAG_MASK);
}

static inline int edrvm_is_stale(const struct edrvm_entry *e)
{
    return edrvm_has_flag(e, EDRVM_FLAG_STALE);
}

static inline void edrvm_set_stale(struct edrvm_entry *e)
{
    edrvm_set_flag(e, EDRVM_FLAG_STALE);
}

static inline void edrvm_clear_stale(struct edrvm_entry *e)
{
    edrvm_clear_flag(e, EDRVM_FLAG_STALE);
}

static inline int edrvm_is_protected(const struct edrvm_entry *e)
{
    return edrvm_has_flag(e, EDRVM_FLAG_PROTECT);
}

static inline void edrvm_set_protected(struct edrvm_entry *e)
{
    edrvm_set_flag(e, EDRVM_FLAG_PROTECT);
}

static inline void edrvm_clear_protected(struct edrvm_entry *e)
{
    edrvm_clear_flag(e, EDRVM_FLAG_PROTECT);
}

static inline int edrvm_is_marked(const struct edrvm_entry *e)
{
    return edrvm_has_flag(e, EDRVM_FLAG_MARK);
}

static inline void edrvm_mark(struct edrvm_entry *e)
{
    edrvm_set_flag(e, EDRVM_FLAG_MARK);
}

static inline void edrvm_unmark(struct edrvm_entry *e)
{
    edrvm_clear_flag(e, EDRVM_FLAG_MARK);
}

static inline int edrvm_has_touched(const struct edrvm_entry *e)
{
    return edrvm_has_flag(e, EDRVM_FLAG_TOUCHED);
}

static inline int edrvm_is_eligible(const struct edrvm_table *t,
                                    const struct edrvm_entry *e)
{
    return edrvm_generation_matches(t, e) && !edrvm_is_stale(e) &&
           !edrvm_is_protected(e);
}

static inline uint32_t edrvm_count_eligible(const struct edrvm_table *t)
{
    uint32_t i;
    uint32_t n = 0;
    if (t == NULL || t->slots == NULL)
        return 0;
    for (i = 0; i < t->cap; i++)
        if (edrvm_is_eligible(t, &t->slots[i]))
            n++;
    return n;
}

static inline void edrvm_debt_add(struct edrvm_entry *e, uint64_t amount)
{
    if (e == NULL || amount == 0)
        return;
    if (amount >= UINT64_MAX - e->debt)
        e->debt = UINT64_MAX;
    else
        e->debt += amount;
}

static inline void edrvm_debt_clear(struct edrvm_entry *e)
{
    if (e != NULL)
        e->debt = 0;
}

static inline void edrvm_touch(struct edrvm_entry *e, uint64_t now)
{
    if (e == NULL)
        return;
    e->last_touch = now;
    e->flags |= EDRVM_FLAG_TOUCHED;
}

static inline void edrvm_clear_access(struct edrvm_table *t)
{
    uint32_t i;
    if (t == NULL || t->slots == NULL)
        return;
    for (i = 0; i < t->cap; i++) {
        struct edrvm_entry *e = &t->slots[i];
        if (!edrvm_generation_matches(t, e))
            continue;
        edrvm_clear_flag(e, EDRVM_FLAG_TOUCHED | EDRVM_FLAG_MARK);
    }
}

static inline int edrvm_victim_before(const struct edrvm_entry *a,
                                      const struct edrvm_entry *b)
{
    if (a == NULL || b == NULL || a == b)
        return 0;
    if (a->recovery_rank != b->recovery_rank)
        return a->recovery_rank < b->recovery_rank;
    if (a->deadline != b->deadline)
        return a->deadline > b->deadline;
    if (a->last_touch != b->last_touch)
        return a->last_touch < b->last_touch;
    return (uintptr_t)a->pointer < (uintptr_t)b->pointer;
}

static inline struct edrvm_entry *edrvm_pick_victim(struct edrvm_table *t)
{
    struct edrvm_entry *best = NULL;
    uint32_t i;
    if (t == NULL || t->slots == NULL)
        return NULL;
    for (i = 0; i < t->cap; i++) {
        struct edrvm_entry *e = &t->slots[i];
        if (!edrvm_is_eligible(t, e))
            continue;
        if (best == NULL || edrvm_victim_before(e, best))
            best = e;
    }
    return best;
}

#ifdef __cplusplus
}
#endif

#endif

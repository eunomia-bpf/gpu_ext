#include <stdio.h>
#include <stdlib.h>

#include "eviction_debt_range_model.h"

static long g_checks;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%ld: %s\n", __FILE__, (long)__LINE__,   \
                    #cond);                                                  \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static void test_bounds_and_generation(void)
{
    struct edrvm_entry slots[4];
    struct edrvm_table t;

    edrvm_table_init(&t, NULL, 4, 3);
    CHECK(edrvm_entry_at(&t, 0) == NULL);
    CHECK(edrvm_pick_victim(&t) == NULL);
    CHECK(edrvm_count_eligible(&t) == 0);

    edrvm_table_init(&t, slots, 4, 3);
    CHECK(edrvm_entry_at(&t, 0) == &slots[0]);
    CHECK(edrvm_entry_at(&t, 3) == &slots[3]);
    CHECK(edrvm_entry_at(&t, 4) == NULL);
    CHECK(edrvm_entry_at(&t, UINT32_MAX) == NULL);
    CHECK(edrvm_entry_at(NULL, 0) == NULL);
    CHECK(edrvm_pick_victim(NULL) == NULL);
    CHECK(edrvm_count_eligible(NULL) == 0);

    edrvm_entry_reset(&slots[0], 3, 0, 10, (void *)(uintptr_t)0x10);
    edrvm_entry_reset(&slots[1], 3, 1, EDRVM_DEADLINE_UNKNOWN,
                      (void *)(uintptr_t)0x20);
    CHECK(edrvm_generation_matches(&t, &slots[0]));
    CHECK(edrvm_generation_matches(&t, &slots[1]));
    CHECK(edrvm_is_eligible(&t, &slots[0]));
    CHECK(edrvm_is_eligible(&t, &slots[1]));
    CHECK(edrvm_count_eligible(&t) == 2);
    CHECK(!edrvm_generation_matches(&t, NULL));
    CHECK(!edrvm_generation_matches(NULL, &slots[0]));

    edrvm_table_init(&t, slots, 4, 3);
    edrvm_entry_reset(&slots[0], 3, 0, 10, (void *)(uintptr_t)0x10);
    CHECK(edrvm_generation_matches(&t, &slots[0]));

    edrvm_advance_generation(&t);
    CHECK(t.generation == 4);
    CHECK(!edrvm_generation_matches(&t, &slots[0]));
    CHECK(edrvm_is_stale(&slots[0]));
    CHECK(edrvm_is_stale(&slots[1]));
    CHECK(!edrvm_is_eligible(&t, &slots[0]));

    edrvm_entry_reset(&slots[1], 4, 1, EDRVM_DEADLINE_UNKNOWN,
                      (void *)(uintptr_t)0x20);
    CHECK(edrvm_generation_matches(&t, &slots[1]));
    CHECK(edrvm_is_eligible(&t, &slots[1]));

    edrvm_advance_generation(NULL);
}

static void test_flags_and_eligibility(void)
{
    struct edrvm_entry slots[2];
    struct edrvm_table t;
    struct edrvm_entry *e;

    edrvm_table_init(&t, slots, 2, 5);
    edrvm_entry_reset(&slots[0], 5, 1, 100, (void *)(uintptr_t)0x20);
    edrvm_entry_reset(&slots[1], 6, 1, 100, (void *)(uintptr_t)0x40);
    e = &slots[0];

    CHECK(edrvm_is_eligible(&t, e));
    CHECK(!edrvm_is_eligible(&t, &slots[1]));

    edrvm_mark(e);
    CHECK(edrvm_is_marked(e));
    CHECK(edrvm_is_eligible(&t, e));
    edrvm_unmark(e);
    CHECK(!edrvm_is_marked(e));
    edrvm_unmark(e);

    edrvm_set_protected(e);
    CHECK(edrvm_is_protected(e));
    CHECK(!edrvm_is_eligible(&t, e));
    edrvm_clear_protected(e);
    CHECK(edrvm_is_eligible(&t, e));

    edrvm_set_stale(e);
    CHECK(edrvm_is_stale(e));
    CHECK(!edrvm_is_eligible(&t, e));
    edrvm_clear_stale(e);
    CHECK(edrvm_is_eligible(&t, e));

    edrvm_mark(e);
    edrvm_set_protected(e);
    edrvm_set_stale(e);
    e->flags |= (edrvm_flags_t)0x10u;
    edrvm_clear_flag(e, EDRVM_FLAG_MARK | EDRVM_FLAG_PROTECT |
                            EDRVM_FLAG_STALE);
    CHECK(!edrvm_is_marked(e));
    CHECK(!edrvm_is_protected(e));
    CHECK(!edrvm_is_stale(e));
    CHECK((e->flags & 0x10u) != 0);
    CHECK((e->flags & ~EDRVM_FLAG_MASK) == 0x10u);

    CHECK(!edrvm_is_marked(NULL));
    CHECK(!edrvm_is_protected(NULL));
    CHECK(!edrvm_is_stale(NULL));
    CHECK(!edrvm_has_touched(NULL));
    CHECK(!edrvm_is_eligible(NULL, e));
    CHECK(!edrvm_is_eligible(&t, NULL));
    edrvm_mark(NULL);
    edrvm_set_flag(NULL, EDRVM_FLAG_MARK);
    edrvm_clear_flag(NULL, EDRVM_FLAG_MARK);
    edrvm_entry_reset(NULL, 1, 1, 1, NULL);
}

static void test_debt_saturating(void)
{
    struct edrvm_entry e;

    edrvm_entry_reset(&e, 1, 0, 0, NULL);
    CHECK(e.debt == 0);

    edrvm_debt_add(&e, 5);
    CHECK(e.debt == 5);
    edrvm_debt_add(&e, 0);
    CHECK(e.debt == 5);
    edrvm_debt_add(&e, UINT64_MAX);
    CHECK(e.debt == UINT64_MAX);
    edrvm_debt_add(&e, 1);
    CHECK(e.debt == UINT64_MAX);

    edrvm_debt_clear(&e);
    CHECK(e.debt == 0);
    edrvm_debt_add(&e, UINT64_MAX - 2);
    CHECK(e.debt == UINT64_MAX - 2);
    edrvm_debt_add(&e, 3);
    CHECK(e.debt == UINT64_MAX);
    edrvm_debt_add(&e, UINT64_MAX);
    CHECK(e.debt == UINT64_MAX);

    edrvm_debt_clear(&e);
    edrvm_debt_clear(&e);
    CHECK(e.debt == 0);
    edrvm_debt_add(NULL, 7);
}

static void test_touch_and_clear_access(void)
{
    struct edrvm_entry slots[3];
    struct edrvm_table t;

    edrvm_table_init(&t, slots, 3, 9);
    edrvm_entry_reset(&slots[0], 9, 0, 0, NULL);
    edrvm_entry_reset(&slots[1], 9, 0, 0, NULL);
    edrvm_entry_reset(&slots[2], 8, 0, 0, NULL);

    CHECK(!edrvm_has_touched(&slots[0]));
    edrvm_touch(&slots[0], 7);
    CHECK(slots[0].last_touch == 7);
    CHECK(edrvm_has_touched(&slots[0]));
    edrvm_touch(&slots[1], 3);
    edrvm_touch(&slots[2], 9);
    edrvm_touch(NULL, 1);
    CHECK(slots[0].last_touch != slots[1].last_touch);

    edrvm_mark(&slots[0]);
    edrvm_mark(&slots[1]);
    edrvm_mark(&slots[2]);
    edrvm_set_protected(&slots[1]);
    edrvm_set_stale(&slots[0]);

    edrvm_clear_access(&t);
    CHECK(!edrvm_has_touched(&slots[0]));
    CHECK(!edrvm_is_marked(&slots[0]));
    CHECK(!edrvm_has_touched(&slots[1]));
    CHECK(!edrvm_is_marked(&slots[1]));
    CHECK(slots[0].last_touch == 7);
    CHECK(slots[1].last_touch == 3);
    CHECK(edrvm_is_protected(&slots[1]));
    CHECK(edrvm_is_stale(&slots[0]));
    CHECK(edrvm_has_touched(&slots[2]));
    CHECK(edrvm_is_marked(&slots[2]));

    edrvm_clear_access(NULL);
}

static void test_victim_order(void)
{
    struct edrvm_entry a;
    struct edrvm_entry b;

    edrvm_entry_reset(&a, 1, 3, 100, (void *)(uintptr_t)0x8);
    edrvm_entry_reset(&b, 1, 5, 900, (void *)(uintptr_t)0x4);
    CHECK(edrvm_victim_before(&a, &b));
    CHECK(!edrvm_victim_before(&b, &a));

    edrvm_entry_reset(&a, 1, 2, EDRVM_DEADLINE_UNKNOWN, (void *)(uintptr_t)0x8);
    edrvm_entry_reset(&b, 1, 2, 100, (void *)(uintptr_t)0x4);
    CHECK(edrvm_victim_before(&a, &b));
    CHECK(!edrvm_victim_before(&b, &a));

    edrvm_entry_reset(&a, 1, 2, 700, (void *)(uintptr_t)0x8);
    CHECK(edrvm_victim_before(&a, &b));
    CHECK(!edrvm_victim_before(&b, &a));

    edrvm_entry_reset(&a, 1, 2, 100, (void *)(uintptr_t)0x8);
    edrvm_entry_reset(&b, 1, 2, 100, (void *)(uintptr_t)0x8);
    CHECK(!edrvm_victim_before(&a, &b));
    CHECK(!edrvm_victim_before(&b, &a));

    a.last_touch = 40;
    b.last_touch = 90;
    CHECK(edrvm_victim_before(&a, &b));
    CHECK(!edrvm_victim_before(&b, &a));

    b.last_touch = 40;
    b.pointer = (void *)(uintptr_t)0x4;
    CHECK(!edrvm_victim_before(&a, &b));
    CHECK(edrvm_victim_before(&b, &a));

    a.pointer = b.pointer;
    b.pointer = NULL;
    CHECK(edrvm_victim_before(&b, &a));
    a.pointer = NULL;
    CHECK(!edrvm_victim_before(&a, &b));
    CHECK(!edrvm_victim_before(&b, &a));

    CHECK(!edrvm_victim_before(&a, &a));
    CHECK(!edrvm_victim_before(&a, NULL));
    CHECK(!edrvm_victim_before(NULL, &a));
}

static void test_pick_victim(void)
{
    struct edrvm_entry slots[4];
    struct edrvm_table t;

    edrvm_table_init(&t, slots, 4, 3);
    CHECK(edrvm_pick_victim(&t) == NULL);
    CHECK(edrvm_count_eligible(&t) == 0);

    edrvm_table_init(&t, slots, 4, 3);
    edrvm_entry_reset(&slots[0], 3, 1, 50, (void *)(uintptr_t)0x40);
    slots[0].last_touch = 100;
    edrvm_entry_reset(&slots[1], 3, 2, 10, (void *)(uintptr_t)0x80);
    slots[1].last_touch = 200;
    edrvm_entry_reset(&slots[2], 3, 9, 0, (void *)(uintptr_t)0x20);
    edrvm_set_protected(&slots[2]);
    slots[2].last_touch = 0;
    edrvm_entry_reset(&slots[3], 3, 9, 0, (void *)(uintptr_t)0x10);
    edrvm_set_stale(&slots[3]);
    slots[3].last_touch = 0;

    CHECK(edrvm_count_eligible(&t) == 2);
    CHECK(edrvm_pick_victim(&t) == &slots[0]);

    edrvm_set_protected(&slots[0]);
    CHECK(edrvm_pick_victim(&t) == &slots[1]);
    edrvm_clear_protected(&slots[0]);
    edrvm_set_stale(&slots[0]);
    CHECK(edrvm_pick_victim(&t) == &slots[1]);

    edrvm_table_init(&t, slots, 4, 6);
    edrvm_entry_reset(&slots[0], 6, 4, 100, (void *)(uintptr_t)0x40);
    slots[0].last_touch = 5;
    edrvm_entry_reset(&slots[1], 6, 4, EDRVM_DEADLINE_UNKNOWN,
                      (void *)(uintptr_t)0x80);
    slots[1].last_touch = 99;
    edrvm_entry_reset(&slots[2], 6, 4, 700, (void *)(uintptr_t)0x20);
    slots[2].last_touch = 1;
    edrvm_entry_reset(&slots[3], 6, 4, 700, (void *)(uintptr_t)0x10);
    slots[3].last_touch = 2;
    CHECK(edrvm_count_eligible(&t) == 4);
    CHECK(edrvm_pick_victim(&t) == &slots[1]);
    CHECK(edrvm_victim_before(&slots[1], &slots[2]));
    CHECK(edrvm_victim_before(&slots[2], &slots[0]));

    slots[3].deadline = EDRVM_DEADLINE_UNKNOWN;
    CHECK(edrvm_pick_victim(&t) == &slots[3]);

    edrvm_unmark(&slots[1]);
    edrvm_clear_access(&t);
    CHECK(!edrvm_has_touched(&slots[0]));
    CHECK(!edrvm_is_marked(&slots[3]));

    edrvm_clear_stale(&slots[3]);
    edrvm_clear_protected(&slots[3]);
    edrvm_debt_clear(&slots[3]);
    CHECK(edrvm_is_eligible(&t, &slots[3]));
}

int main(void)
{
    test_bounds_and_generation();
    test_flags_and_eligibility();
    test_debt_saturating();
    test_touch_and_clear_access();
    test_victim_order();
    test_pick_victim();
    printf("ok eviction_debt_range_model: %ld checks passed\n", g_checks);
    return 0;
}

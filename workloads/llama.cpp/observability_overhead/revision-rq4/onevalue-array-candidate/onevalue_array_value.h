/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 *
 * One-value GPU_ARRAY map (type 1503) record layout, shared byte-for-byte by
 * the BPF program and the host collector. The BPF side defines
 * ONEVALUE_ARRAY_VALUE_U64 (to u64 from vmlinux.h) before including this
 * header; the host side falls back to uint64_t.
 */
#ifndef ONEVALUE_ARRAY_VALUE_H
#define ONEVALUE_ARRAY_VALUE_H

#ifdef ONEVALUE_ARRAY_VALUE_U64
typedef ONEVALUE_ARRAY_VALUE_U64 onevalue_u64;
#else
#include <stdint.h>
typedef uint64_t onevalue_u64;
#endif

#define ONEVALUE_ARRAY_MAX_WARPS 16384ULL
#define ONEVALUE_ARRAY_MAX_EVENTS_PER_WARP 44ULL

struct onevalue_event {
	onevalue_u64 coordinate_x;
	onevalue_u64 coordinate_y;
	onevalue_u64 coordinate_z;
	onevalue_u64 timestamp;
};

struct onevalue_value {
	/* Reserved, never written by the device: total committed is derived
	 * host-side as the sum of warp_counters, so the hot path has no
	 * shared global counter. */
	onevalue_u64 total_committed;
	onevalue_u64 total_overflow;
	onevalue_u64 total_out_of_range;
	onevalue_u64 warp_counters[ONEVALUE_ARRAY_MAX_WARPS];
	struct onevalue_event
		events[ONEVALUE_ARRAY_MAX_WARPS *
			 ONEVALUE_ARRAY_MAX_EVENTS_PER_WARP];
};

_Static_assert(sizeof(struct onevalue_event) == 32,
	       "onevalue event must stay 32 bytes");
_Static_assert(sizeof(struct onevalue_value) == 23199768ULL,
	       "onevalue value layout drifted from the frozen 22.125 MiB payload");

#endif

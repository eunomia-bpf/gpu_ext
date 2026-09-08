/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 *
 * GPU-local full-record device buffer layout, shared byte-for-byte by the
 * BPF program and the host collector. The BPF side defines
 * FULL_RECORD_VALUE_U64 (to u64 from vmlinux.h) before including this
 * header; the host side falls back to uint64_t.
 *
 * The full per-thread record set (524288 slots x 256 records x 80 bytes,
 * about 10 GiB) is split into 32 equal banks of 16384 thread slots.
 * Each value stays below 512 MiB: the first 1.25-GiB bank was truncated
 * in the actual compiler BTF output despite its correct C sizeof.
 */
#ifndef FULL_RECORD_DEVICE_BUFFER_VALUE_H
#define FULL_RECORD_DEVICE_BUFFER_VALUE_H

#ifdef FULL_RECORD_VALUE_U64
typedef FULL_RECORD_VALUE_U64 frdb_u64;
#else
#include <stdint.h>
typedef uint64_t frdb_u64;
#endif

#define FRDB_NUM_BANKS 32ULL
#define FRDB_SLOTS_PER_BANK 16384ULL
#define FRDB_TOTAL_SLOTS (FRDB_NUM_BANKS * FRDB_SLOTS_PER_BANK) /* 524288 */
#define FRDB_RECORDS_PER_SLOT 256ULL

/* One original kernelretsnoop record: all ten u64 fields, 80 bytes. */
struct frdb_record {
	frdb_u64 block_x, block_y, block_z;
	frdb_u64 thread_x, thread_y, thread_z;
	frdb_u64 block_dim_x, block_dim_y, block_dim_z;
	frdb_u64 timestamp;
};

struct frdb_value {
	/* Bank-local: appends rejected because a slot's 256-record capacity
	 * was reached. */
	frdb_u64 total_overflow;
	/* Global sink: threads whose per-thread linear coordinate exceeds the
	 * 524288-slot capacity are counted in bank 0, because an out-of-range
	 * coordinate maps to no bank. */
	frdb_u64 total_out_of_range;
	/* One 64-bit append counter per thread slot in this bank. */
	frdb_u64 slot_counters[FRDB_SLOTS_PER_BANK];
	/* Record-major storage: for a fixed record index (append counter),
	 * all slots in this bank are contiguous, so simultaneous
	 * appends by adjacent threads land 80 bytes apart instead of
	 * ~20 KiB apart. A slot's record k sits at
	 * records[k * FRDB_SLOTS_PER_BANK + slot]. */
	struct frdb_record
		records[FRDB_SLOTS_PER_BANK * FRDB_RECORDS_PER_SLOT];
};

_Static_assert(sizeof(struct frdb_record) == 80,
	       "frdb record must stay 80 bytes");
_Static_assert(sizeof(struct frdb_value) == 335675408ULL,
	       "frdb bank value layout drifted from the 320 MiB payload");

#endif

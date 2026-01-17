#!/usr/bin/env python3
"""
Analyze VA block mapping from chunk_trace CSV output

This script focuses on analyzing the relationship between:
- GPU memory chunks (physical addresses)
- VA blocks (virtual address ranges)
- Chunk-to-VA mapping patterns

CSV Format:
time_ms,hook_type,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index
"""

import sys
import csv
import argparse
from collections import defaultdict, Counter
from statistics import mean, median


def parse_hex(value):
    """Parse hex value, return 0 if empty or invalid"""
    if not value or value == '':
        return 0
    try:
        return int(value, 16) if value.startswith('0x') else int(value)
    except ValueError:
        return 0


def analyze_va_blocks(filename):
    """Analyze VA block mapping patterns"""

    # Data structures
    va_blocks = {}  # va_block_addr -> {'start': ..., 'end': ..., 'chunks': set(), 'events': [...]}
    chunk_to_va = {}  # chunk_addr -> va_block_addr
    va_ranges = defaultdict(list)  # (va_start, va_end) -> list of va_block addrs

    # Read CSV
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip summary lines
            if not row.get('time_ms') or not row['time_ms'][0].isdigit():
                break

            try:
                time_ms = int(row['time_ms'])
                hook_type = row['hook_type']
                chunk_addr = parse_hex(row['chunk_addr'])
                va_block = parse_hex(row['va_block'])
                va_start = parse_hex(row['va_start'])
                va_end = parse_hex(row['va_end'])

                if hook_type == 'EVICTION_PREPARE' or va_block == 0:
                    continue

                # Track VA block info
                if va_block not in va_blocks:
                    va_blocks[va_block] = {
                        'start': va_start,
                        'end': va_end,
                        'size': va_end - va_start + 1 if va_end > va_start else 0,
                        'chunks': set(),
                        'events': []
                    }

                va_blocks[va_block]['chunks'].add(chunk_addr)
                va_blocks[va_block]['events'].append((time_ms, hook_type, chunk_addr))

                # Track chunk to VA mapping
                if chunk_addr not in chunk_to_va:
                    chunk_to_va[chunk_addr] = va_block

                # Track VA ranges
                if va_start != 0 and va_end != 0:
                    va_ranges[(va_start, va_end)].append(va_block)

            except (ValueError, KeyError) as e:
                continue

    if not va_blocks:
        print("Error: No VA block information found", file=sys.stderr)
        return

    # Analysis
    print("=" * 80)
    print("VA BLOCK MAPPING ANALYSIS")
    print("=" * 80)
    print()

    # Basic VA block statistics
    print("VA BLOCK STATISTICS")
    print("-" * 80)
    print(f"Total VA blocks:           {len(va_blocks):,}")
    print(f"Total chunks mapped:       {len(chunk_to_va):,}")
    print(f"Unique VA ranges:          {len(va_ranges):,}")
    print()

    # VA block size distribution
    va_sizes = [info['size'] for info in va_blocks.values() if info['size'] > 0]
    if va_sizes:
        print("VA block size distribution:")
        print(f"  Mean:                    {mean(va_sizes):,.0f} bytes ({mean(va_sizes)/1024/1024:.1f} MB)")
        print(f"  Median:                  {median(va_sizes):,.0f} bytes ({median(va_sizes)/1024/1024:.1f} MB)")
        print(f"  Min:                     {min(va_sizes):,.0f} bytes ({min(va_sizes)/1024/1024:.1f} MB)")
        print(f"  Max:                     {max(va_sizes):,.0f} bytes ({max(va_sizes)/1024/1024:.1f} MB)")
        print()

    # Chunks per VA block
    chunks_per_va = [len(info['chunks']) for info in va_blocks.values()]
    print("Chunks per VA block:")
    print(f"  Mean:                    {mean(chunks_per_va):.2f}")
    print(f"  Median:                  {median(chunks_per_va):.1f}")
    print(f"  Min:                     {min(chunks_per_va)}")
    print(f"  Max:                     {max(chunks_per_va)}")
    print()

    # Distribution of chunks per VA block
    chunks_per_va_dist = Counter(chunks_per_va)
    print("Distribution of chunks per VA block:")
    for count in sorted(chunks_per_va_dist.keys())[:10]:
        num_va_blocks = chunks_per_va_dist[count]
        pct = 100.0 * num_va_blocks / len(va_blocks)
        print(f"  {count:2d} chunks: {num_va_blocks:6,} VA blocks ({pct:5.1f}%)")
    if len(chunks_per_va_dist) > 10:
        print(f"  ... and {len(chunks_per_va_dist) - 10} more")
    print()

    # Multiple VA blocks per VA range (should be rare/none)
    multi_va_ranges = {k: v for k, v in va_ranges.items() if len(v) > 1}
    if multi_va_ranges:
        print(f"⚠️  WARNING: {len(multi_va_ranges)} VA ranges have multiple VA block pointers!")
        print("   This might indicate aliasing or reuse of VA block structures.")
        for (start, end), va_list in list(multi_va_ranges.items())[:5]:
            print(f"   Range 0x{start:x}-0x{end:x}: {len(va_list)} VA blocks")
        print()

    # Most active VA blocks (by event count)
    print("TOP 10 MOST ACTIVE VA BLOCKS (by event count)")
    print("-" * 80)
    va_by_events = sorted(va_blocks.items(),
                         key=lambda x: len(x[1]['events']),
                         reverse=True)[:10]

    for i, (va_addr, info) in enumerate(va_by_events, 1):
        print(f"{i:2d}. VA block 0x{va_addr:x}")
        print(f"    VA range:  0x{info['start']:x} - 0x{info['end']:x} "
              f"({info['size']/1024/1024:.1f} MB)")
        print(f"    Chunks:    {len(info['chunks'])} chunks")
        print(f"    Events:    {len(info['events'])} events")

        # Event type breakdown
        event_types = Counter(hook for _, hook, _ in info['events'])
        print(f"    Breakdown: ", end="")
        print(", ".join(f"{hook}={count}" for hook, count in event_types.most_common()))
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze VA block mapping from chunk_trace CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trace.csv
  %(prog)s /tmp/chunk_trace_core_5s.csv
        """
    )
    parser.add_argument('csv_file', help='CSV file from chunk_trace')

    args = parser.parse_args()

    try:
        analyze_va_blocks(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""Use the first chronological 6,000 successful public BurstGPT arrivals.

Only timing is reused, not LLM service times or tokens. Stable sorting preserves
ties; scaling leaves the same final 10 ms guard as periodic 100 requests/s.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import heapq
import json
from pathlib import Path

REVISION = 'd895a53bb7b8ec137d0d2fe203b335835a78c10a'
SOURCE_URL = f'https://raw.githubusercontent.com/HPMLL/BurstGPT/{REVISION}/data/BurstGPT_1.csv'
LAST_NS = 59_990_000_000


def extract(path, count=6000, last_ns=LAST_NS):
    if count < 2 or last_ns <= 0:
        raise ValueError('at least two arrivals and a positive span are required')
    total_rows = 0
    successful_rows = 0

    def successful():
        nonlocal total_rows, successful_rows
        with path.open(newline='') as source:
            reader = csv.DictReader(source)
            if not {'Timestamp', 'Response tokens'} <= set(reader.fieldnames or ()):
                raise ValueError('not the documented BurstGPT CSV schema')
            for line, row in enumerate(reader, 2):
                total_rows += 1
                try:
                    timestamp = Decimal(row['Timestamp'])
                    response_tokens = int(row['Response tokens'])
                except (ValueError, InvalidOperation, TypeError) as error:
                    raise ValueError(f'malformed source row {line}') from error
                if not timestamp.is_finite() or timestamp < 0 or response_tokens < 0:
                    raise ValueError(f'invalid source row {line}')
                if response_tokens:
                    successful_rows += 1
                    yield timestamp, line, response_tokens

    selected = heapq.nsmallest(count, successful())
    if len(selected) != count:
        raise ValueError('not enough successful source requests')
    begin, end = selected[0][0], selected[-1][0]
    if begin == end:
        raise ValueError('zero-span source segment')
    offsets = [int((t - begin) * last_ns / (end - begin)) for t, _, _ in selected]
    if offsets[0] != 0 or offsets[-1] != last_ns or offsets != sorted(offsets):
        raise ValueError('arrival scaling failed')
    return {'source_url': SOURCE_URL, 'source_revision': REVISION,
            'source_file': path.name, 'source_bytes': path.stat().st_size,
            'source_rows': total_rows, 'successful_source_rows': successful_rows,
            'selection': 'first chronological successful rows; original row number breaks ties',
            'window_ns': 60_000_000_000, 'last_arrival_ns': last_ns,
            'offsets_ns': offsets,
            'selected_rows': [{'csv_line': line, 'timestamp_seconds': str(t),
                               'response_tokens': tokens} for t, line, tokens in selected]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.source)
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2)
        destination.write('\n')
    print(f"{len(result['offsets_ns'])} arrivals; {result['successful_source_rows']} successful source rows; {args.output}")

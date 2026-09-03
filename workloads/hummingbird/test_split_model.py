#!/usr/bin/env python3
"""Synthetic CPU syntax/coverage tests, not GPU numerical or performance evidence."""
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

import split_model as splitter


SOURCE = '''extern "C" __global__ void __launch_bounds__(32) f(float* __restrict__ out);
extern "C" __global__ void __launch_bounds__(32) f(float* __restrict__ out) {
  // A comment with { blockIdx.y } must stay a comment.
  unsigned int i = blockIdx.x + 5 * blockIdx.y + 20 * blockIdx.z;
  __shared__ float temp[32];
  temp[threadIdx.x] = i;
  __syncthreads();
  out[i * 32 + threadIdx.x] = temp[threadIdx.x];
}
'''


def host():
    return {'funcs': [{'name': 'test', 'kernels': [
        {'name': 'f', 'launch_params': [5, 4, 3, 32, 1, 1], 'args': [0]}]}]}


class SplitModelTests(unittest.TestCase):
    def test_offsets_in_definitions_and_prototypes_keep_block_barriers(self):
        changed, arguments = splitter.transform(SOURCE)
        self.assertEqual(arguments, {'f': 1})
        self.assertEqual(changed.count(splitter.OFFSET_PARAMETERS), 2)
        for axis in 'xyz':
            self.assertIn(f'(blockIdx.{axis} + hb_offset_{axis})', changed)
        self.assertIn('__syncthreads();', changed)
        self.assertIn('// A comment with { blockIdx.y }', changed)

    def test_rejects_unsupported_or_ambiguous_cuda(self):
        for expression in ('gridDim.x', 'atomicAdd(out, 1)', 'cooperative_groups::this_grid()',
                           'asm("nop;")', '__threadfence()', 'blockIdx.w', 'blockIdx'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                splitter.transform(SOURCE.replace('unsigned int i =', f'{expression}; unsigned int i ='))
        for source in (SOURCE.replace('float* __restrict__ out', 'int n'),
                       SOURCE + SOURCE, SOURCE.replace('__launch_bounds__(32)', ''),
                       SOURCE[:-3], SOURCE.replace('unsigned int i', 'unsigned int hb_offset_x')):
            with self.assertRaises(ValueError):
                splitter.transform(source)

    def test_three_dimensional_partition_exact_once_and_bounded(self):
        rng = random.Random(20260903)
        cases = [([5, 4, 3], 7), ([1, 112, 1], 13), ([2, 14, 16], 41), ([1, 1, 1], 1)]
        cases += [([rng.randrange(1, 10) for _ in range(3)], rng.randrange(1, 70)) for _ in range(80)]
        for grid, cap in cases:
            with self.subTest(grid=grid, cap=cap):
                seen = []
                for offset, tile in splitter.split_grid(grid, cap):
                    self.assertLessEqual(tile[0] * tile[1] * tile[2], cap)
                    for axis in range(3):
                        self.assertGreaterEqual(offset[axis], 0)
                        self.assertLessEqual(offset[axis] + tile[axis], grid[axis])
                    seen += [tuple(offset[i] + point[i] for i in range(3))
                             for point in itertools.product(*(range(n) for n in tile))]
                expected = list(itertools.product(*(range(n) for n in grid)))
                self.assertEqual(sorted(seen), expected)

    def test_consolidation_full_grid_and_invalid_dimensions(self):
        self.assertEqual(list(splitter.split_grid([2, 3, 4], 24)), [((0, 0, 0), (2, 3, 4))])
        for grid, cap in [([0, 1, 1], 1), ([1, 2], 1), ([1, True, 1], 1),
                          ([1, 1, 1], 0), ([1, 1, 1], 2**31), ([1, 1, 1], 1.5)]:
            with self.assertRaises(ValueError):
                list(splitter.split_grid(grid, cap))

    def test_host_metadata_matches_source_arguments_and_all_dimensions(self):
        inventory = splitter.launch_inventory(host(), {'f': 1})
        self.assertEqual(inventory[0]['grid'], [5, 4, 3])
        self.assertEqual(inventory[0]['block'], [32, 1, 1])
        temporary = host()
        temporary['temp_args'] = [128]
        temporary['funcs'][0]['kernels'][0]['args'] = [-1]
        self.assertEqual(len(splitter.launch_inventory(temporary, {'f': 1})), 1)
        for change in ({'name': 'missing'}, {'args': []}, {'launch_params': [1] * 5},
                       {'args': [True]}, {'args': [-1]}, {'launch_params': [1, 1, 1, 1024, 2, 1]}):
            bad = host()
            bad['funcs'][0]['kernels'][0].update(change)
            with self.assertRaises(ValueError):
                splitter.launch_inventory(bad, {'f': 1})

    def test_file_preparation_is_cpu_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata = root / 'source.cu', root / 'host.json'
            source.write_text(SOURCE)
            metadata.write_text(json.dumps(host()))
            report = splitter.prepare(source, metadata, None)
            self.assertFalse(report['gpu_correctness_validated'])
            self.assertFalse(report['profiled'])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['host.json', 'source.cu'])
            output = root / 'new'
            splitter.prepare(source, metadata, output)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ['launches.json', 'mod.cu'])
            with self.assertRaises(FileExistsError):
                splitter.prepare(source, metadata, output)
            self.assertEqual(source.read_text(), SOURCE)


if __name__ == '__main__':
    unittest.main()

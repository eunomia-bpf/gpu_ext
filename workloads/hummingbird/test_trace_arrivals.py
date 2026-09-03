import tempfile
from pathlib import Path
import unittest

from trace_arrivals import extract


class TraceTests(unittest.TestCase):
    def source(self, text):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'trace.csv'
        path.write_text(text)
        return path

    def test_chronological_success_filter_and_ties(self):
        source = self.source('Timestamp,Response tokens\n30,1\n10,1\n5,0\n10,2\n20,1\n')
        result = extract(source, count=4, last_ns=100)
        self.assertEqual(result['offsets_ns'], [0,0,50,100])
        self.assertEqual([r['csv_line'] for r in result['selected_rows']], [3,5,6,2])
        self.assertEqual(result['successful_source_rows'], 4)
        self.assertEqual(result['source_rows'], 5)

    def test_decimal_timing_and_first_subset(self):
        source = self.source('Timestamp,Response tokens\n0.1,1\n0.3,1\n0.2,1\n9,1\n')
        result = extract(source, count=3, last_ns=59_990_000_000)
        self.assertEqual(result['offsets_ns'], [0,29_995_000_000,59_990_000_000])
        self.assertLess(result['offsets_ns'][-1], result['window_ns'])

    def test_empty_short_and_zero_span_rejected(self):
        for text in ('Timestamp,Response tokens\n',
                     'Timestamp,Response tokens\n1,1\n',
                     'Timestamp,Response tokens\n1,1\n1,1\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract(self.source(text), count=2)

    def test_malformed_nonfinite_and_negative_rejected(self):
        for row in ('NaN,1','-1,1','1,-1','1,broken'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                extract(self.source('Timestamp,Response tokens\n'+row+'\n2,1\n'), count=2)
        with self.assertRaises(ValueError):
            extract(self.source('unrelated\n1\n'))


if __name__ == '__main__':
    unittest.main()

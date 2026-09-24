import unittest
from pathlib import Path
from record_stride import apply
from bandwidth_scopes import instrument


class StrideTest(unittest.TestCase):
    def test_all_variants(self):
        source = (Path(__file__).resolve().parents[1]/'python/test_npu_batch_get_perf.py').read_text()
        for stride in (656,704,768,1024):
            compile(instrument(apply(source,stride)), 'adapted', 'exec')
            physical = ((stride*128+655)//656+3)//4*4
            self.assertLessEqual(127*stride+656, physical*656)
            self.assertEqual((physical*656)%64, 0)
            self.assertEqual(stride%64, 0 if stride != 656 else 16)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            apply('', 512)
        with self.assertRaises(RuntimeError):
            apply('', 704)


if __name__ == '__main__':
    unittest.main()

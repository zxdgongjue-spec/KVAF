import unittest
from pathlib import Path
from bandwidth_scopes import instrument, summarize_sample


class ScopesTest(unittest.TestCase):
    def test_upstream_instrumentation(self):
        source = Path(__file__).resolve().parents[1]/'python/test_npu_batch_get_perf.py'
        compile(instrument(source.read_text()), str(source), 'exec')

    def test_fail_closed(self):
        with self.assertRaises(RuntimeError):
            instrument('')

    def test_paired_scopes(self):
        p = dict(kernel_start_cycle=100, kernel_finish_cycle=61100,
                 first_doorbell_cycle=4100, completion_done_cycle=60100)
        result = summarize_sample([p], 1000, 1343488, 212, 141, 200)
        self.assertEqual(result['kernel_wall_us'], 61)
        self.assertEqual(result['transfer_window_us'], 56)
        self.assertAlmostEqual(result['GBps']['host'], 6.337207547, places=8)
        self.assertEqual(result['host_outside_kernel_us'], 151)


if __name__ == '__main__':
    unittest.main()

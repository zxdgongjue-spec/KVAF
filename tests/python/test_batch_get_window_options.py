"""Large Direct Get workloads remain distinct from bounded Gather/Scatter/Put."""
import argparse
import unittest
from types import SimpleNamespace

from utils.get_options import add_backend_arguments, add_workload_arguments, resolve_backend, validate_workload
from utils.perf_execution import BenchmarkExecution


def workload(backend, count):
    parser = argparse.ArgumentParser()
    add_backend_arguments(parser)
    add_workload_arguments(parser)
    args = parser.parse_args(["--backend", backend, "--request-length", str(count),
                              "--token-bytes", "1", "--tokens-per-block", "1"])
    resolve_backend(args, parser)
    return args


class WindowOptions(unittest.TestCase):
    def test_large_direct_get(self):
        for backend in ("aiv", "ubmem"):
            for count in (65537, 130815, 1000000, (1 << 31) - 1):
                self.assertIsNone(validate_workload(workload(backend, count)))
            self.assertIsNotNone(validate_workload(workload(backend, 1 << 31)))

    def test_aicpu_capacity_stays_bounded(self):
        for backend in ("gather_read", "cpu_push_scatter"):
            self.assertIsNone(validate_workload(workload(backend, 65536)))
            self.assertIsNotNone(validate_workload(workload(backend, 65537)))

    def test_benchmark_resources_and_put_limit(self):
        kvaf = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **kw: kw)
        for transport in ("urma", "ubmem"):
            run = BenchmarkExecution(kvaf, transport=transport, get_capacity=1000000, max_inflight_total=0)
            self.assertEqual(run.get_capacity, 1000000)
            with self.assertRaisesRegex(ValueError, "Put request"):
                BenchmarkExecution(kvaf, transport=transport, put_capacity=65537)
        for algorithm in ("gather", "scatter"):
            with self.assertRaisesRegex(ValueError, "Get request"):
                BenchmarkExecution(kvaf, executor="aicpu", algorithm=algorithm, get_capacity=65537)


if __name__ == "__main__":
    unittest.main()

"""All Get paths accept large batches; workspace windows and Put remain bounded."""
import argparse
import unittest
from types import SimpleNamespace

from utils.get_options import add_backend_arguments, add_workload_arguments, resolve_backend, validate_workload, workload_command
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

    def test_large_aicpu_get(self):
        for backend in ("gather_read", "cpu_push_scatter"):
            for count in (65536, 65537, 1000000):
                self.assertIsNone(validate_workload(workload(backend, count)))
            self.assertIsNotNone(validate_workload(workload(backend, 1 << 31)))

    def test_benchmark_resources_and_put_limit(self):
        kvaf = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **kw: kw)
        for transport in ("urma", "ubmem"):
            run = BenchmarkExecution(kvaf, transport=transport, get_capacity=1000000, max_inflight_total=0)
            self.assertEqual(run.get_capacity, 1000000)
            with self.assertRaisesRegex(ValueError, "Put request"):
                BenchmarkExecution(kvaf, transport=transport, put_capacity=65537)
        for algorithm in ("gather", "scatter"):
            run = BenchmarkExecution(kvaf, executor="aicpu", algorithm=algorithm, get_capacity=65537)
            self.assertEqual(run.get_capacity, 65537)

    def test_workspace_budgets_and_child_command(self):
        kvaf = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **kw: kw)
        for backend, algorithm, name in (("gather_read", "gather", "gather_ddr_bytes"),
                                         ("cpu_push_scatter", "scatter", "scatter_hbm_bytes")):
            args = workload(backend, 1000000)
            self.assertEqual(getattr(args, name), 0)
            setattr(args, name, 64 << 20)
            self.assertIsNone(validate_workload(args))
            command = workload_command(args)
            self.assertIn("--" + name.replace("_", "-"), command)
            run = BenchmarkExecution(kvaf, executor="aicpu", algorithm=algorithm,
                                     get_cores=5, put_cores=1, **{name: 64 << 20})
            self.assertEqual(run.runtime_kwargs["get_resources"][0][1][name], 64 << 20)
            self.assertEqual(run.runtime_kwargs["get_resources"][0][1]["cores"], 5)
            self.assertEqual(run.runtime_kwargs["put_resources"][0][1]["cores"], 1)
            setattr(args, name, -1)
            self.assertIsNotNone(validate_workload(args))
            with self.assertRaises(ValueError):
                BenchmarkExecution(kvaf, **{name: 1})
            with self.assertRaises(ValueError):
                BenchmarkExecution(kvaf, executor="aicpu", algorithm=algorithm, **{name: 1 << 64})


if __name__ == "__main__":
    unittest.main()

"""All Get paths accept large batches; workspace windows and Put remain bounded."""
import argparse
import csv
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.get_options import (
    add_backend_arguments, add_workload_arguments, gather_read_wave_settings,
    resolve_backend, validate_workload, workload_command,
)
from utils.perf_execution import BenchmarkExecution
from utils.batch_get_csv import BatchGetCsv


def workload(backend, count):
    parser = argparse.ArgumentParser()
    add_backend_arguments(parser)
    add_workload_arguments(parser)
    args = parser.parse_args(["--backend", backend, "--request-length", str(count),
                              "--token-bytes", "1", "--tokens-per-block", "1"])
    resolve_backend(args, parser)
    return args


class WindowOptions(unittest.TestCase):
    def test_gather_read_wave_limits_forwarding_and_csv(self):
        kvaf = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **kw: kw)
        self.assertIsNone(workload("gather_read", 1000000).gather_read_wave_wrs)
        for limit in (1, 31, 32, 64, 96, 128, 256, 512, 4096):
            args = workload("gather_read", 1000000)
            args.gather_read_wave_wrs = limit
            self.assertIsNone(validate_workload(args))
            parser = argparse.ArgumentParser()
            add_backend_arguments(parser)
            add_workload_arguments(parser)
            child = parser.parse_args(workload_command(args))
            resolve_backend(child, parser)
            self.assertEqual(child.gather_read_wave_wrs, limit)
            run = BenchmarkExecution(kvaf, executor="aicpu", algorithm="gather", gather_read_wave_wrs=limit)
            self.assertEqual(run.runtime_kwargs["get_resources"][0][1]["gather_read_wave_wrs"], limit)
            self.assertNotIn("gather_read_wave_wrs", run.runtime_kwargs["put_resources"][0][1])
        for limit in (-1, 0, 4097, 1 << 32):
            args.gather_read_wave_wrs = limit
            self.assertIsNotNone(validate_workload(args))
            with self.assertRaisesRegex(ValueError, "gather_read_wave_wrs"):
                BenchmarkExecution(kvaf, executor="aicpu", algorithm="gather", gather_read_wave_wrs=limit)
        for backend in ("aiv", "ubmem", "cpu_push_scatter"):
            args = workload(backend, 8192)
            self.assertNotIn("--gather-read-wave-wrs", workload_command(args))
            args.gather_read_wave_wrs = 64
            self.assertIsNotNone(validate_workload(args))
        with self.assertRaisesRegex(ValueError, "requires AICPU Gather"):
            BenchmarkExecution(kvaf, gather_read_wave_wrs=64)
        with patch.dict(os.environ, {"KVAF_BATCH_GET_GATHER_READ_WAVE_WRS": "96"}):
            self.assertEqual(workload("gather_read", 1000000).gather_read_wave_wrs, 96)
        with tempfile.TemporaryDirectory() as directory:
            args = workload("gather_read", 1000000)
            args.output_dir = Path(directory)
            args.kernel_profile = False
            args.gather_read_wave_wrs = 64
            report = BatchGetCsv(args)
            report.add_summary("graph", {})
            report.write()
            with report.summary_path.open(newline="") as output:
                row = next(csv.DictReader(output))
                self.assertEqual(row["gather_read_wave_wrs"], "64")
                self.assertEqual(row["gather_read_wave_wrs_mode"], "fixed")

    def test_gather_read_auto_default_and_child_command(self):
        kvaf = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **kw: kw)
        args = workload("gather_read", 1000000)
        for count in (0, 8191, 8192, 8193, 1000000):
            args.miss_count = count
            self.assertIsNone(validate_workload(args))
            expected = 512 if count <= 8192 else 64
            self.assertEqual(gather_read_wave_settings(args), dict(
                gather_read_wave_wrs=expected, gather_read_wave_wrs_mode="auto"))
            self.assertIsNone(args.gather_read_wave_wrs)
            command = workload_command(args)
            self.assertNotIn("--gather-read-wave-wrs", command)
            parser = argparse.ArgumentParser()
            add_backend_arguments(parser)
            add_workload_arguments(parser)
            child = parser.parse_args(command)
            resolve_backend(child, parser)
            self.assertIsNone(child.gather_read_wave_wrs)
            self.assertEqual(child.miss_count, count)
        run = BenchmarkExecution(kvaf, executor="aicpu", algorithm="gather")
        self.assertNotIn("gather_read_wave_wrs", run.runtime_kwargs["get_resources"][0][1])
        for count, limit in ((1000000, 512), (8192, 64)):
            args.miss_count, args.gather_read_wave_wrs = count, limit
            self.assertEqual(gather_read_wave_settings(args), dict(
                gather_read_wave_wrs=limit, gather_read_wave_wrs_mode="fixed"))

    def test_gather_read_auto_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            args = workload("gather_read", 1000000)
            args.output_dir = Path(directory)
            args.kernel_profile = False
            for count, expected in ((8192, "512"), (8193, "64")):
                args.miss_count = count
                report = BatchGetCsv(args)
                report.add_summary("graph", {})
                report.write()
                with report.summary_path.open(newline="") as output:
                    row = next(csv.DictReader(output))
                self.assertEqual(row["gather_read_wave_wrs"], expected)
                self.assertEqual(row["gather_read_wave_wrs_mode"], "auto")

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

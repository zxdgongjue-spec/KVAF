"""SGE benchmark configuration reaches resource budgets and child commands."""
import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from test_npu_batch_get_perf import parse_args, validate_args
from utils.batch_get_csv import BatchGetCsv
from utils.get_options import workload_command
from utils.perf_execution import BenchmarkExecution


class CpuSglOptionsTest(unittest.TestCase):
    def test_sge_values_and_worker_forwarding(self):
        for count in (0, 4, 8):
            with self.subTest(count=count):
                args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", str(count),
                                   "--aicpu-workers", "4"])
                self.assertIsNone(validate_args(args))
                child = parse_args(workload_command(args))
                self.assertEqual(child.sge_per_wr, count)
                self.assertEqual(child.aicpu_workers, 4)

    def test_reject_invalid_limits_and_backends(self):
        for value in (-1, 17):
            args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", str(value)])
            self.assertIn("sge-per-wr", validate_args(args))
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            args = parse_args(["--backend", backend, "--sge-per-wr", "4"])
            self.assertIn("cpu_sgl_write", validate_args(args))

    def test_only_get_budget_receives_sge(self):
        kvaf = SimpleNamespace(ExecutionResources=lambda **kw: kw,
                               PutExecutionConfig=lambda *args: args,
                               GetExecutionConfig=lambda *args: args)
        for count in (4, 8):
            execution = BenchmarkExecution(kvaf, executor="aicpu", algorithm="cpu_sgl_write",
                                           sge_per_wr=count)
            self.assertEqual(execution.runtime_kwargs["get_resources"][0][1]["sge_per_wr"], count)
            self.assertNotIn("sge_per_wr", execution.runtime_kwargs["put_resources"][0][1])

    def test_csv_records_selected_sge(self):
        args = parse_args(["--backend", "cpu_sgl_write", "--sge-per-wr", "8"])
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            report = BatchGetCsv(args)
            report.add_summary("eager", {})
            report.write()
            with report.summary_path.open() as source:
                self.assertEqual(next(csv.DictReader(source))["sge_per_wr"], "8")


if __name__ == "__main__":
    unittest.main()

"""Gather checkpoints must fail closed and never turn a prefix into a perf PASS."""
import contextlib
import io
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import aiv_batch_get_test_common as common
import test_aiv_gather_batch_get_perf as perf
import test_aiv_gather_pipeline_debug as debug


def failing(message):
    def call():
        raise RuntimeError(message)
    return call


class GatherProbeResultTests(unittest.TestCase):
    def test_deployment_report_identifies_old_file_and_loaded_python(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gather.o"
            path.write_bytes(b"KvafGatherBatchGetAivKernelV2\0")
            module = SimpleNamespace(__file__="/old/kvaf/__init__.py",
                                     _native=SimpleNamespace(__file__="/old/kvaf/_kvaf.so"))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                common.report_gather_kernel(str(path), module)
            self.assertIn(f"path={path}", output.getvalue())
            self.assertIn("name_markers=V2 expected_symbol=KvafGatherBatchGetAivKernelV3", output.getvalue())
            self.assertIn("native=/old/kvaf/_kvaf.so", output.getvalue())

    def test_name_scan_does_not_pretend_to_validate_acl_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.o"
            path.write_bytes(b"unknown binary format")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                common.report_gather_kernel(str(path), SimpleNamespace())
            self.assertIn("name_markers=unknown", output.getvalue())

    def test_matching_prefix_and_counters_only(self):
        for stage in ("ack", "wqe", "sq", "first_read"):
            wrs = 128 if stage == "first_read" else 0
            message = (f"[KVAF_AIV_GATHER_PROBE_COMPLETE] stage={stage} sequence=1 "
                       f"cores=1 requests=8193 data_wrs={wrs}")
            with contextlib.redirect_stdout(io.StringIO()):
                common.expect_gather_probe(failing(message), stage, requests=8193, data_wrs=wrs)

    def test_device_errors_wrong_stage_and_old_kernels_are_failures(self):
        for message in ("acl=507015", "[KVAF_AIV_GATHER_PROBE_COMPLETE] stage=ack requests=8193 data_wrs=0"):
            with self.assertRaises(RuntimeError):
                common.expect_gather_probe(failing(message), "wqe", requests=8193, data_wrs=0)
        with self.assertRaises(AssertionError):
            common.expect_gather_probe(lambda: None, "wqe", requests=8193, data_wrs=0)
        for fields in ("requests=0 data_wrs=0", "requests=8193 data_wrs=1", "requests=8193"):
            with self.assertRaises(AssertionError):
                common.expect_gather_probe(failing("[KVAF_AIV_GATHER_PROBE_COMPLETE] stage=wqe " + fields),
                                           "wqe", requests=8193, data_wrs=0)

    def test_perf_rejects_probe_before_importing_hardware(self):
        with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_AIV_GATHER_PROBE": "sq"}), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(perf.main(), 2)

    def test_debug_stops_on_first_error_and_uses_benchmark_shape(self):
        with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_NPU_CORE_COUNT": "1"}), \
             patch.object(debug.subprocess, "run", side_effect=[
                 subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 1)]) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(debug.main(), 1)
            self.assertEqual(run.call_count, 2)
            environment = run.call_args.kwargs["env"]
            self.assertEqual(environment["KVAF_AIV_GATHER_PROBE"], "wqe")
            self.assertEqual(environment["KVAF_NPU_REQUESTS"], str(perf.REQUEST_COUNT))
            self.assertEqual(environment["KVAF_NPU_TOKEN_BYTES"], str(perf.TOKEN_BYTES))

    def test_full_regression_has_probe_disabled(self):
        stages = []
        def run(command, *, env, check):
            stages.append(env["KVAF_AIV_GATHER_PROBE"])
            return subprocess.CompletedProcess(command, 0)
        with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1"}), \
             patch.object(debug.subprocess, "run", side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(debug.main(), 0)
        self.assertEqual(stages, ["ack", "wqe", "sq", "first_read", ""])


if __name__ == "__main__":
    unittest.main()

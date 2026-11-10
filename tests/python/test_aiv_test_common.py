"""Shared AIV test helpers: cleanup, deployment reporting and probe results."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import aiv_batch_get_test_common as common
import test_aiv_gather_batch_get_perf as perf


def failing(message):
    def call():
        raise RuntimeError(message)
    return call


class GatherProbeResultTests(unittest.TestCase):
    def test_sparse_functional_rows_reject_coalescing_before_loading_hardware(self):
        with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_AIV_GATHER_COALESCE_SGES": "1"}), \
             patch.object(common, "kernel_paths") as paths, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(common.run("gather_read"), 2)
            paths.assert_not_called()

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
            self.assertIn("name_markers=V2 expected_symbol=KvafGatherBatchGetAivKernelV9", output.getvalue())
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


class ScatterProbeResultTest(unittest.TestCase):
    def test_checkpoint_marker(self):
        for stage in ("done", "flush", "setup", "dma", "load", "copy"):
            def call():
                verification = {"setup": " configured_bytes=656", "dma": " dma_bytes=656",
                                "load": " verification=gm_ub_byte_compare verified_bytes=656"}.get(stage, "")
                raise RuntimeError(f"[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage={stage} sequence=1 cores=1{verification}")
            common.expect_scatter_probe(call, stage, expected_bytes=656)

    def test_setup_and_dma_require_matching_byte_count(self):
        for stage, field in (("setup", "configured_bytes"), ("dma", "dma_bytes")):
            for count in ("", f" {field}=0", f" {field}=655", " verified_bytes=656"):
                def call():
                    raise RuntimeError(f"[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage={stage} "
                                       f"sequence=1 cores=1{count}")
                with self.subTest(stage=stage, count=count), self.assertRaisesRegex(AssertionError, "byte count"):
                    common.expect_scatter_probe(call, stage, expected_bytes=656)

    def test_dma_completion_is_not_load_verification(self):
        def call():
            raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=dma sequence=1 cores=1 dma_bytes=656")
        with self.assertRaises(RuntimeError):
            common.expect_scatter_probe(call, "load", expected_bytes=656)

    def test_legacy_load_checkpoint_is_rejected(self):
        def call():
            raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=load sequence=1 cores=1")
        with self.assertRaisesRegex(AssertionError, "no byte verification"):
            common.expect_scatter_probe(call, "load")

    def test_incomplete_load_verification_is_rejected(self):
        for count in ("", " verified_bytes=0", " verified_bytes=655"):
            def call():
                raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=load sequence=1 cores=1 "
                                   f"verification=gm_ub_byte_compare{count}")
            with self.assertRaisesRegex(AssertionError, "verified byte count"):
                common.expect_scatter_probe(call, "load", expected_bytes=656)

    def test_faults_are_preserved(self):
        for stage in ("done", "setup", "dma"):
            for message in ("acl=507015", "CQ timeout", "expected_result=6",
                            "[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=flush sequence=1 cores=1"):
                error = RuntimeError(message)
                def call():
                    raise error
                with self.assertRaises(RuntimeError) as caught:
                    common.expect_scatter_probe(call, stage)
                self.assertIs(caught.exception, error)

    def test_old_host_cannot_silently_pass(self):
        for stage in ("done", "setup", "dma"):
            with self.assertRaisesRegex(AssertionError, "rebuild Host and kernel"):
                common.expect_scatter_probe(lambda: None, stage)


class Runtime:
    initialized = True

    def shutdown(self):
        raise RuntimeError("cleanup failed")


class CleanupTests(unittest.TestCase):
    def test_original_exception_survives_cleanup_failure(self):
        original = RuntimeError("execution failed")
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            with self.assertRaises(RuntimeError) as raised:
                try:
                    raise original
                finally:
                    common.shutdown_runtime(Runtime())
        self.assertIs(raised.exception, original)
        self.assertIn("shutdown also failed", log.getvalue())
        self.assertIn("cleanup failed", log.getvalue())

    def test_cleanup_failure_without_original_still_raises(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            common.shutdown_runtime(Runtime())

    def test_uninitialized_runtime_does_not_shutdown(self):
        runtime = Runtime()
        runtime.initialized = False
        common.shutdown_runtime(runtime)


if __name__ == "__main__":
    unittest.main()

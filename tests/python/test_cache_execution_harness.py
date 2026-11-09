"""Portable checks for the hardware runner's oracle, reporting and cleanup."""
from array import array
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call

import test_cache_execution_e2e as e2e


class HarnessTests(unittest.TestCase):
    def test_matrix_plan_excludes_reserved_providers(self):
        args = e2e.parser().parse_args(["--list"])
        e2e.validate(args)
        plan = e2e.plan(args)
        self.assertEqual([plan[m]["matrix_pairs"] for m in e2e.MODES], [6, 1, 12])
        for mode in plan.values():
            self.assertNotIn("urma/aiv/scatter", mode["gets"])
            self.assertNotIn("urma/aiv/gather", mode["gets"])
        self.assertEqual(e2e.plan(e2e.parser().parse_args(
            ["--modes", "ubmem", "--cases", "graph"]))["ubmem"]["cases"], ["graph"])

    def test_oracle_uses_offsets_reorders_and_preserves_sentinel(self):
        source = bytes(range(24))  # Three blocks, two 4-byte tokens each.
        request = ([2, 0, 999], [4, 0, -1], [2], [3, 1, -1])
        expected = bytearray([0xA5]) * 24
        expected[12:16] = bytes([20, 21, 22, 23])
        expected[4:8] = bytes([0, 1, 2, 3])
        actual = e2e.oracle(source, request, 4, 8, 6)
        self.assertEqual(actual, expected)
        corrupted = bytearray(actual)
        corrupted[0] = 0
        with self.assertRaisesRegex(AssertionError, "byte 0"):
            e2e.compare_bytes(corrupted, actual)
        self.assertEqual(e2e.oracle(source, ([999], [-1], [0], [-1]), 4, 8, 6),
                         bytes([0xA5]) * 24)

    def test_generated_requests_cover_nonzero_offsets_and_invalid_suffix(self):
        indices, offsets, miss, output = e2e.layout(5, 2, 16, 3, epoch=1)
        self.assertEqual(miss, [3])
        self.assertTrue(all(i < 5 for i in indices[:3]))
        self.assertIn(16, offsets[:3])
        self.assertEqual(indices[3:], [7, 7])
        self.assertEqual(offsets[3:], [-1, -1])
        self.assertEqual(output[3:], [-1, -1])
        self.assertEqual(len(set(output[:3])), 3)
        self.assertNotEqual(e2e.payload(2, 8, 1), e2e.payload(2, 8, 2))

    def test_borrowed_snapshot_does_not_keep_native_memoryview(self):
        addresses = array("Q", [4096, 8192])
        view = dict(addresses=memoryview(addresses), lengths=[8, 8], valid=[1, 1], size=2)
        copied = e2e.snapshot(view)
        addresses[0] = 0
        self.assertEqual(copied["addresses"], [4096, 8192])

    def test_negative_tests_require_the_expected_failure(self):
        def failure():
            raise ValueError("capacity exceeded")
        e2e.expect_error(failure, ValueError, "capacity")
        with self.assertRaises(AssertionError):
            e2e.expect_error(failure, ValueError, "graph")
        with self.assertRaises(AssertionError):
            e2e.expect_error(lambda: None, ValueError, "capacity")

    def test_rejected_put_checks_every_key_for_not_found(self):
        runtime = Mock()
        runtime.prepare_access_table.side_effect = KeyError("LockKeys key was not found")
        e2e.check_missing_keys(runtime, [64, 65, 66])
        self.assertEqual(runtime.prepare_access_table.call_args_list,
                         [call([64]), call([65]), call([66])])
        runtime.release_access_table.assert_not_called()

    def test_rejected_put_detects_partial_commit_and_releases_probe_table(self):
        runtime = Mock()
        runtime.prepare_access_table.side_effect = [
            KeyError("LockKeys key was not found"), "committed-key-table"]
        with self.assertRaisesRegex(AssertionError, "rejected Put committed key 65"):
            e2e.check_missing_keys(runtime, [64, 65, 66])
        runtime.release_access_table.assert_called_once_with("committed-key-table")

    def test_missing_key_check_does_not_accept_unrelated_failures(self):
        for error, expected in (
            (KeyError("AccessTable was not found"), AssertionError),
            (RuntimeError("Runtime has not been initialized"), RuntimeError),
        ):
            with self.subTest(error=error):
                runtime = Mock()
                runtime.prepare_access_table.side_effect = error
                with self.assertRaises(expected):
                    e2e.check_missing_keys(runtime, [64])
                runtime.release_access_table.assert_not_called()

    def test_report_records_failures_without_swallowing_them(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            path = Path(temporary) / "report.json"
            report = e2e.Report(path, {})
            with report.case("ok"):
                pass
            with self.assertRaisesRegex(ValueError, "replay failed"):
                with report.case("bad"):
                    raise ValueError("replay failed")
            saved = json.loads(path.read_text())
            self.assertEqual([r["status"] for r in saved["results"]], ["PASS", "FAIL"])
            self.assertFalse(saved["hardware_overlap_measured"])

    def test_unresolved_get_does_not_release_table_or_keys(self):
        runtime = Mock()
        runtime.prepare_access_table.return_value = "table"
        runtime.access_table_view.side_effect = [
            dict(addresses=[4096], lengths=[8], valid=[1], size=1), "device-table"]
        runtime.drain_get.side_effect = RuntimeError("DMA unresolved")
        suite = object.__new__(e2e.Suite)
        suite.runtime, suite.block, suite.slot = runtime, 8, 4096
        suite.region = {"base": 4096}
        with self.assertRaisesRegex(RuntimeError, "DMA unresolved"):
            with suite.existing_data([0], b"12345678", object()):
                pass
        runtime.release_access_table.assert_not_called()
        runtime.batch_free.assert_not_called()

    def test_missing_environment_exits_nonzero_and_writes_fail_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            result = subprocess.run(
                [sys.executable, str(Path(e2e.__file__)), "--build-dir", temporary,
                 "--peer", "127.0.0.1:1", "--report", str(path)],
                text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn("KVAF_EXECUTION_SUMMARY FAIL", result.stdout)
            self.assertEqual(json.loads(path.read_text())["status"], "FAIL")
            self.assertNotIn("KVAF_EXECUTION_RESULT {\"case\": \"environment\", \"status\": \"PASS\"",
                             result.stdout)

    def test_wrapper_forwards_arguments_with_spaces_without_hardware(self):
        wrapper = e2e.ROOT / "tests/run_cache_execution_e2e.sh"
        result = subprocess.run(
            ["bash", str(wrapper), "/tmp/build with spaces", "--list", "--modes", "urma+ubmem",
             "--cases", "matrix"], text=True, capture_output=True, timeout=10,
            env={**os.environ, "PYTHON": sys.executable})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["urma+ubmem"]["matrix_pairs"], 12)


if __name__ == "__main__":
    unittest.main()

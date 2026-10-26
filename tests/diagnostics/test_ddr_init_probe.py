"""Exercise the probe CLI and lifecycle with an explicit fake NPU backend."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import ddr_init_probe as probe


class DdrInitProbeTests(unittest.TestCase):
    def invoke(self, extra, *, initialize_error=None, shutdown_error=None,
               input_error=None):
        calls = []

        class Runtime:
            def __init__(self, kernel, **options):
                calls.append(("construct", kernel, options))
                self.remote_region = {}

            def link_to(self, address):
                calls.append(("link_to", address))

            def initialize(self, size, slot):
                calls.append(("initialize", size, slot))
                if initialize_error:
                    raise initialize_error
                self.remote_region = {"length": size, "slot_size": slot}

            def shutdown(self):
                calls.append(("shutdown",))
                if shutdown_error:
                    raise shutdown_error

        fake_modules = {
            "torch": types.SimpleNamespace(npu=types.SimpleNamespace(
                set_device=lambda device: calls.append(("set_device", device)))),
            "torch_npu": types.ModuleType("torch_npu"),
            "kvaf": types.SimpleNamespace(NpuRuntime=Runtime),
        }
        stdout, stderr = io.StringIO(), io.StringIO()
        result, error = None, None
        with tempfile.TemporaryDirectory(prefix="kvaf-probe-test-") as temp:
            build = Path(temp)
            (build / "python/kvaf").mkdir(parents=True)
            (build / "bin").mkdir()
            for name in ("get", "put"):
                (build / f"bin/kvaf_execute_batch_{name}_kernels.o").touch()
            argv = ["ddr_init_probe.py", "--build", str(build), "--device", "7", *extra]
            with patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)), \
                    patch.dict(sys.modules, fake_modules), patch.object(probe.signal, "signal"), \
                    patch("builtins.input", side_effect=input_error or [""]), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    result = probe.main()
                except BaseException as exc:
                    error = exc
        records = [json.loads(line.split(" ", 1)[1])
                   for line in stdout.getvalue().splitlines() if line.startswith("DDR_INIT_PROBE ")]
        return result, error, calls, records

    def test_all_requested_capacities_round_to_slots_and_release(self):
        for gib in (1, 8, 16, 32, 64):
            with self.subTest(gib=gib):
                result, error, calls, records = self.invoke(["--ddr-gib", str(gib)])
                self.assertEqual(result, 0)
                self.assertIsNone(error)
                self.assertEqual(calls[0], ("set_device", 7))
                allocation = next(c for c in calls if c[0] == "initialize")
                size, slot = allocation[1:]
                self.assertEqual(size % slot, 0)
                self.assertGreaterEqual(size, gib * 1024**3)
                self.assertLess(size - gib * 1024**3, slot)
                self.assertEqual(calls[-1], ("shutdown",))
                self.assertTrue(any(r["event"] == "ready" for r in records))

    def test_profile_opt_in_passes_native_flag_only_when_requested(self):
        for profile in (False, True):
            extra = ["--ddr-gib", "1"] + (["--profile-initialize"] if profile else [])
            result, error, calls, records = self.invoke(extra)
            self.assertEqual(result, 0)
            self.assertIsNone(error)
            options = next(c for c in calls if c[0] == "construct")[2]
            self.assertEqual(options.get("profile_initialize", False), profile)
            self.assertEqual("profile_initialize" in options, profile)
            self.assertEqual(records[0]["profile_initialize"], profile)

    def test_failed_initialize_is_recorded_and_still_releases(self):
        original = RuntimeError("injected registration failure")
        _, error, calls, records = self.invoke(["--ddr-gib", "1"], initialize_error=original)
        self.assertIs(error, original)
        self.assertEqual(calls[-1], ("shutdown",))
        failure = next(r for r in records if r.get("operation") == "initialize" and r["event"] == "failed")
        self.assertEqual(failure["error"], str(original))
        self.assertFalse(any(r["event"] == "ready" for r in records))

    def test_cleanup_failure_does_not_hide_original_initialize_failure(self):
        original = RuntimeError("injected initialize failure")
        _, error, _, records = self.invoke(["--ddr-gib", "1"], initialize_error=original,
                                            shutdown_error=ValueError("injected cleanup failure"))
        self.assertIs(error, original)
        failed = [r["operation"] for r in records if r["event"] == "failed"]
        self.assertEqual(failed, ["initialize", "shutdown"])

    def test_cleanup_failure_after_success_is_not_swallowed(self):
        cleanup = RuntimeError("injected cleanup failure")
        _, error, _, _ = self.invoke(["--ddr-gib", "1"], shutdown_error=cleanup)
        self.assertIs(error, cleanup)

    def test_hold_interrupt_releases_successful_region(self):
        result, error, calls, records = self.invoke(["--ddr-gib", "1", "--hold"],
                                                    input_error=KeyboardInterrupt())
        self.assertEqual(result, 130)
        self.assertIsNone(error)
        self.assertEqual(calls[-1], ("shutdown",))
        self.assertTrue(any(r["event"] == "ready" for r in records))
        self.assertTrue(any(r["event"] == "interrupted" for r in records))

    def test_dry_run_avoids_device_and_runtime(self):
        result, error, calls, records = self.invoke(["--ddr-gib", "64", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertIsNone(error)
        self.assertEqual(calls, [])
        self.assertEqual(len(records), 1)

    def test_invalid_slot_is_rejected_before_device_use(self):
        _, error, calls, _ = self.invoke(["--ddr-gib", "1", "--slot-size", "83968"])
        self.assertIsInstance(error, SystemExit)
        self.assertEqual(error.code, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

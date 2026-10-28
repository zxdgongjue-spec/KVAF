"""Startup regressions: no hardware or network is needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import run_probe


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.get = self.root / "get.o"
        self.put = self.root / "put.o"
        self.get.touch()
        self.put.touch()
        self.env = mock.patch.dict(os.environ, {"KVAF_BATCH_GET_KERNEL": "", "KVAF_BATCH_PUT_KERNEL": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.argv = ["decode", "--control", "127.0.0.1:29991", "--cpu", "127.0.0.1:29990",
                     "--device", "7", "--eid", "00000000003f030000100000df00eb00"]

    def assert_local_error(self, argv, text):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["run_probe.py"] + argv), \
                mock.patch.object(run_probe, "rpc") as rpc, \
                contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as exc:
            run_probe.main()
        self.assertEqual(exc.exception.code, 2)
        rpc.assert_not_called()
        self.assertIn(text, output.getvalue())

    def test_missing_put_kernel_fails_before_any_rpc(self):
        self.assert_local_error(self.argv + ["--get-kernel", str(self.get)], "--put-kernel")

    def test_missing_kernel_file_fails_before_any_rpc(self):
        self.assert_local_error(self.argv + ["--get-kernel", str(self.get), "--put-kernel",
                                            str(self.root / "absent.o")], "file does not exist")

    def test_same_service_port_fails_before_any_rpc(self):
        argv = list(self.argv)
        argv[argv.index("--cpu") + 1] = "127.0.0.1:29991"
        self.assert_local_error(argv, "normally --cpu K_IP:29990")

    def test_loopback_alias_same_endpoint_rejected(self):
        argv = list(self.argv)
        argv[argv.index("--cpu") + 1] = "localhost:29991"
        self.assert_local_error(argv, "different service endpoints")

    def test_environment_paths_reach_runtime_constructor(self):
        with mock.patch.dict(os.environ, {"KVAF_BATCH_GET_KERNEL": str(self.get),
                                          "KVAF_BATCH_PUT_KERNEL": str(self.put)}):
            args = run_probe.parse_args(self.argv)
        runtime = mock.Mock()
        # Stop before initialize allocates any remote memory.
        runtime.link_to.side_effect = RuntimeError("constructor observed")
        constructor = mock.Mock(return_value=runtime)
        with mock.patch.dict(sys.modules, {"kvaf": types.SimpleNamespace(NpuRuntime=constructor)}), \
                self.assertRaisesRegex(RuntimeError, "constructor observed"):
            run_probe.decode(args, None, "npu:7")
        self.assertEqual(constructor.call_args.args, (str(self.get), 656, 1))
        self.assertEqual(constructor.call_args.kwargs["batch_put_kernel_object_path"], str(self.put))
        runtime.link_to.assert_called_once_with("127.0.0.1:29990")

    def test_json_kernel_paths_supported_without_environment(self):
        config = self.root / "options.json"
        config.write_text(json.dumps({"batch_get_kernel_object_path": str(self.get),
                                      "batch_put_kernel_object_path": str(self.put)}))
        args = run_probe.parse_args(self.argv + ["--runtime-options", str(config)])
        self.assertEqual(args.get_kernel, str(self.get))
        self.assertEqual(args.runtime_kwargs["batch_put_kernel_object_path"], str(self.put))

    def test_cli_kernel_paths_override_json_and_environment(self):
        config = self.root / "options.json"
        config.write_text(json.dumps({"batch_get_kernel_object_path": "/missing/json-get.o",
                                      "batch_put_kernel_object_path": "/missing/json-put.o"}))
        with mock.patch.dict(os.environ, {"KVAF_BATCH_GET_KERNEL": "/missing/env-get.o",
                                          "KVAF_BATCH_PUT_KERNEL": "/missing/env-put.o"}):
            args = run_probe.parse_args(self.argv + ["--runtime-options", str(config),
                "--get-kernel", str(self.get), "--put-kernel", str(self.put)])
        self.assertEqual(args.runtime_kwargs["batch_put_kernel_object_path"], str(self.put))
        self.assertEqual(args.get_kernel, str(self.get))

    def test_prefill_does_not_require_batch_kernels(self):
        exporter = self.root / "exporter.so"
        exporter.touch()
        args = run_probe.parse_args(["prefill", "--control", "127.0.0.1:29991", "--device", "6",
            "--eid", "00000000003f030000100000df00cb00", "--exporter", str(exporter)])
        self.assertEqual(args.exporter, str(exporter))


if __name__ == "__main__":
    unittest.main()

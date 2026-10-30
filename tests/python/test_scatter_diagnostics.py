"""Failure isolation and artifact evidence without torch, CANN, or a device."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnose_aiv_scatter as diagnostic


class ScatterDiagnosticsTest(unittest.TestCase):
    def test_worker_uses_explicit_stage_even_if_shell_has_done(self):
        for stage in ("gm", "body", "dma", "ub", "load"):
            def run(mode):
                self.assertEqual(mode, "cpu_push_scatter")
                self.assertEqual(os.environ["KVAF_AIV_SCATTER_PROBE"], stage)
                self.assertEqual(os.environ["KVAF_AIV_SCATTER_DIAGNOSTICS"], "1")
                return 0
            with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_AIV_SCATTER_PROBE": "done"}), \
                 patch.object(diagnostic, "run", side_effect=run):
                self.assertEqual(diagnostic.main(["--worker-stage", stage]), 0)

    def test_first_failure_stops_and_saves_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel, put = root / "scatter.o", root / "put.o"
            kernel.write_bytes(b"scatter-binary")
            put.write_bytes(b"put-binary")
            with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_NPU_TOKEN_BYTES": "656"}), \
                 patch.object(diagnostic, "kernel_paths", return_value=(str(kernel), str(put))), \
                 patch.object(diagnostic.subprocess, "check_output", return_value="commit\n"), \
                 patch.object(diagnostic, "run_stage", side_effect=[0, 23]) as stage:
                self.assertEqual(diagnostic.main(["--stages", "setup", "dma", "load",
                                                 "--log-dir", str(root / "logs")]), 23)
                self.assertEqual([c.args[0] for c in stage.call_args_list], ["setup", "dma"])
            manifest = (root / "logs/manifest.txt").read_text(encoding="utf-8")
            self.assertIn(hashlib.sha256(b"scatter-binary").hexdigest(), manifest)
            self.assertIn("checkout_commit=commit", manifest)

    def test_short_token_has_no_false_body_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "binary.o"
            binary.write_bytes(b"binary")
            with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_NPU_TOKEN_BYTES": "13",
                                         "KVAF_AIV_SCATTER_PROBE": "done"}), \
                 patch.object(diagnostic, "kernel_paths", return_value=(str(binary), str(binary))), \
                 patch.object(diagnostic.subprocess, "check_output", return_value="commit\n"), \
                 patch.object(diagnostic, "run_stage", return_value=0) as stage:
                self.assertEqual(diagnostic.main(["--log-dir", str(Path(temporary) / "logs")]), 0)
                self.assertEqual([c.args[0] for c in stage.call_args_list], ["load"])
                self.assertEqual(diagnostic.main(["--stages", "body"]), 2)

    def test_child_has_own_stage_and_merged_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(diagnostic.subprocess, "Popen") as popen:
                child = popen.return_value.__enter__.return_value
                child.pid = 123
                child.stdout = ["probe evidence\n"]
                child.wait.return_value = 1
                self.assertEqual(diagnostic.run_stage("ub", Path(temporary)), 1)
                command = popen.call_args.args[0]
                self.assertEqual(command[-2:], ["--worker-stage", "ub"])
                self.assertEqual(popen.call_args.kwargs["env"]["KVAF_AIV_SCATTER_PROBE"], "ub")
            log = (Path(temporary) / "scatter-ub.log").read_text(encoding="utf-8")
            self.assertIn("pid=123", log)
            self.assertIn("probe evidence", log)
            self.assertIn("exit_code=1", log)


if __name__ == "__main__":
    unittest.main()

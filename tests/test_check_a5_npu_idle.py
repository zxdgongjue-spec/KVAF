#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "check_a5_npu_idle.py"
FIXTURE = ROOT / "tests" / "fixtures" / "npu_smi_info_ascend950dt.txt"

SPEC = importlib.util.spec_from_file_location("check_a5_npu_idle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CheckA5NpuIdleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = FIXTURE.read_text(encoding="utf-8")

    def test_ascend950dt_explicit_idle_record_is_proof(self) -> None:
        result = MODULE.inspect_npu_smi_info(self.output, 2)
        self.assertEqual(result.status, "idle")
        self.assertEqual(result.process_ids, ())

    def test_ascend950dt_named_process_is_busy(self) -> None:
        result = MODULE.inspect_npu_smi_info(self.output, 0)
        self.assertEqual(result.status, "busy")
        self.assertEqual(result.process_ids, (3277159,))

    def test_blank_process_name_is_still_busy(self) -> None:
        result = MODULE.inspect_npu_smi_info(self.output, 1)
        self.assertEqual(result.status, "busy")
        self.assertEqual(result.process_ids, (326837,))

    def test_missing_target_is_unproven(self) -> None:
        result = MODULE.inspect_npu_smi_info(self.output, 99)
        self.assertEqual(result.status, "unproven")

    def test_missing_process_table_is_unproven(self) -> None:
        truncated = self.output.split(
            "| NPU ID                    | Process id", maxsplit=1
        )[0]
        result = MODULE.inspect_npu_smi_info(truncated, 2)
        self.assertEqual(result.status, "unproven")
        self.assertIn("process table header", result.reason)

    def test_absent_target_process_record_is_not_treated_as_idle(self) -> None:
        missing = self.output.replace(
            "| No running processes found in NPU 2                                                                              |\n",
            "",
        )
        result = MODULE.inspect_npu_smi_info(missing, 2)
        self.assertEqual(result.status, "unproven")

    def test_conflicting_idle_and_busy_records_are_unproven(self) -> None:
        conflict = self.output.replace(
            "| No running processes found in NPU 2                                                                              |",
            "| 2                         | 42             | python3            | 1                     | NA                      |\n"
            "| No running processes found in NPU 2                                                                              |",
        )
        result = MODULE.inspect_npu_smi_info(conflict, 2)
        self.assertEqual(result.status, "unproven")
        self.assertEqual(result.process_ids, (42,))

    def test_malformed_target_process_row_is_unproven(self) -> None:
        malformed = self.output.replace(
            "| No running processes found in NPU 2                                                                              |",
            "| 2                         | unknown        | python3            | 1                     | NA                      |",
        )
        result = MODULE.inspect_npu_smi_info(malformed, 2)
        self.assertEqual(result.status, "unproven")
        self.assertIn("malformed", result.reason)

    def test_cli_exit_codes_distinguish_idle_busy_and_unproven(self) -> None:
        for device, expected_rc, expected_status in (
            (2, 0, "idle"),
            (1, 3, "busy"),
            (99, 2, "unproven"),
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--physical-device-id",
                    str(device),
                ],
                input=self.output,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, expected_rc, completed.stderr)
            self.assertIn(
                f"KVAF_NPU_IDLE_STATUS={expected_status}", completed.stdout
            )


if __name__ == "__main__":
    unittest.main()

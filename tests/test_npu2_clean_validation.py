#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "scripts" / "run_a5_direct_urma_npu2_clean_validation.sh"
COORDINATOR = (
    ROOT / "scripts" / "prepare_and_run_a5_direct_urma_npu2_clean.sh"
)
TRACE = ROOT / "scripts" / "trace_a5_direct_urma_npu2_jfr_failure.sh"


class Npu2CleanValidationTests(unittest.TestCase):
    def test_script_is_valid_and_pins_clean_npu2_contract(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        for script in (WRAPPER, COORDINATOR, TRACE):
            result = subprocess.run(
                [bash, "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        source = WRAPPER.read_text(encoding="utf-8")
        for required in (
            "kvaf_pin_value KVAF_LOGICAL_DEVICE_ID 2",
            "kvaf_pin_value KVAF_PHYSICAL_DEVICE_ID 2",
            "kvaf_pin_value KVAF_HOST_EID 00000000007f040000100000df0a0b00",
            "kvaf_pin_value KVAF_DEVICE_EID 000000000000020000100000df004100",
            "kvaf_pin_value KVAF_DIRECT_MANAGEMENT_MODE ub",
            "kvaf_pin_value KVAF_DIRECT_MANAGEMENT_JETTY_ID 502",
            "kvaf_pin_value KVAF_DIRECT_ALLOW_BUSY_NPU 0",
            "KVAF_CROSS_NPU[A-Za-z0-9_]*",
            "kvaf_pin_value KVAF_DIRECT_PROVIDER_BUILD_DIR",
            "kvaf_pin_value KVAF_A5_DIRECT_BUILD_DIR",
            "kvaf_pin_value KVAF_DIRECT_PROVIDER_SONAME",
            "kvaf_pin_value KVAF_DIRECT_AICPU_ARCHIVE",
            "unset KVAF_HOST_URMA_DEVICE",
            "unset KVAF_DEVICE_URMA_DEVICE",
            "KVAF_NPU2_TRANSFER_COUNT:-4096",
            "KVAF_NPU2_BYTES_PER_TRANSFER:-1024",
            "KVAF_NPU2_VALIDATION_LIMIT:-full",
            "PASS_LIMITED_ZERO_WQE",
            "kvaf_run_stage zero-wqe 0 1024 1 1",
            "kvaf_run_stage one-kib 1 1024 1 1",
            "run_a5_direct_urma_epoch_owner_validation.sh",
        ):
            self.assertIn(required, source)
        self.assertNotIn("risk-accepted-cross-npu2-v1", source)
        self.assertNotIn("KVAF_DIRECT_ALLOW_BUSY_NPU=1", source)
        self.assertNotIn("rm -f /data/.kvaf-state", source)
        self.assertNotIn("kill ", source)

        coordinator = COORDINATOR.read_text(encoding="utf-8")
        provider = coordinator.index("build_a5_direct_urma_provider.sh")
        package = coordinator.index("package_a5_direct_urma_aicpu.sh")
        direct = coordinator.index('build_a5_direct_urma.sh"')
        owner_probe = coordinator.index("run_a5_aicpu_owner_probe.sh")
        validation = coordinator.index(
            "run_a5_direct_urma_npu2_clean_validation.sh"
        )
        self.assertLess(provider, package)
        self.assertLess(package, direct)
        self.assertLess(direct, owner_probe)
        self.assertLess(owner_probe, validation)
        for required in (
            "/usr/local/Ascend/cann-9.1.0",
            "kvaf_cann_alias_real",
            "kvaf_cann_entry_real",
            'source "${kvaf_cann_entry_real}"',
            "ASCEND_HOME_PATH",
            "kvaf_bisheng_real",
        ):
            self.assertIn(required, coordinator)
        self.assertNotIn(
            '[[ -L "${KVAF_ASCEND_SET_ENV}"', coordinator
        )
        self.assertNotIn("risk-accepted-cross-npu2-v1", coordinator)
        self.assertNotIn("rm -f /data/.kvaf-state", coordinator)
        self.assertNotIn("kill ", coordinator)

        trace = TRACE.read_text(encoding="utf-8")
        for required in (
            "EXECUTION_ENV=SERVER_B_HOST",
            "/proc/sys/kernel/random/boot_id",
            "trace identity would be ambiguous",
            "journalctl -k -n 0 --show-cursor",
            'journalctl -k --after-cursor "${kvaf_journal_cursor}"',
            "dmesg --time-format=iso",
            "dmesg-ring-unscoped-fallback",
            "KVAF_NPU2_H2D_TIMEOUT_SECONDS=20",
            "KVAF_NPU2_RUN_TIMEOUT_SECONDS=90",
            "KVAF_NPU2_VALIDATION_LIMIT=zero-wqe",
            "prepare_and_run_a5_direct_urma_npu2_clean.sh",
            "this boot-id/commit trace was already selected",
            "kvaf_tee_rc=${kvaf_pipeline_status[1]}",
            "console evidence write failed",
            "KVAF_JFR_KERNEL_TAIL_BEGIN",
            "/sys/kernel/tracing",
            "/sys/kernel/debug/tracing",
            "available_filter_functions",
            "uburma_cmd_create_jfr",
            "ubcore_create_jfr",
            "udma_create_jfr",
            "function_graph",
            "set_graph_function",
            "KVAF_JFR_FUNCTION_GRAPH_ROOT=",
            "KVAF_JFR_FUNCTION_GRAPH_TAIL_BEGIN",
            "No hardware operation was attempted and no one-shot trace directory was consumed.",
        ):
            self.assertIn(required, trace)
        for forbidden in ("npu-smi set", "docker rm", "rm -f", "kill "):
            self.assertNotIn(forbidden, trace)

    def run_preflight(self, **updates: str) -> subprocess.CompletedProcess[str]:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        env = os.environ.copy()
        for variable in (
            "KVAF_CROSS_NPU_RISK_ACCEPTANCE",
            "KVAF_CROSS_NPU_SOURCE_RUN_ID",
            "KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256",
            "KVAF_CROSS_NPU_ARTIFACT_ID",
            "KVAF_CROSS_NPU_ZERO_WQE_RESULT_DIR",
            "KVAF_LOGICAL_DEVICE_ID",
            "KVAF_PHYSICAL_DEVICE_ID",
            "KVAF_NPU2_VALIDATION_LIMIT",
        ):
            env.pop(variable, None)
        env.update(
            {
                "ASCEND_HOME_PATH": "/nonexistent/test-cann",
                "KVAF_NPU2_TRANSFER_COUNT": "4096",
            }
        )
        env.update(updates)
        return subprocess.run(
            [bash, str(WRAPPER)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_rejects_target_and_risk_overrides_before_hardware(self) -> None:
        result = self.run_preflight(KVAF_LOGICAL_DEVICE_ID="7")
        self.assertEqual(result.returncode, 2)
        self.assertIn("KVAF_LOGICAL_DEVICE_ID is source-owned", result.stderr)
        self.assertNotIn("RESULT_DIR=", result.stdout)

        result = self.run_preflight(
            KVAF_CROSS_NPU_RISK_ACCEPTANCE=(
                "I_ACCEPT_UNKNOWN_CROSS_NPU_DMA_RISK"
            )
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("is forbidden", result.stderr)
        self.assertNotIn("RESULT_DIR=", result.stdout)

        result = self.run_preflight(
            KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN="0000000000000002"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN is forbidden",
            result.stderr,
        )
        self.assertNotIn("RESULT_DIR=", result.stdout)

    def test_rejects_invalid_batch_shape_before_hardware(self) -> None:
        result = self.run_preflight(KVAF_NPU2_TRANSFER_COUNT="0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("KVAF_NPU2_TRANSFER_COUNT", result.stderr)
        self.assertNotIn("RESULT_DIR=", result.stdout)

        result = self.run_preflight(
            KVAF_NPU2_TRANSFER_COUNT="999999999999999999999999999999"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("KVAF_NPU2_TRANSFER_COUNT", result.stderr)

        result = self.run_preflight(
            KVAF_NPU2_TRANSFER_COUNT="4096",
            KVAF_NPU2_BYTES_PER_TRANSFER="65536",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("64 MiB registration limit", result.stderr)

        result = self.run_preflight(KVAF_NPU2_VALIDATION_LIMIT="one-kib")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be full or zero-wqe", result.stderr)
        self.assertNotIn("RESULT_DIR=", result.stdout)


if __name__ == "__main__":
    unittest.main()

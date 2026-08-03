#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
WRAPPER = (
    ROOT / "scripts" / "run_a5_direct_urma_cross_npu2_risk_accepted.sh"
)
COORDINATOR = (
    ROOT / "scripts" / "run_a5_cross_npu2_recovery_experiment.sh"
)
BUILD = ROOT / "scripts" / "build_a5_direct_urma.sh"
RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"
GATE_SCRIPT = ROOT / "scripts" / "validate_a5_cross_npu_zero_wqe_pass.py"
SPEC = importlib.util.spec_from_file_location("cross_npu2_gate", GATE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

SOURCE_RUN_ID = "5" * 32
TARGET_RUN_ID = "2" * 32
ARTIFACT_ID = "a" * 32
SOURCE_COMMIT = "b" * 40


class CrossNpu2RiskPathTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "KVAF"
        self.results = (
            self.state
            / "experimental"
            / "npu-2"
            / SOURCE_RUN_ID
            / "results"
        )
        self.result = self.results / "zero-wqe-pass"
        self.exchange = self.result / "exchange"
        self.exchange.mkdir(parents=True)
        self.source_quarantine = self.state / "direct-urma.quarantine"
        self.source_quarantine.write_text(
            "schema=kvaf-a5-direct-quarantine-v1\n",
            encoding="utf-8",
        )
        self.source_sha = hashlib.sha256(
            self.source_quarantine.read_bytes()
        ).hexdigest()
        self.write_manifest()
        self.write_final("host")
        self.write_final("device")
        shutil.copy2(
            ROOT / "scripts" / "classify_a5_direct_urma_outcome.py",
            self.result / "classify_a5_direct_urma_outcome.py",
        )
        (self.result / "host.log").write_text(
            "KVAF_A5_DIRECT_TEARDOWN_OK role=host\n",
            encoding="utf-8",
        )
        (self.result / "device.log").write_text(
            "KVAF_A5_DIRECT_ZERO_WQE_OK\n"
            "KVAF_A5_DIRECT_TEARDOWN_OK role=device\n",
            encoding="utf-8",
        )

    def write_manifest(self, **updates: str) -> None:
        values = {
            "aicpu_affinity_run_mode": "risk-accepted-cross-npu2-v1",
            "kvaf_commit": SOURCE_COMMIT,
            "cross_npu_artifact_id": ARTIFACT_ID,
            "cross_npu_source_run_id": SOURCE_RUN_ID,
            "cross_npu_source_quarantine_sha256": self.source_sha,
            "run_id": TARGET_RUN_ID,
            "logical_device_id": "2",
            "physical_device_id": "2",
            "npu_idle_preflight_override": "0",
            "host_eid": "00000000007f040000100000df0a0b00",
            "device_eid": "000000000000020000100000df004100",
            "host_urma_device": "udmac1d1e4",
            "host_eid_index": "0",
            "device_urma_device": "udmac0d0e2",
            "device_eid_index": "0",
            "management_mode": "ub",
            "management_jetty_id": "502",
            "h2d_transfer_count": "0",
            "h2d_bytes_per_transfer": "1024",
            "h2d_batch_iterations": "1",
            "h2d_descriptor_capacity": "1",
        }
        values.update(updates)
        (self.result / "manifest.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def write_final(self, role: str, **updates: str) -> None:
        values = {
            "schema": "kvaf-a5-direct-final-v1",
            "run_id": TARGET_RUN_ID,
            "role": role,
            "pid": "101" if role == "host" else "102",
            "outcome": "PASS",
            "reason": "success",
            "resource_armed": "1",
            "stream_quiesced": "1",
            "queue_drained": "1",
            "provider_closed": "1",
            "backing_released": "1",
            "host_mapping_unregistered": "1",
            "allocations_released": "1",
        }
        values.update(updates)
        (self.exchange / f"{role}.final").write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def validate(self) -> str:
        return GATE.validate_zero_wqe_pass(
            result_dir=self.result,
            source_run_id=SOURCE_RUN_ID,
            source_quarantine_sha256=self.source_sha,
            artifact_id=ARTIFACT_ID,
            source_commit=SOURCE_COMMIT,
            state_root=self.state,
            source_quarantine=self.source_quarantine,
        )

    def test_accepts_complete_bound_zero_wqe_pass(self) -> None:
        self.assertEqual(self.validate(), TARGET_RUN_ID)

    def test_rejects_incomplete_cleanup_and_target_quarantine(self) -> None:
        self.write_final("device", provider_closed="0")
        with self.assertRaisesRegex(GATE.StageGateError, "provider_closed"):
            self.validate()
        self.write_final("device")
        lane = self.results.parent
        (lane / "direct-urma.quarantine").write_text(
            "unsafe\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(GATE.StageGateError, "not clean"):
            self.validate()

    def test_rejects_source_quarantine_change(self) -> None:
        self.source_quarantine.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(GATE.StageGateError, "changed"):
            self.validate()

    def test_entry_scripts_are_valid_and_npu2_is_source_owned(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        for script in (WRAPPER, COORDINATOR, BUILD):
            result = subprocess.run(
                [bash, "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("risk-accepted-cross-npu2-v1", source)
        self.assertIn("KVAF_LOGICAL_DEVICE_ID=2", source)
        self.assertIn("KVAF_PHYSICAL_DEVICE_ID=2", source)
        self.assertIn("KVAF_DIRECT_MANAGEMENT_JETTY_ID=502", source)
        self.assertIn("unset KVAF_HOST_URMA_DEVICE", source)
        self.assertIn("unset KVAF_HOST_EID_INDEX", source)
        self.assertIn("unset KVAF_DEVICE_URMA_DEVICE", source)
        self.assertIn("unset KVAF_DEVICE_EID_INDEX", source)
        self.assertNotIn("export KVAF_DEVICE_URMA_DEVICE=", source)
        self.assertIn(
            "export KVAF_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool",
            source,
        )
        self.assertIn("KVAF_DIRECT_ALLOW_BUSY_NPU=0", source)
        self.assertNotIn("KVAF_DIRECT_ALLOW_BUSY_NPU=1", source)
        self.assertIn("zero-wqe", source)
        self.assertIn("one-kib", source)
        self.assertNotIn("4096", source)
        coordinator = COORDINATOR.read_text(encoding="utf-8")
        self.assertIn("build_a5_direct_urma.sh", coordinator)
        self.assertIn("cross-npu2-risk-offline-v1", coordinator)
        self.assertIn("zero-wqe-pass.handoff", coordinator)
        self.assertIn("KVAF_NPU2_ONE_KIB_H2D_ACCEPTANCE=PASS", coordinator)

    def test_offline_build_is_fresh_and_does_not_package_aicpu(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn("cross-npu2-risk-offline-v1", source)
        self.assertIn("risk build requires a fresh artifact ID", source)
        self.assertIn("cross-npu2-artifact.manifest", source)
        self.assertNotIn("package_a5_direct_urma_aicpu.sh", source)

    def test_internal_risk_mode_cannot_bypass_wrapper_contract(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "I_ACCEPT_UNKNOWN_CROSS_NPU_DMA_RISK", source
        )
        self.assertIn("KVAF_CROSS_NPU_ARTIFACT_ID", source)
        self.assertIn(
            "/KVAF/experimental/artifacts/", source
        )
        self.assertIn("cross-npu2-artifact.manifest", source)
        self.assertIn("acquire_cross_npu2_risk_session_guards", source)
        self.assertIn("acquire_persistent_session_guards", source)
        host_launch = source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host'
        )
        device_launch = source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role device'
        )
        self.assertGreater(
            source.rfind("validate_cross_npu_risk_acceptance", 0, host_launch),
            0,
        )
        self.assertGreater(
            source.rfind(
                "validate_cross_npu_risk_acceptance", 0, device_launch
            ),
            host_launch,
        )


if __name__ == "__main__":
    unittest.main()

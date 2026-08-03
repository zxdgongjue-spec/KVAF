from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"
AFFINITY = ROOT / "scripts" / "a5_aicpu_affinity_guard.sh"


class CrossNpu2ReuseRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.affinity = AFFINITY.read_text(encoding="utf-8")

    def test_new_mode_has_exact_risk_and_token_contract(self) -> None:
        self.assertIn("risk-accepted-cross-npu2-reuse-v2", self.source)
        self.assertIn(
            "I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK", self.source
        )
        self.assertIn("^[0-9a-f]{16}$", self.source)
        self.assertIn(
            'source "${SCRIPT_DIR}/a5_cross_npu2_reuse_state.sh"', self.source
        )
        self.assertIn(
            'source "${SCRIPT_DIR}/a5_cross_npu2_versioned_runtime_preflight.sh"',
            self.source,
        )

    def test_claim_is_exact_and_checked_before_new_state(self) -> None:
        verifier = self.source.index("verify_cross_npu2_reuse_attempt_claim()")
        self.assertIn("attempt.selected", self.source[verifier:])
        self.assertIn("'%h:%a'", self.source[verifier:])
        self.assertIn('"${identity}" == 1:400', self.source[verifier:])
        for binding in (
            "schema=kvaf-a5-cross-npu2-reuse-attempt-v1",
            "source_run_id=",
            "prior_target_run_id=",
            "artifact_token=",
            "source_commit=",
            "execution_contract=risk-accepted-cross-npu2-reuse-v2",
        ):
            self.assertIn(binding, self.source[verifier:])

        guard_start = self.source.index("acquire_cross_npu2_reuse_session_guards()")
        guard_end = self.source.index(
            "acquire_cross_npu2_risk_session_guards()", guard_start
        )
        guard = self.source[guard_start:guard_end]
        self.assertLess(
            guard.index("validate_cross_npu2_reuse_acceptance"),
            guard.index('mkdir -p "${path}"'),
        )

    def test_old_incident_lanes_are_read_only(self) -> None:
        guard_start = self.source.index("acquire_cross_npu2_reuse_session_guards()")
        guard_end = self.source.index(
            "acquire_cross_npu2_risk_session_guards()", guard_start
        )
        guard = self.source[guard_start:guard_end]
        self.assertNotIn("KVAF_CROSS_NPU2_REUSE_PRIOR_ACTIVE_FILE}", guard)
        self.assertNotIn("KVAF_CROSS_NPU2_REUSE_SOURCE_QUARANTINE_FILE}", guard)
        self.assertNotIn("kill ", guard)
        self.assertNotIn("rm ", guard)
        self.assertIn("KVAF_CROSS_NPU2_REUSE_LOCK_FILE", guard)

    def test_versioned_artifacts_and_fresh_jetty_are_pinned(self) -> None:
        self.assertIn(
            "kvaf_verify_cross_npu2_versioned_runtime_artifacts", self.source
        )
        self.assertIn(
            '"libkvaf_ud_${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.so"',
            self.source,
        )
        self.assertIn(
            'cann-kvaf-du-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.tar.gz',
            self.source,
        )
        self.assertIn("KVAF_DIRECT_MANAGEMENT_JETTY_ID=602", self.source)
        self.assertIn(
            '"${KVAF_DIRECT_MANAGEMENT_JETTY_ID}" != 602', self.source
        )

    def test_reuse_mode_is_strict_zero_wqe_and_idle_only(self) -> None:
        self.assertIn("KVAF_H2D_TRANSFER_COUNT != 0", self.source)
        self.assertIn("KVAF_H2D_BYTES_PER_TRANSFER != 1024", self.source)
        self.assertIn("KVAF_H2D_BATCH_ITERATIONS != 1", self.source)
        self.assertIn("KVAF_H2D_DESCRIPTOR_CAPACITY != 1", self.source)
        self.assertIn(
            '"${KVAF_DIRECT_ALLOW_BUSY_NPU}" != 0', self.source
        )
        for endpoint in (
            "00000000007f040000100000df0a0b00",
            "000000000000020000100000df004100",
            "udmac1d1e4",
            "udmac0d0e2",
        ):
            self.assertIn(endpoint, self.source)

    def test_pinned_evidence_is_revalidated_before_both_forks(self) -> None:
        host_fork = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host'
        )
        device_fork = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role device'
        )
        host_validation = self.source.rfind(
            "validate_cross_npu_risk_acceptance", 0, host_fork
        )
        device_validation = self.source.rfind(
            "validate_cross_npu_risk_acceptance", host_fork, device_fork
        )
        self.assertGreater(host_validation, 0)
        self.assertGreater(device_validation, host_fork)

    def test_affinity_guard_allows_only_the_named_experimental_mode(self) -> None:
        self.assertIn("risk-accepted-cross-npu2-reuse-v2", self.affinity)
        command = (
            f"source {AFFINITY}; "
            "kvaf_require_aicpu_affinity_contract "
            "risk-accepted-cross-npu2-reuse-v2"
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PRODUCTION_SAFETY_CLAIM=none", result.stderr)

    def test_changed_shell_files_are_valid_bash(self) -> None:
        for script in (RUNNER, AFFINITY):
            result = subprocess.run(
                ["bash", "-n", str(script)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

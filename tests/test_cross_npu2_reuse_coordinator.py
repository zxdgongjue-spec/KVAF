from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "scripts" / "run_a5_cross_npu2_reuse_experiment.sh"


class CrossNpu2ReuseCoordinatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = COORDINATOR.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(COORDINATOR)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_only_zero_wqe_is_accepted(self) -> None:
        environment = os.environ.copy()
        environment.pop("KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE", None)
        for arguments in ([], ["one-kib"], ["zero-wqe", "again"]):
            completed = subprocess.run(
                ["bash", str(COORDINATOR), *arguments],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("Usage:", completed.stderr)

    def test_exact_second_order_risk_phrase_and_token_are_required(self) -> None:
        self.assertIn(
            "I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK", self.source
        )
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE", self.source
        )
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN", self.source
        )
        self.assertIn("^[0-9a-f]{16}$", self.source)
        self.assertIn("/proc/sys/kernel/random/uuid", self.source)

    def test_evidence_precedes_every_mutating_or_hardware_stage(self) -> None:
        initial = self.source.index("kvaf_validate_reuse_evidence initial")
        provider = self.source.index(
            '"${SCRIPT_DIR}/build_a5_direct_urma_provider.sh"'
        )
        claim = self.source.index("kvaf_publish_one_shot_claim\n")
        provider_recheck = self.source.index(
            "kvaf_validate_reuse_evidence after-provider-build"
        )
        package = self.source.index(
            '"${SCRIPT_DIR}/package_a5_cross_npu2_reuse_aicpu.sh"'
        )
        package_recheck = self.source.index(
            "kvaf_validate_reuse_evidence after-package-install"
        )
        direct = self.source.index('"${SCRIPT_DIR}/build_a5_direct_urma.sh"')
        direct_recheck = self.source.index(
            "kvaf_validate_reuse_evidence after-direct-build"
        )
        probe_precheck = self.source.index(
            "kvaf_validate_reuse_evidence before-owner-probe"
        )
        probe = self.source.index('"${KVAF_A5_OWNER_PROBE_BINARY}"')
        probe_recheck = self.source.index(
            "kvaf_validate_reuse_evidence after-owner-probe"
        )
        run_precheck = self.source.index(
            "kvaf_validate_reuse_evidence before-zero-wqe-run"
        )
        runner = self.source.index('"${SCRIPT_DIR}/run_a5_direct_urma_h2d_smoke.sh"')
        self.assertLess(initial, provider)
        self.assertLess(provider, provider_recheck)
        self.assertLess(provider_recheck, package)
        self.assertLess(package, package_recheck)
        self.assertLess(package_recheck, direct)
        self.assertLess(direct, direct_recheck)
        self.assertLess(direct_recheck, probe_precheck)
        self.assertLess(probe_precheck, claim)
        self.assertLess(claim, probe)
        self.assertLess(probe, probe_recheck)
        self.assertLess(probe_recheck, run_precheck)
        self.assertLess(run_precheck, runner)

    def test_artifacts_are_versioned_and_runtime_preflight_is_mandatory(self) -> None:
        for required in (
            "cross-npu2-reuse-versioned-offline-v1",
            "package_a5_cross_npu2_reuse_aicpu.sh",
            "kvaf_verify_cross_npu2_versioned_runtime_artifacts",
            "libkvaf_ud_",
            "cann-kvaf-du-",
            "--provider-soname",
        ):
            self.assertIn(required, self.source)
        self.assertNotIn("run_a5_aicpu_owner_probe.sh", self.source)

    def test_checkout_is_clean_and_commit_stable_before_hardware(self) -> None:
        self.assertIn("kvaf_require_clean_source()", self.source)
        self.assertIn("diff-index --quiet HEAD --", self.source)
        self.assertIn("ls-files --others --exclude-standard", self.source)
        self.assertGreaterEqual(
            self.source.count("kvaf_require_clean_source\n"),
            3,
        )

    def test_attempt_is_atomically_claimed_once_before_hardware_probe(self) -> None:
        for required in (
            "attempt.selected",
            "schema=kvaf-a5-cross-npu2-reuse-attempt-v1",
            "source_run_id=${kvaf_source_run_id}",
            "prior_target_run_id=${kvaf_prior_run_id}",
            "artifact_token=${kvaf_token}",
            "source_commit=${kvaf_source_commit}",
            'ln "${temporary}" "${claim}"',
            "chmod 0400",
            "A new token and a same-token retry are both forbidden.",
        ):
            self.assertIn(required, self.source)
        claim = self.source.index("kvaf_publish_one_shot_claim\n")
        direct = self.source.index('"${SCRIPT_DIR}/build_a5_direct_urma.sh"')
        probe = self.source.index('"${KVAF_A5_OWNER_PROBE_BINARY}"')
        self.assertLess(direct, claim)
        self.assertLess(claim, probe)

    def test_npu2_probe_and_zero_wqe_are_strictly_bounded(self) -> None:
        for required in (
            "--logical-device-id 2",
            "--physical-device-id 2",
            "export KVAF_LOGICAL_DEVICE_ID=2",
            "export KVAF_PHYSICAL_DEVICE_ID=2",
            "export KVAF_H2D_TRANSFER_COUNT=0",
            "export KVAF_H2D_BYTES_PER_TRANSFER=1024",
            "export KVAF_H2D_BATCH_ITERATIONS=1",
            "export KVAF_H2D_DESCRIPTOR_CAPACITY=1",
            "export KVAF_H2D_RUN_TIMEOUT_SECONDS=180",
            "export KVAF_DIRECT_ALLOW_BUSY_NPU=0",
        ):
            self.assertIn(required, self.source)
        self.assertNotIn("KVAF_DIRECT_ALLOW_BUSY_NPU=1", self.source)

    def test_fixed_npu2_transport_identity_uses_new_jetty(self) -> None:
        for required in (
            "KVAF_HOST_EID=00000000007f040000100000df0a0b00",
            "KVAF_DEVICE_EID=000000000000020000100000df004100",
            "KVAF_HOST_URMA_DEVICE=udmac1d1e4",
            "KVAF_DEVICE_URMA_DEVICE=udmac0d0e2",
            "KVAF_DIRECT_MANAGEMENT_MODE=ub",
            "KVAF_DIRECT_MANAGEMENT_JETTY_ID=602",
            "risk-accepted-cross-npu2-reuse-v2",
        ):
            self.assertIn(required, self.source)

    def test_coordinator_has_no_incident_cleanup_or_retry_loop(self) -> None:
        sanitized = "\n".join(
            line for line in self.source.splitlines() if not line.lstrip().startswith("#")
        )
        for forbidden in (
            r"\bkill\b",
            r"\bpkill\b",
            r"\breset\b",
            r"\bunlink\b",
        ):
            self.assertIsNone(re.search(forbidden, sanitized), forbidden)
        for protected in (
            "rm -f /data/.kvaf-state",
            'rm -f "${KVAF_DIRECT_ACTIVE_SESSION_FILE}"',
            'rm -f "${KVAF_DIRECT_QUARANTINE_FILE}"',
        ):
            self.assertNotIn(protected, sanitized)
        self.assertNotIn("while ", sanitized)
        self.assertNotIn("until ", sanitized)
        self.assertNotIn("direct-urma.quarantine\"", sanitized)


if __name__ == "__main__":
    unittest.main()

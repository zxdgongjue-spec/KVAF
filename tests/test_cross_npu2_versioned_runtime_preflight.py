from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "a5_cross_npu2_versioned_runtime_preflight.sh"
COORDINATOR = ROOT / "scripts" / "run_a5_cross_npu2_reuse_experiment.sh"
RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"


class CrossNpu2VersionedRuntimePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = PREFLIGHT.read_text(encoding="utf-8")
        cls.coordinator = COORDINATOR.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        completed = subprocess.run(
            [bash, "-n", str(PREFLIGHT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_provider_identity_binds_mode_filename_hash_and_soname(self) -> None:
        for required in (
            "kvaf-a5-cross-npu2-versioned-provider-v1",
            "cross-npu2-reuse-versioned-offline-v1",
            "experiment_profile",
            'libkvaf_ud_${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.so',
            "device_provider_sha256",
            "Device provider SONAME is not unique",
        ):
            self.assertIn(required, self.preflight)

    def test_direct_profile_is_bound_to_live_reuse_evidence(self) -> None:
        for required in (
            "reuse_evidence_sha256",
            "KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256",
            "direct build and runtime evidence generations differ",
            "artifact_mutation_policy:forbidden-during-hardware-run",
        ):
            self.assertIn(required, self.preflight)
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256=${evidence_sha}",
            self.coordinator,
        )

    def test_unique_package_registration_and_entries_are_exact(self) -> None:
        for required in (
            'cann-kvaf-du-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.tar.gz',
            "kvaf_aicpu_registration_is_exact",
            "versioned archive entries are inconsistent",
            "archive entry hash mismatch",
            "KVAF_VERIFIED_AICPU_ARCHIVE_SHA",
        ):
            self.assertIn(required, self.preflight)

    def test_both_hardware_entries_run_the_same_versioned_preflight(self) -> None:
        self.assertIn(
            "kvaf_verify_cross_npu2_versioned_runtime_artifacts",
            self.coordinator,
        )
        self.assertIn(
            "kvaf_verify_cross_npu2_versioned_runtime_artifacts",
            self.runner,
        )


if __name__ == "__main__":
    unittest.main()

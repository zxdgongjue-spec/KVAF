#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
BUILD = ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
CMAKE = ROOT / "access_operator" / "a5" / "provider" / "CMakeLists.txt"
MODE = "cross-npu2-reuse-versioned-offline-v1"
ACCEPTANCE = "I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK"


class VersionedProviderBuildTests(unittest.TestCase):
    def run_build(self, *arguments: str, **updates: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        for name in (
            "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE",
            "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN",
            "KVAF_DIRECT_PROVIDER_BUILD_DIR",
        ):
            environment.pop(name, None)
        environment.update(updates)
        return subprocess.run(
            ["bash", str(BUILD), *arguments],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_shell_syntax_and_invalid_mode_fail_closed(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        syntax = subprocess.run(
            [bash, "-n", str(BUILD)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        invalid = self.run_build("not-a-provider-mode")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid direct URMA provider build mode", invalid.stderr)

    def test_versioned_mode_requires_exact_acceptance_and_token(self) -> None:
        missing = self.run_build(MODE)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("exact second-order risk acceptance", missing.stderr)

        wrong = self.run_build(
            MODE,
            KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE=ACCEPTANCE,
            KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN="A" * 16,
        )
        self.assertEqual(wrong.returncode, 2)
        self.assertIn("16 lowercase hexadecimal digits", wrong.stderr)

    def test_cmake_owns_versioned_output_name_and_soname(self) -> None:
        source = CMAKE.read_text(encoding="utf-8")
        self.assertIn("KVAF_DEVICE_PROVIDER_OUTPUT_NAME", source)
        self.assertIn('"^kvaf_ud_[0-9a-f]+$"', source)
        self.assertIn("EQUAL 24", source)
        self.assertIn('"-Wl,-soname,lib${OUTPUT_BASENAME}.so"', source)
        self.assertIn('"${KVAF_DEVICE_PROVIDER_OUTPUT_NAME}"', source)

    def test_experiment_path_lock_and_dynamic_device_name_are_fixed(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn(MODE, source)
        self.assertIn("KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE", source)
        self.assertIn(ACCEPTANCE, source)
        self.assertIn("KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN", source)
        self.assertIn(
            "/data/.kvaf-state/KVAF/experimental/reuse-artifacts/",
            source,
        )
        self.assertIn('kvaf_device_provider_output_name="kvaf_ud_${', source)
        self.assertIn('kvaf_experiment_artifact_dir}/provider-build.lock', source)
        self.assertIn("flock -n 9", source)
        self.assertIn("the versioned provider build requires a fresh artifact token", source)

        experiment_branch = source.split(
            "if (( kvaf_versioned_experiment_build != 0 )); then", 2
        )[2].split("else", 1)[0]
        self.assertNotIn("KVAF_DIRECT_DEPLOY_RUN_LOCK", experiment_branch)
        self.assertNotIn("kvaf_check_legacy_direct_urma_state", experiment_branch)

    def test_sealed_profile_contains_all_bound_artifact_hashes(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn(
            "schema=kvaf-a5-cross-npu2-versioned-provider-v1", source
        )
        for key in (
            "mode",
            "artifact_token",
            "source_commit",
            "provider_dir",
            "host_provider_filename",
            "host_provider_sha256",
            "device_provider_filename",
            "device_provider_sha256",
            "source_manifest_filename",
            "source_manifest_sha256",
            "build_manifest_filename",
            "build_manifest_sha256",
            "success_stamp_filename",
            "success_stamp_sha256",
        ):
            self.assertIn(f"{key}=", source)
        self.assertIn('chmod 0500 "${KVAF_DIRECT_PROVIDER_BUILD_DIR}"', source)
        self.assertIn(
            '"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/experiment-provider.profile"',
            source,
        )

    def test_production_still_uses_canonical_names_and_global_guards(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn(
            "kvaf_device_provider_filename=libkvaf_urma_provider_device.so",
            source,
        )
        self.assertIn("kvaf_configure_direct_urma_state_paths", source)
        self.assertIn("kvaf_check_legacy_direct_urma_state", source)
        self.assertIn("KVAF_DIRECT_DEPLOY_RUN_LOCK", source)
        self.assertIn("KVAF_DIRECT_ACTIVE_SESSION_LOCK", source)


if __name__ == "__main__":
    unittest.main()

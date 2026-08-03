#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts" / "package_a5_cross_npu2_reuse_aicpu.sh"
REGISTRATION = ROOT / "scripts" / "a5_aicpu_package_registration.sh"
RISK_PHRASE = "I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK"


class CrossNpu2VersionedAicpuPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PACKAGER.read_text(encoding="utf-8")

    def test_script_has_valid_bash_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(PACKAGER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_risk_phrase_and_token_are_required_before_mutation(self) -> None:
        self.assertIn(RISK_PHRASE, self.source)
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE", self.source
        )
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN", self.source
        )
        self.assertIn("^[0-9a-f]{16}$", self.source)
        risk_gate = self.source.index(
            '[[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}"'
        )
        token_gate = self.source.index(
            '[[ "${kvaf_artifact_token}" =~ ^[0-9a-f]{16}$ ]]'
        )
        archive_publish = self.source.index(
            'ln -- "${kvaf_install_candidate}" "${kvaf_archive_path}"'
        )
        registration_publish = self.source.index(
            'kvaf_aicpu_install_registration "${kvaf_registration_file}"'
        )
        self.assertLess(risk_gate, archive_publish)
        self.assertLess(token_gate, archive_publish)
        self.assertLess(archive_publish, registration_publish)

    def test_bad_acceptance_and_token_fail_before_environment_access(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        base_env = os.environ.copy()
        base_env.pop("KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE", None)
        base_env.pop("KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN", None)
        result = subprocess.run(
            [bash, "--noprofile", "--norc", str(PACKAGER)],
            env=base_env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("exact NPU2 reuse/shared UMMU risk phrase", result.stderr)

        base_env["KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE"] = RISK_PHRASE
        base_env["KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN"] = "NOT-HEX"
        result = subprocess.run(
            [bash, "--noprofile", "--norc", str(PACKAGER)],
            env=base_env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("16 lowercase hexadecimal digits", result.stderr)

    def test_provider_and_package_profiles_are_source_owned_and_sealed(self) -> None:
        for required in (
            "/data/.kvaf-state/KVAF/experimental/reuse-artifacts",
            "${kvaf_source_commit}/${kvaf_artifact_token}",
            'kvaf_provider_profile="${kvaf_provider_dir}/experiment-provider.profile"',
            "kvaf-a5-cross-npu2-versioned-provider-v1",
            "kvaf-a5-cross-npu2-versioned-package-v1",
            "schema mode experiment_profile artifact_token",
            '[[ "${profile_mode}" == 400 ]]',
            "provider_profile_path",
            "provider_profile_sha256",
            "archive_path",
            "archive_sha256",
            "registration_file",
        ):
            self.assertIn(required, self.source)
        self.assertIn(
            'kvaf_package_profile="${kvaf_artifact_root}/experiment-package.profile"',
            self.source,
        )
        self.assertIn(
            'ln -- "${kvaf_candidate_profile}" "${kvaf_package_profile}"',
            self.source,
        )
        self.assertNotIn(
            'mv -f -- "${kvaf_candidate_profile}"', self.source
        )

    def test_package_and_every_archive_entry_are_token_namespaced(self) -> None:
        self.assertIn(
            'kvaf_package_name="cann-kvaf-du-${kvaf_artifact_token}.tar.gz"',
            self.source,
        )
        self.assertIn(
            '"libkvaf_ud_${kvaf_artifact_token}.so"', self.source
        )
        for stem in (
            "kvaf-du-provider-",
            "kvaf-du-source-",
            "kvaf-du-build-",
            "kvaf-du-success-",
        ):
            self.assertIn(f'"{stem}${{kvaf_artifact_token}}', self.source)
        self.assertIn(
            '(( ${#kvaf_device_provider_filename} < 32 ))', self.source
        )
        self.assertIn("archive_device_entry", self.source)
        self.assertIn("archive has missing, colliding, or unexpected", self.source)

    def test_archive_is_deterministic_and_published_without_overwrite(self) -> None:
        for option in (
            "--sort=name",
            "--mtime='@0'",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
        ):
            self.assertIn(option, self.source)
        self.assertIn(
            'ln -- "${kvaf_install_candidate}" "${kvaf_archive_path}"',
            self.source,
        )
        self.assertNotRegex(
            self.source,
            re.compile(r"mv\s+-f[^\n]*\$\{kvaf_archive_path\}"),
        )

    def test_canonical_package_is_never_replaced_or_removed(self) -> None:
        self.assertIn("cann-kvaf-direct-urma-compat.tar.gz", self.source)
        self.assertIn("verify_canonical_archive_unchanged", self.source)
        forbidden = (
            'rm -f -- "${kvaf_canonical_archive}"',
            'mv -f "${kvaf_canonical_archive}"',
            'cp -- "${kvaf_candidate_archive}" "${kvaf_canonical_archive}"',
            "package_a5_direct_urma_aicpu.sh",
        )
        for text in forbidden:
            self.assertNotIn(text, self.source)

    def test_registry_uses_independent_lock_and_exact_append_proof(self) -> None:
        self.assertIn("versioned-aicpu-registry.lock", self.source)
        self.assertIn('exec 9>"${kvaf_registry_lock}"', self.source)
        self.assertIn("flock -w 30 9", self.source)
        self.assertNotIn("KVAF_DIRECT_DEPLOY_RUN_LOCK", self.source)
        self.assertNotIn("KVAF_DIRECT_ACTIVE_SESSION_LOCK", self.source)
        self.assertNotIn("kvaf_check_legacy_direct_urma_state", self.source)
        for required in (
            "kvaf_aicpu_registration_is_absent",
            "kvaf_aicpu_registration_is_exact",
            "kvaf_aicpu_registration_block",
            "kvaf_aicpu_install_registration",
            'cmp -s -- "${kvaf_registry_expected}" "${kvaf_registration_file}"',
            "the AICPU registry changed by more than the one versioned stanza",
        ):
            self.assertIn(required, self.source)
        helper = REGISTRATION.read_text(encoding="utf-8")
        self.assertIn("cp -p --", helper)
        self.assertIn("mv -f --", helper)

    def test_profile_declares_required_runtime_handoff_fields(self) -> None:
        for assignment in (
            '"artifact_token=${kvaf_artifact_token}"',
            '"source_commit=${kvaf_source_commit}"',
            '"provider_profile_path=${kvaf_provider_profile}"',
            '"provider_profile_sha256=${kvaf_provider_profile_sha}"',
            '"package_name=${kvaf_package_name}"',
            '"package_relative_dir=${kvaf_package_relative_dir}"',
            '"archive_path=${kvaf_archive_path}"',
            '"archive_sha256=${kvaf_archive_sha}"',
            '"registration_file=${kvaf_registration_file}"',
            '"device_provider_filename=${kvaf_device_provider_filename}"',
            '"device_provider_sha256=${kvaf_device_provider_sha}"',
        ):
            self.assertIn(assignment, self.source)


if __name__ == "__main__":
    unittest.main()

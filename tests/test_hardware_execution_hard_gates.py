from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HardwareBinaryHardGateTests(unittest.TestCase):
    SOURCES = (
        (
            "direct-worker",
            ROOT / "access_operator/a5/smoke/aiv_simt_direct_urma_h2d_smoke.cpp",
            "KVAF_A5_DIRECT_BINARY_HARDWARE_EXECUTION_DISABLED",
            ("ParseOptions(", "aclInit(", "aclrtSetDevice(", "RunHost(", "RunDevice("),
        ),
        (
            "runtime-probe",
            ROOT / "access_operator/a5/smoke/aicpu_runtime_urma_probe.cpp",
            "KVAF_A5_RUNTIME_URMA_BINARY_DYNAMIC_PROBE_DISABLED",
            ("ParseOptions(", "RunProbe("),
        ),
        (
            "owner-probe",
            ROOT / "access_operator/a5/smoke/aicpu_owner_concurrency_probe.cpp",
            "KVAF_A5_AICPU_OWNER_BINARY_DYNAMIC_PROBE_DISABLED",
            ("ParseOptions(", "RunProbe("),
        ),
        (
            "legacy-hcomm",
            ROOT / "access_operator/a5/smoke/aiv_simt_urma_h2d_smoke.cpp",
            "KVAF_A5_HCOMM_BINARY_HARDWARE_EXECUTION_DISABLED",
            ("ParseOptions(", "aclInit(", "aclrtSetDevice(", "RunHost(", "RunDevice("),
        ),
    )

    def test_every_resource_capable_main_returns_two_before_parsing_or_acl(self) -> None:
        for label, path, marker, later_operations in self.SOURCES:
            with self.subTest(binary=label):
                source = path.read_text(encoding="utf-8")
                main = source[source.index("int main(int argc, char** argv)") :]
                gate = main.index(marker)
                fixed_return = main.index("return 2;", gate)

                self.assertLess(gate, fixed_return)
                self.assertNotRegex(
                    main[:fixed_return],
                    re.compile(r"\b(?:if|for|while|switch)\s*\("),
                    "the binary gate must not depend on a runtime branch",
                )
                self.assertNotIn("getenv", main[:fixed_return])
                for operation in later_operations:
                    self.assertLess(
                        fixed_return,
                        main.index(operation, fixed_return),
                        f"{operation} must remain unreachable behind the fixed return",
                    )


class HardwareShellHardGateTests(unittest.TestCase):
    RUNNERS = (
        (
            "direct-worker",
            ROOT / "scripts/run_a5_direct_urma_h2d_smoke.sh",
            "KVAF_A5_DIRECT_HARDWARE_EXECUTION_DISABLED",
            ': "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"',
        ),
        (
            "runtime-probe",
            ROOT / "scripts/run_a5_aicpu_runtime_urma_probe.sh",
            "KVAF_A5_RUNTIME_URMA_DYNAMIC_PROBE_DISABLED",
            ': "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"',
        ),
        (
            "owner-probe",
            ROOT / "scripts/run_a5_aicpu_owner_probe.sh",
            "KVAF_A5_AICPU_OWNER_DYNAMIC_PROBE_DISABLED",
            'source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"',
        ),
        (
            "legacy-hcomm",
            ROOT / "scripts/run_a5_aiv_urma_h2d_smoke.sh",
            "KVAF_A5_HCOMM_HARDWARE_EXECUTION_DISABLED",
            "KVAF_A5_BUILD_DIR=",
        ),
        (
            "canonical-aicpu-packager",
            ROOT / "scripts/package_a5_direct_urma_aicpu.sh",
            "KVAF_A5_DIRECT_AICPU_PACKAGE_MUTATION_DISABLED",
            'SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)',
        ),
        (
            "retired-cross-npu2-packager",
            ROOT / "scripts/package_a5_cross_npu2_reuse_aicpu.sh",
            "KVAF_A5_CROSS_NPU2_AICPU_PACKAGE_MUTATION_DISABLED",
            'source "${SCRIPT_DIR}/a5_aicpu_package_registration.sh"',
        ),
        (
            "retired-host-jfr-trace",
            ROOT / "scripts/trace_a5_direct_urma_npu2_jfr_failure.sh",
            "KVAF_JFR_TRACE_RETIRED=host-trace-cannot-observe-kvaf-aicpu-device-os",
            'SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)',
        ),
    )

    @staticmethod
    def gate_slice(source: str, marker: str) -> str:
        lines = source.splitlines()
        defense = [
            "unset BASH_ENV ENV",
            "unset -f exit builtin 2>/dev/null || true",
        ]
        gate_line = next(
            index for index, line in enumerate(lines) if marker in line
        )
        branch_line = next(
            index
            for index in range(gate_line, len(lines))
            if lines[index] == "if [[ 0 == 1 ]]; then"
        )
        return "\n".join(
            ["#!/usr/bin/env bash", "set -Eeuo pipefail"]
            + defense
            + lines[gate_line : branch_line + 1]
            + ['printf hardware-entered >"${KVAF_HARDWARE_SENTINEL}"', "fi"]
        ) + "\n"

    def test_every_runner_wraps_the_complete_old_body_in_constant_false(self) -> None:
        for label, path, marker, hardware_operation in self.RUNNERS:
            with self.subTest(runner=label):
                source = path.read_text(encoding="utf-8")
                clear_env = source.index("unset BASH_ENV ENV")
                clear_functions = source.index(
                    "unset -f exit builtin 2>/dev/null || true"
                )
                gate = source.index(marker)
                fixed_exit = source.index("builtin exit 2", gate)
                false_branch = source.index("if [[ 0 == 1 ]]; then", fixed_exit)
                hardware = source.index(hardware_operation, false_branch)

                self.assertEqual(
                    [clear_env, clear_functions, gate, fixed_exit, false_branch, hardware],
                    sorted(
                        [clear_env, clear_functions, gate, fixed_exit, false_branch, hardware]
                    ),
                )
                self.assertNotIn("${KVAF_", source[gate:fixed_exit])
                self.assertTrue(
                    source.rstrip().endswith("fi\n/bin/sh -c 'exit 2'"),
                    "the constant-false legacy body must extend to end of file",
                )

    def test_bash_env_exit_function_cannot_enter_any_old_hardware_body(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")

        for label, path, marker, _ in self.RUNNERS:
            with self.subTest(runner=label), tempfile.TemporaryDirectory() as temporary:
                temporary_path = Path(temporary)
                bash_env = temporary_path / "bash-env"
                hook_loaded = temporary_path / "hook-loaded"
                sentinel = temporary_path / "hardware-entered"
                harness = temporary_path / "gate-harness.sh"
                bash_env.write_text(
                    'printf loaded >"${KVAF_HOOK_LOADED}"\n'
                    'exit() { printf shadow-exit >"${KVAF_HARDWARE_SENTINEL}"; return 0; }\n',
                    encoding="utf-8",
                )
                harness.write_text(
                    self.gate_slice(path.read_text(encoding="utf-8"), marker),
                    encoding="utf-8",
                )
                environment = os.environ.copy()
                environment.update(
                    {
                        "BASH_ENV": str(bash_env),
                        "KVAF_HOOK_LOADED": str(hook_loaded),
                        "KVAF_HARDWARE_SENTINEL": str(sentinel),
                    }
                )
                result = subprocess.run(
                    [bash, "--noprofile", "--norc", str(harness)],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue(hook_loaded.is_file(), "BASH_ENV was not exercised")
                self.assertFalse(
                    sentinel.exists(),
                    "the imported exit function or old hardware body ran",
                )

    def test_packagers_exit_before_environment_or_argument_validation(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")

        packagers = (
            (
                ROOT / "scripts/package_a5_direct_urma_aicpu.sh",
                "KVAF_A5_DIRECT_AICPU_PACKAGE_MUTATION_DISABLED",
                (),
            ),
            (
                ROOT / "scripts/package_a5_cross_npu2_reuse_aicpu.sh",
                "KVAF_A5_CROSS_NPU2_AICPU_PACKAGE_MUTATION_DISABLED",
                ("unexpected-argument",),
            ),
        )
        for path, marker, arguments in packagers:
            with self.subTest(packager=path.name), tempfile.TemporaryDirectory() as temporary:
                impossible_home = Path(temporary) / "must-not-be-created"
                environment = os.environ.copy()
                environment.pop("ASCEND_HOME_PATH", None)
                environment["KVAF_GLOBAL_STATE_DIR"] = str(impossible_home)
                result = subprocess.run(
                    [bash, "--noprofile", "--norc", str(path), *arguments],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(marker, result.stderr)
                self.assertIn("CAPABILITY_LEDGER=not-implemented", result.stderr)
                self.assertFalse(impossible_home.exists())


class AicpuPackagePublicationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = (
            ROOT / "scripts/package_a5_direct_urma_aicpu.sh"
        ).read_text(encoding="utf-8")

    def test_registry_rollback_and_recovery_use_one_pinned_real_path(self) -> None:
        resolve = self.package.index(
            "registration_real_path=$(realpath -e --"
        )
        cleanup = self.package.index("cleanup()\n{", resolve)
        restore = self.package.index(
            '"${registration_real_path}"; then', cleanup
        )
        recovery = self.package.index(
            '$(dirname -- "${registration_real_path}")/'
            '.ascend_package_load.ini.kvaf-recovery.XXXXXX',
            restore,
        )
        self.assertLess(resolve, cleanup)
        self.assertLess(cleanup, restore)
        self.assertLess(restore, recovery)
        self.assertNotIn(
            '$(dirname -- "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}")/'
            '.ascend_package_load.ini.kvaf-recovery.XXXXXX',
            self.package[cleanup:],
        )

    def test_operator_backup_is_verified_then_durably_published_atomically(self) -> None:
        start = self.package.index(
            'operator_backup="${KVAF_DIRECT_AICPU_ARCHIVE}.before-kvaf-direct"'
        )
        end = self.package.index("temporary_install_archive=", start)
        block = self.package[start:end]
        staged = block.index("operator_backup_candidate=$(mktemp")
        copied = block.index('"${operator_backup_candidate}" ||', staged)
        verified = block.index(
            'sha256sum "${operator_backup_candidate}"', copied
        )
        target_rechecked = block.index(
            '[[ ! -e "${operator_backup}" && ! -L "${operator_backup}" ]]',
            verified,
        )
        published = block.index(
            '"${operator_backup_candidate}" "${operator_backup}"',
            target_rechecked,
        )
        candidate_cleared = block.index(
            "operator_backup_candidate=", published
        )
        committed = block.index("operator_backup_created=1", candidate_cleared)
        self.assertEqual(
            [staged, copied, verified, target_rechecked, published, candidate_cleared, committed],
            sorted(
                [staged, copied, verified, target_rechecked, published, candidate_cleared, committed]
            ),
        )
        self.assertIn(
            'python3 "${KVAF_DURABLE_STATE_HELPER}" publish', block
        )
        self.assertNotRegex(
            block,
            re.compile(
                r"cp\s+-p(?:\s+--)?\s+\"\$\{KVAF_DIRECT_AICPU_ARCHIVE\}\"\s+"
                r"\"\$\{operator_backup\}\""
            ),
        )

        cleanup = self.package[
            self.package.index("cleanup()\n{") : self.package.index("trap cleanup EXIT")
        ]
        self.assertIn(
            'if [[ -n "${operator_backup_candidate}" ]]; then', cleanup
        )
        self.assertIn(
            'rm -f -- "${operator_backup_candidate}"', cleanup
        )

    def test_recovery_copy_is_unique_verified_and_durable_before_publication(self) -> None:
        start = self.package.index("preserve_recovery_copy()")
        end = self.package.index("\n  }", start)
        function = self.package[start:end]
        staged = function.index('recovery_file=$(mktemp "${recovery_template}")')
        copied = function.index(
            'cp -p -- "${source_file}" "${recovery_file}"', staged
        )
        verified = function.index(
            'recovery_sha=$(sha256sum -- "${recovery_file}"', copied
        )
        synced = function.index(
            'python3 "${KVAF_DURABLE_STATE_HELPER}" sync', verified
        )
        announced = function.index("_RECOVERY_COPY=%s", synced)
        self.assertEqual(
            [staged, copied, verified, synced, announced],
            sorted([staged, copied, verified, synced, announced]),
        )
        self.assertGreaterEqual(function.count('rm -f -- "${recovery_file}"'), 2)


if __name__ == "__main__":
    unittest.main()

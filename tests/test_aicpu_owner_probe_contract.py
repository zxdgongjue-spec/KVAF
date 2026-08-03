#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"
ABI_HEADER = PROVIDER / "include" / "kvaf_urma_provider_abi.h"
PROBE_SOURCE = (
    ROOT
    / "access_operator"
    / "a5"
    / "smoke"
    / "aicpu_owner_concurrency_probe.cpp"
)
PROVIDER_CMAKE = ROOT / "access_operator" / "a5" / "CMakeLists.txt"
EXPORT_MAP = PROVIDER / "provider_exports.map"
AFFINITY_GUARD = ROOT / "scripts" / "a5_aicpu_affinity_guard.sh"
DIRECT_RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"
PROVIDER_BUILD = ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
AICPU_PACKAGE = ROOT / "scripts" / "package_a5_direct_urma_aicpu.sh"
RUNTIME_PREFLIGHT = ROOT / "scripts" / "a5_direct_urma_runtime_preflight.sh"
OWNER_RUNNER = ROOT / "scripts" / "run_a5_aicpu_owner_probe.sh"
DIRECT_BUILD = ROOT / "scripts" / "build_a5_direct_urma.sh"
OWNER_DOC = ROOT / "docs" / "a5_aicpu_owner_probe.md"


def function_body(source: str, name: str) -> str:
    match = re.search(
        rf"\bint\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", source
    )
    if match is None:
        raise AssertionError(f"could not find definition of {name}")
    start = source.index("{", match.start())
    depth = 0
    for offset in range(start, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated definition of {name}")


class AicpuOwnerProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.abi = ABI_HEADER.read_text(encoding="utf-8")
        cls.cmake = PROVIDER_CMAKE.read_text(encoding="utf-8")
        cls.exports = EXPORT_MAP.read_text(encoding="utf-8")
        cls.guard = AFFINITY_GUARD.read_text(encoding="utf-8")
        cls.runner = DIRECT_RUNNER.read_text(encoding="utf-8")
        cls.probe = PROBE_SOURCE.read_text(encoding="utf-8")
        cls.provider_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(PROVIDER.glob("*.c"))
        )

    def test_owner_probe_abi_is_fixed_and_cache_line_separated(self) -> None:
        compiler = shutil.which("c++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("no C++ compiler is installed")
        unit_source = textwrap.dedent(
            """
            #include <cstdint>
            #include "kvaf_urma_provider_abi.h"

            static_assert(KVAF_URMA_OWNER_PROBE_MAGIC == 0x4b4f5052U);
            static_assert(KVAF_URMA_OWNER_PROBE_ABI_VERSION == 1U);
            static_assert(KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE == 1ULL);
            static_assert(KVAF_URMA_OWNER_PROBE_COMMAND_NONE == 0U);
            static_assert(KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE == 1U);
            static_assert(KVAF_URMA_OWNER_PROBE_STATE_EMPTY == 0U);
            static_assert(KVAF_URMA_OWNER_PROBE_STATE_READY == 1U);
            static_assert(KVAF_URMA_OWNER_PROBE_STATE_RELEASED == 2U);
            static_assert(KVAF_URMA_OWNER_PROBE_STATE_TIMED_OUT == 3U);
            static_assert(KVAF_URMA_OWNER_PROBE_STATE_FAILED == 4U);
            static_assert(KVAF_URMA_OWNER_PROBE_RESULT_OK == 0);
            static_assert(KVAF_URMA_OWNER_PROBE_RESULT_TIMED_OUT == 1);
            static_assert(KVAF_URMA_OWNER_PROBE_RESULT_CLOCK_FAILED == 2);
            static_assert(KVAF_URMA_OWNER_PROBE_RESULT_INVALID_COMMAND == 3);
            static_assert(sizeof(kvaf_urma_owner_probe_command_v1) == 64);
            static_assert(alignof(kvaf_urma_owner_probe_command_v1) == 64);
            static_assert(sizeof(kvaf_urma_owner_probe_status_v1) == 64);
            static_assert(alignof(kvaf_urma_owner_probe_status_v1) == 64);
            static_assert(sizeof(kvaf_urma_aicpu_owner_probe_args_v1) == 128);
            static_assert(alignof(kvaf_urma_aicpu_owner_probe_args_v1) == 64);

            int main() {
              auto* entry = &aicpu_kvaf_urma_owner_probe_v1;
              return entry == nullptr;
            }
            """
        )
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary) / "owner_probe_abi.cpp"
            unit.write_text(unit_source, encoding="utf-8")
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsyntax-only",
                    "-I",
                    str(PROVIDER / "include"),
                    str(unit),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_owner_probe_entry_is_declared_defined_and_exported(self) -> None:
        symbol = "aicpu_kvaf_urma_owner_probe_v1"
        self.assertIn(symbol, self.abi)
        self.assertIn(symbol, self.provider_sources)
        self.assertIn(symbol, self.exports)
        for script in (PROVIDER_BUILD, AICPU_PACKAGE, RUNTIME_PREFLIGHT):
            with self.subTest(script=script.name):
                self.assertIn(
                    symbol, script.read_text(encoding="utf-8")
                )

    def test_owner_probe_entry_is_bounded_and_resource_free(self) -> None:
        body = function_body(
            self.provider_sources, "aicpu_kvaf_urma_owner_probe_v1"
        )
        # Timeout validation and monotonic clock access deliberately live in
        # adjacent helpers; do not force those details into the exported ABI
        # entry merely to satisfy a source-shape test.
        for required in (
            "KVAF_URMA_OWNER_PROBE_MAX_TIMEOUT_MILLISECONDS",
            "KVAF_URMA_OWNER_PROBE_STATE_READY",
            "KVAF_URMA_OWNER_PROBE_STATE_RELEASED",
            "KVAF_URMA_OWNER_PROBE_STATE_TIMED_OUT",
            "KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE",
            "__ATOMIC_ACQUIRE",
            "__ATOMIC_RELEASE",
            "CLOCK_MONOTONIC",
        ):
            self.assertIn(required, self.provider_sources)
        self.assertIn("kvaf_owner_probe_args_are_valid", body)
        for forbidden_call in (
            "kvaf_urma_provider_open_v2(",
            "kvaf_urma_provider_close_v2(",
            "kvaf_udma_source_runtime_init(",
            "kvaf_source_ummu_get_status(",
            "urma_init(",
            "urma_uninit(",
        ):
            self.assertNotIn(forbidden_call, body)

    def test_cmake_exposes_a_dedicated_owner_probe_target(self) -> None:
        target = self.cmake.index(
            "add_executable(kvaf_a5_aicpu_owner_probe"
        )
        target_end = self.cmake.index("endif()", target)
        target_body = self.cmake[target:target_end]
        self.assertIn("smoke/aicpu_owner_concurrency_probe.cpp", target_body)
        self.assertIn(
            "kvaf_configure_a5_target(kvaf_a5_aicpu_owner_probe)",
            target_body,
        )
        self.assertNotIn("direct_urma_resources.cpp", target_body)
        self.assertNotIn("host_ddr_mapping.cpp", target_body)

        build_script = DIRECT_BUILD.read_text(encoding="utf-8")
        self.assertIn("--target", build_script)
        self.assertIn("kvaf_a5_aiv_direct_urma_h2d_smoke", build_script)
        self.assertIn("kvaf_a5_aicpu_owner_probe", build_script)
        self.assertIn("owner_probe_path=${owner_probe}", build_script)
        self.assertIn("owner_probe_sha256=${owner_probe_sha}", build_script)

    def test_host_probe_implements_the_ready_release_protocol(self) -> None:
        for required in (
            "aicpu_kvaf_urma_owner_probe_v1",
            "KVAF_URMA_OWNER_PROBE_STATE_READY",
            "KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE",
            "KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE",
            "KVAF_URMA_OWNER_PROBE_STATE_RELEASED",
            "LaunchAicpuOwnerConcurrencyProbe",
            "KVAF_A5_AICPU_OWNER_PROBE_OK",
            "KVAF_A5_AICPU_OWNER_PROBE_FAILED",
            "--logical-device-id",
            "--timeout-seconds",
            "--provider-soname",
            "IsProviderBasename",
            "options.provider_soname.c_str()",
            "launch_args.timeout",
            "kRuntimeTimeoutGraceSeconds",
        ):
            self.assertIn(required, self.probe)

        self.assertIn("name.size() < 32U", self.probe)
        self.assertIn("name.find('/')", self.probe)
        self.assertIn("name.find('\\\\')", self.probe)

        aiv_sync = self.probe.index(
            "aclrtSynchronizeStream(resources.aiv_stream)"
        )
        owner_sync = self.probe.index(
            "aclrtSynchronizeStream(resources.owner_stream)", aiv_sync
        )
        self.assertLess(aiv_sync, owner_sync)

        owner_armed = self.probe.index("resources.owner_submitted = true;")
        owner_launch = self.probe.index("resources.aicpu_launch(")
        aiv_armed = self.probe.index("resources.aiv_submitted = true;")
        aiv_launch = self.probe.index(
            "LaunchAicpuOwnerConcurrencyProbe", aiv_armed
        )
        self.assertLess(owner_armed, owner_launch)
        self.assertLess(aiv_armed, aiv_launch)

    def test_probe_pass_cannot_override_the_direct_urma_gate(self) -> None:
        self.assertIn(
            "readonly KVAF_AICPU_AFFINITY_CONTRACT="
            "candidate-epoch-owner-unvalidated-v1",
            self.guard,
        )
        self.assertIn("probe PASS does NOT prove", self.guard)
        self.assertIn("does NOT unlock direct URMA", self.guard)

        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [
                bash,
                "-c",
                f"source {AFFINITY_GUARD!s}; "
                "kvaf_require_aicpu_affinity_contract",
            ],
            env={
                "PATH": str(Path(bash).parent),
                "KVAF_AICPU_OWNER_PROBE_RESULT": "PASS",
                "KVAF_AICPU_AFFINITY_CONTRACT":
                    "single-invocation-owner-v1",
            },
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "CONTRACT=candidate-epoch-owner-unvalidated-v1",
            result.stderr,
        )
        self.assertIn("does NOT unlock direct URMA", result.stderr)

    def test_direct_runner_still_blocks_before_topology_or_workers(self) -> None:
        guard = self.runner.index("kvaf_require_aicpu_affinity_contract")
        for later_operation in (
            "resolve_a5_urma_eids.py",
            "resolve_a5_urma_endpoints.py",
            '"${kvaf_npu_smi_binary}" info',
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host',
        ):
            self.assertLess(guard, self.runner.index(later_operation, guard))

    def test_owner_probe_documentation_is_explicitly_non_unlocking(self) -> None:
        documentation = OWNER_DOC.read_text(encoding="utf-8")
        for required in (
            "Probe PASS 不会解锁 direct URMA",
            "MVP session owner",
            "production epoch owner",
            "aicpu_kvaf_urma_owner_probe_v1",
            "kvaf_a5_aicpu_owner_probe",
            "run_a5_aicpu_owner_probe.sh",
            "AICPU_REGISTRATION=verified-exact",
            "KVAF_A5_AICPU_OWNER_PROBE_OK",
        ):
            self.assertIn(required, documentation)
        self.assertIn("KVAF_RUNTIME_UDMA_UAPI_HEADER", documentation)
        self.assertIn("KVAF_RUNTIME_UMMU_UAPI_HEADER", documentation)
        self.assertIn("provider build failed; package", documentation)

    def test_owner_probe_runner_verifies_registration_before_acl_exec(self) -> None:
        runner = OWNER_RUNNER.read_text(encoding="utf-8")
        documentation = OWNER_DOC.read_text(encoding="utf-8")
        lock = runner.index('exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"')
        provider = runner.index("kvaf_verify_direct_provider_build_v3")
        runtime = runner.index("kvaf_verify_direct_runtime_artifacts_v3")
        owner_hash = runner.index("actual_owner_probe_sha=")
        launch = runner.index(
            'exec "${KVAF_A5_OWNER_PROBE_BINARY}" "$@"'
        )
        self.assertLess(lock, provider)
        self.assertLess(provider, runtime)
        self.assertLess(runtime, owner_hash)
        self.assertLess(owner_hash, launch)
        for required in (
            "cann-kvaf-direct-urma-compat.tar.gz",
            "ascend_package_load.ini",
            "KVAF_A5_AICPU_OWNER_PROBE_PREFLIGHT_OK",
            "AICPU_REGISTRATION=verified-exact",
            "owner_probe_sha256",
            "libascendcl.so",
            "device_provider_sha=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}",
            "build_manifest_sha=${KVAF_VERIFIED_BUILD_MANIFEST_SHA}",
        ):
            self.assertIn(required, runner)
        self.assertNotIn("a5_aicpu_affinity_guard.sh", runner)

        self.assertIn(
            "bash scripts/run_a5_aicpu_owner_probe.sh", documentation
        )
        self.assertNotIn('"$kvaf_owner_probe_binary"', documentation)

    def test_provider_build_finds_the_shared_running_kernel_uapi(self) -> None:
        build_script = PROVIDER_BUILD.read_text(encoding="utf-8")
        self.assertIn(
            '"${ROOT_DIR}/../kernel-uapi/${kernel_release}/udma_abi.h"',
            build_script,
        )
        self.assertIn(
            '"${ROOT_DIR}/../kernel-uapi/${kernel_release}/ummu_core.h"',
            build_script,
        )

    def test_documented_commands_never_start_with_test(self) -> None:
        command_pattern = re.compile(r"(?m)^[ \t]*test(?:[ \t]|$)")
        self.assertIsNone(
            command_pattern.search(OWNER_DOC.read_text(encoding="utf-8")),
            "use an explicit if [[ ... ]] block in owner-probe commands",
        )


if __name__ == "__main__":
    unittest.main()

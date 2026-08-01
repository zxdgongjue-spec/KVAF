#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"
MANIFEST_V3 = ROOT / "scripts" / "a5_direct_urma_manifest_v3.sh"
RUNTIME_PREFLIGHT = (
    ROOT / "scripts" / "a5_direct_urma_runtime_preflight.sh"
)


class DirectUrmaRunnerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_mutating_and_run_scripts_are_valid_bash(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        scripts = (
            RUNNER,
            ROOT / "scripts" / "a5_direct_urma_legacy_guard.sh",
            ROOT / "scripts" / "a5_aicpu_affinity_guard.sh",
            MANIFEST_V3,
            ROOT / "scripts" / "a5_aicpu_package_registration.sh",
            RUNTIME_PREFLIGHT,
            ROOT / "scripts" / "run_a5_aiv_urma_h2d_smoke.sh",
            ROOT / "scripts" / "run_a5_aicpu_owner_probe.sh",
            ROOT / "scripts" /
            "run_a5_direct_urma_epoch_owner_validation.sh",
            ROOT / "scripts" / "build_a5_direct_urma.sh",
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh",
            ROOT / "scripts" / "package_a5_direct_urma_aicpu.sh",
        )
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [bash, "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_persistent_guard_precedes_eid_access(self) -> None:
        guard = self.source.index("acquire_persistent_session_guards\n")
        eid = self.source.index("resolve_a5_urma_eids.py")
        self.assertLess(guard, eid)
        self.assertNotIn("query_a5_topology.py", self.source)

    def test_unproven_aicpu_affinity_fails_before_hardware_access(self) -> None:
        guard = self.source.index("kvaf_require_aicpu_affinity_contract")
        eid = self.source.index("resolve_a5_urma_eids.py", guard)
        endpoint = self.source.index("resolve_a5_urma_endpoints.py", guard)
        npu_smi = self.source.index(
            '"${kvaf_npu_smi_binary}" info', guard
        )
        worker = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host', guard
        )
        self.assertLess(guard, eid)
        self.assertLess(guard, endpoint)
        self.assertLess(guard, npu_smi)
        self.assertLess(guard, worker)

        recovery = self.source.index("acquire_persistent_session_guards\n")
        self.assertLess(recovery, guard)

    def test_aicpu_affinity_gate_has_no_environment_override(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        helper = ROOT / "scripts" / "a5_aicpu_affinity_guard.sh"
        command = (
            f"source {helper!s}; "
            "kvaf_require_aicpu_affinity_contract"
        )
        result = subprocess.run(
            [bash, "-c", command],
            env={
                "PATH": str(Path(bash).parent),
                "KVAF_AICPU_AFFINITY_CONTRACT":
                    "documented-same-stream-process-affinity-v1",
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
        self.assertIn(
            "complete\nOpen-transfer-SHUTDOWN_QUIESCED-Close lifecycle",
            result.stderr,
        )
        self.assertIn(
            "No topology/EID/NPU/URMA operation was attempted.",
            result.stderr,
        )

    def test_candidate_epoch_owner_has_dedicated_staged_entry(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        helper = ROOT / "scripts" / "a5_aicpu_affinity_guard.sh"
        command = (
            f"source {helper!s}; "
            "kvaf_require_aicpu_affinity_contract "
            "candidate-epoch-owner-validation-v1"
        )
        result = subprocess.run(
            [bash, "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("staged resource-bearing", result.stderr)
        wrapper = (
            ROOT / "scripts" /
            "run_a5_direct_urma_epoch_owner_validation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("run_a5_direct_urma_h2d_smoke.sh", wrapper)
        self.assertIn("candidate-epoch-owner-validation-v1", wrapper)

    def test_zero_wqe_is_validation_only_and_pool_limit_is_not_cumulative(self) -> None:
        self.assertIn(
            '[[ "${kvaf_affinity_run_mode}" != \\\n'
            '      candidate-epoch-owner-validation-v1 ]]',
            self.source,
        )
        self.assertIn("KVAF_H2D_TRANSFER_COUNT != 0", self.source)
        self.assertIn("KVAF_H2D_PAYLOAD_CAPACITY_BYTES", self.source)
        self.assertIn("h2d_payload_capacity_bytes=%s", self.source)
        self.assertIn("KVAF_H2D_PROVIDER_SEGMENT_PAYLOAD_BYTES", self.source)
        self.assertIn("KVAF_H2D_PROVIDER_LOCAL_ALLOCATION_BYTES", self.source)
        self.assertIn(
            "h2d_provider_local_allocation_bytes_per_endpoint=%s",
            self.source,
        )
        self.assertNotIn("h2d_registered_bytes=%s", self.source)
        pool_limit = self.source.index(
            "the descriptor pool exceeds the 64 MiB registration limit"
        )
        capacity = self.source.index(
            "KVAF_H2D_PAYLOAD_CAPACITY_BYTES", pool_limit
        )
        total = self.source.index("KVAF_H2D_TOTAL_BYTES", capacity)
        self.assertLess(pool_limit, capacity)
        self.assertLess(capacity, total)

    def test_run_timeout_is_independent_from_provider_control_timeout(self) -> None:
        self.assertIn(
            "kvaf_run_timeout_is_set=0",
            self.source,
        )
        self.assertIn(
            "require_uint_max KVAF_H2D_RUN_TIMEOUT_SECONDS",
            self.source,
        )
        self.assertIn(
            '"${KVAF_H2D_RUN_TIMEOUT_SECONDS}" 604800',
            self.source,
        )
        self.assertIn(
            "KVAF_H2D_RUN_TIMEOUT_SECONDS == 0",
            self.source,
        )
        self.assertIn(
            "KVAF_H2D_TIMEOUT_SECONDS * 3 + 10",
            self.source,
        )
        self.assertIn(
            '--timeout-seconds "${KVAF_H2D_TIMEOUT_SECONDS}"',
            self.source,
        )
        self.assertIn(
            'deadline=$(( $(date +%s) + '
            'KVAF_H2D_RUN_TIMEOUT_SECONDS ))',
            self.source,
        )
        self.assertNotIn("total_timeout_seconds=", self.source)
        self.assertIn("h2d_control_timeout_seconds=%s", self.source)
        self.assertIn("h2d_run_timeout_seconds=%s", self.source)

    def test_tcp_control_addresses_are_explicit_validated_and_audited(self) -> None:
        for expected in (
            "KVAF_DIRECT_CONTROL_SERVER_ADDRESS",
            "KVAF_DIRECT_CONTROL_BIND_ADDRESS",
            "normalize_numeric_ip",
            "control_server_address=%s",
            "control_bind_address=%s",
            '--control-server-address "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"',
            '--control-bind-address "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}"',
            "URMA EIDs are not TCP management addresses.",
        ):
            self.assertIn(expected, self.source)
        validation = self.source.index(
            "if ! KVAF_DIRECT_CONTROL_SERVER_ADDRESS=$("
        )
        affinity = self.source.index(
            "kvaf_require_aicpu_affinity_contract", validation
        )
        eid = self.source.index("resolve_a5_urma_eids.py", affinity)
        self.assertLess(validation, affinity)
        self.assertLess(validation, eid)

    def test_urma_endpoints_are_exactly_resolved_before_npu_resources(self) -> None:
        pair = self.source.index("resolve_a5_urma_eids.py")
        endpoints = self.source.index("endpoint_resolver_args=(", pair)
        resolver = self.source.index("resolve_a5_urma_endpoints.py", endpoints)
        reserved_sq = self.source.index(
            "/sys/module/udma/parameters/sq_reserved", endpoints
        )
        idle = self.source.index(
            "# Take the occupancy snapshot as late as possible", reserved_sq
        )
        self.assertLess(pair, endpoints)
        self.assertLess(endpoints, resolver)
        self.assertLess(endpoints, reserved_sq)
        self.assertLess(endpoints, idle)
        resolution = self.source[endpoints:reserved_sq]
        self.assertIn("KVAF_UBCORE_SYSFS_ROOT", resolution)
        self.assertIn("KVAF_HCCN_TIMEOUT_SECONDS", resolution)
        self.assertIn("disagrees with the exact EID match", resolution)
        self.assertIn(
            "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created.",
            resolution,
        )

    def test_npu_idle_proof_precedes_result_and_worker_resources(self) -> None:
        preflight = self.source.index(
            "# Take the occupancy snapshot as late as possible"
        )
        result_create = self.source.index(
            'mkdir "${KVAF_A5_RESULT_DIR}"', preflight
        )
        host_launch = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host', preflight
        )
        device_launch = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role device', host_launch
        )
        self.assertLess(preflight, result_create)
        self.assertLess(preflight, host_launch)
        self.assertLess(preflight, device_launch)
        fail_closed = self.source[preflight:result_create]
        self.assertIn("KVAF_DIRECT_ALLOW_BUSY_NPU", fail_closed)
        self.assertIn("timeout --signal=TERM --kill-after=2s", fail_closed)
        self.assertIn("KVAF_NPU_SMI_TIMEOUT_SECONDS", fail_closed)
        self.assertIn("exit 2", fail_closed)
        self.assertIn(
            "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created.",
            fail_closed,
        )

    def test_npu_idle_proof_is_auditable_in_result_bundle(self) -> None:
        for name in (
            "npu-smi-info.txt",
            "npu-idle-check.txt",
            "check_a5_npu_idle.py",
            "npu_idle_preflight_status",
            "npu_idle_preflight_override",
            "npu_smi_snapshot_sha256",
            "npu_idle_checker_sha256",
        ):
            self.assertIn(name, self.source)

    def test_physical_id_is_required_without_acl_topology_side_effects(self) -> None:
        physical_guard = self.source.index(
            'if [[ -z "${KVAF_PHYSICAL_DEVICE_ID}" ]]; then'
        )
        eid_discovery = self.source.index(
            'if [[ -z "${KVAF_HOST_EID}" ]]; then', physical_guard
        )
        branch = self.source[physical_guard:eid_discovery]
        self.assertIn("resource-free NPU idle preflight", branch)
        self.assertIn("exit 2", branch)
        self.assertNotIn("query_a5_topology.py", self.source)

    def test_idle_check_cannot_target_a_different_physical_card(self) -> None:
        mismatch = self.source.index(
            'if [[ "${KVAF_LOGICAL_DEVICE_ID}" !='
        )
        preflight = self.source.index(
            "# Take the occupancy snapshot as late as possible",
            mismatch,
        )
        self.assertLess(mismatch, preflight)
        self.assertIn(
            "KVAF_LOGICAL_DEVICE_ID == KVAF_PHYSICAL_DEVICE_ID",
            self.source[mismatch:preflight],
        )

    def test_launch_intent_is_durable_before_each_fork(self) -> None:
        host_intent = self.source.index("active_phase=host_launch_intent")
        host_write = self.source.index("write_active_session", host_intent)
        host_fork = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host', host_write
        )
        host_started = self.source.index("active_phase=host_started", host_fork)
        self.assertLess(host_intent, host_write)
        self.assertLess(host_write, host_fork)
        self.assertLess(host_fork, host_started)

        device_intent = self.source.index("active_phase=device_launch_intent")
        device_write = self.source.index("write_active_session", device_intent)
        device_fork = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role device', device_write
        )
        device_started = self.source.index(
            "active_phase=device_started", device_fork
        )
        self.assertLess(device_intent, device_write)
        self.assertLess(device_write, device_fork)
        self.assertLess(device_fork, device_started)

    def test_worker_inherits_legacy_and_active_locks(self) -> None:
        for role in ("host", "device"):
            launch = self.source.index(
                f'nohup "${{KVAF_A5_DIRECT_BINARY}}" --role {role}'
            )
            launch_line = self.source[launch : launch + 180]
            self.assertIn("9>&-", launch_line)
            self.assertNotIn("6>&-", launch_line)
            self.assertNotIn("7>&-", launch_line)
            self.assertNotIn("8>&-", launch_line)

    def test_interrupt_after_host_launch_reaches_device_pair_first(self) -> None:
        host_launch = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role host'
        )
        device_launch = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role device', host_launch
        )
        supervise_signal = self.source.index(
            "if (( runner_signal_requested != 0 )); then", device_launch
        )
        between_launches = self.source[host_launch:device_launch]
        self.assertNotIn("exit 130", between_launches)
        self.assertLess(device_launch, supervise_signal)

    def test_host_terminal_is_checked_before_device_fork(self) -> None:
        device_launch = self.source.index(
            'nohup "${KVAF_A5_DIRECT_BINARY}" --role device'
        )
        host_check = self.source.index(
            "pre_device_host_classifier_rc", 0, device_launch
        )
        self.assertLess(host_check, device_launch)
        self.assertIn(
            "host_unsafe_before_device_launch",
            self.source[host_check:device_launch],
        )

    def test_recovery_requires_pair_after_device_launch_intent(self) -> None:
        recovery = self.source[
            self.source.index("recover_stale_active_session()") :
            self.source.index("acquire_persistent_session_guards()")
        ]
        self.assertIn(
            "device_started|safe_terminal_process_stuck",
            recovery,
        )
        self.assertIn("pair_required=1", recovery)
        self.assertIn('"${host_pid}" =~ ^(0|[1-9][0-9]*)$', recovery)
        self.assertNotIn(
            '[[ "${device_pid}" == 0 ]]',
            recovery,
            "PID zero must not decide whether Device may have launched",
        )
        self.assertIn(
            '[[ "${host_pid}" != 0 && "${device_pid}" == 0 ]]',
            recovery,
        )
        self.assertIn(
            '[[ "${host_pid}" != 0 && "${device_pid}" != 0 ]]',
            recovery,
        )

    def test_safe_terminal_stuck_keeps_active_state(self) -> None:
        start = self.source.index(
            "# The transport is safe, but at least one worker has not exited."
        )
        end = self.source.index(
            "retain_unproven_active_session \\\n"
            "      bounded_wait_expired",
            start,
        )
        branch = self.source[start:end]
        self.assertIn("write_active_session", branch)
        self.assertNotIn("clear_active_session", branch)
        self.assertIn("ACTIVE_SESSION=", branch)

    def test_legacy_unverified_recovery_remains_fail_closed(self) -> None:
        self.assertIn(
            "(( pair_classifier_rc == 0 || pair_classifier_rc == 3 ))",
            self.source,
        )

    def test_new_runs_require_source_owned_checked_udma_and_ummu(self) -> None:
        self.assertNotIn(
            "KVAF_DIRECT_ALLOW_UNVERIFIED_UDMA_TEARDOWN", self.source
        )
        self.assertNotIn("unverified-system-provider)", self.source)
        verifier = MANIFEST_V3.read_text(encoding="utf-8")
        for binding in (
            "kvaf-direct-urma-provider-v3",
            "official-umdk-source-built-internal",
            "paired-libummu-source-built-internal",
            "umdk_libummu_pairing:verified",
            "checked-retryable-v1",
        ):
            self.assertIn(binding, verifier)
        self.assertIn('"${prefix}_header_sha256"', verifier)
        self.assertIn("lib(urma_common|urma|ummu|udma|urma-udma)", verifier)

    def test_provider_v3_pins_source_pair_kernel_and_runtime_uapi(self) -> None:
        builder = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        verifier = MANIFEST_V3.read_text(encoding="utf-8")

        for value in (
            "12917421beb93c1351eb2f124bb2f34612d8bf2d",
            "70de020f776d8def0d8a8941cc7cbc5928d66c10",
            "https://gitcode.com/openeuler/umdk.git",
            "https://gitcode.com/openeuler/libummu.git",
            "26.06.0",
            "1.0.5",
        ):
            self.assertIn(value, builder)
            self.assertIn(value, verifier)
        self.assertIn("build_kernel_release=${kernel_release}", builder)
        self.assertIn('"${build_kernel_release}" == "$(uname -r)"', verifier)
        self.assertIn('"runtime_${prefix}_header_path"', verifier)
        self.assertIn('"runtime_${prefix}_match" verified-exact', verifier)
        self.assertIn('"runtime_${prefix}_header_sha256"', verifier)

    def test_recovery_precedes_provider_v3_and_hardware_access(self) -> None:
        verify = self.source.index("kvaf_verify_direct_provider_build_v3")
        recovery = self.source.index("acquire_persistent_session_guards\n")
        eid = self.source.index("resolve_a5_urma_eids.py")
        endpoint = self.source.index("resolve_a5_urma_endpoints.py")
        npu = self.source.index('"${kvaf_npu_smi_binary}" info')
        self.assertLess(recovery, verify)
        self.assertLess(verify, eid)
        self.assertLess(verify, endpoint)
        self.assertLess(verify, npu)
        verifier = MANIFEST_V3.read_text(encoding="utf-8")
        for forbidden in ("/proc/", "npu-smi", "aclrt", "resolve_a5_urma"):
            self.assertNotIn(forbidden, verifier)

    def test_complete_runtime_artifact_preflight_precedes_hardware(self) -> None:
        provider = self.source.index("kvaf_verify_direct_provider_build_v3")
        affinity = self.source.index(
            "kvaf_require_aicpu_affinity_contract"
        )
        runtime = self.source.index(
            "kvaf_verify_direct_runtime_artifacts_v3"
        )
        eid = self.source.index("resolve_a5_urma_eids.py")
        endpoint = self.source.index("resolve_a5_urma_endpoints.py")
        reserved_sq = self.source.index(
            "/sys/module/udma/parameters/sq_reserved"
        )
        npu = self.source.index('"${kvaf_npu_smi_binary}" info')
        self.assertLess(provider, affinity)
        self.assertLess(affinity, runtime)
        self.assertLess(runtime, eid)
        self.assertLess(runtime, endpoint)
        self.assertLess(runtime, reserved_sq)
        self.assertLess(runtime, npu)

        helper = RUNTIME_PREFLIGHT.read_text(encoding="utf-8")
        for required in (
            "kvaf-a5-direct-smoke-build-v2",
            "kvaf-direct-urma-provider.manifest",
            "kvaf-direct-urma-umdk-source.sha256",
            "kvaf_udma_private_abi_probe",
            "libascendcl.so",
        ):
            self.assertIn(required, helper)
        for forbidden in (
            "/proc/",
            "hccn_tool",
            "npu-smi",
            "aclrt",
            "/sys/module/",
        ):
            self.assertNotIn(forbidden, helper)

    def test_package_verifies_provider_once_under_deploy_lock(self) -> None:
        package = (
            ROOT / "scripts" / "package_a5_direct_urma_aicpu.sh"
        ).read_text(encoding="utf-8")
        lock = package.index('exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"')
        acquired = package.index("flock -w 30 9", lock)
        verify = package.index("kvaf_verify_direct_provider_build_v3")
        self.assertLess(acquired, verify)
        self.assertEqual(
            package.count("kvaf_verify_direct_provider_build_v3"), 1
        )

    def test_package_registration_is_unique_atomic_and_fail_closed(self) -> None:
        package = (
            ROOT / "scripts" / "package_a5_direct_urma_aicpu.sh"
        ).read_text(encoding="utf-8")
        registration = (
            ROOT / "scripts" / "a5_aicpu_package_registration.sh"
        ).read_text(encoding="utf-8")
        preflight = RUNTIME_PREFLIGHT.read_text(encoding="utf-8")

        for source in (package, preflight):
            self.assertIn(
                "cann-kvaf-direct-urma-compat.tar.gz", source
            )
            self.assertIn(
                "opp/vendors/cust/op_impl/aicpu/kernel", source
            )
            self.assertIn("ascend_package_load.ini", source)
            self.assertIn("kvaf_aicpu_registration_is_exact", source)

        registry_check = package.index(
            "kvaf_aicpu_registration_is_absent"
        )
        archive_publish = package.index(
            'mv -f "${temporary_install_archive}"'
        )
        installed_verify = package.index(
            'verify_aicpu_archive "${KVAF_DIRECT_AICPU_ARCHIVE}" installed'
        )
        registration_publish = package.index(
            "kvaf_aicpu_install_registration"
        )
        self.assertLess(registry_check, archive_publish)
        self.assertLess(archive_publish, installed_verify)
        self.assertLess(installed_verify, registration_publish)
        self.assertIn("name:Ascend-aicpu_legacy.tar.gz", package)
        self.assertIn("legacy liburma_perftest registration", package)
        self.assertIn("rollback_archive", package)
        self.assertIn("deployment_started=1", package)
        self.assertIn("deployment_committed=1", package)
        self.assertIn("KVAF_AICPU_ARCHIVE_ROLLBACK_OK", package)
        rollback = package.index("KVAF_AICPU_ARCHIVE_ROLLBACK_OK")
        publish_started = package.index("deployment_started=1")
        publish = package.index(
            'mv -f "${temporary_install_archive}"'
        )
        commit = package.index("deployment_committed=1")
        self.assertLess(rollback, publish_started)
        self.assertLess(publish_started, publish)
        self.assertLess(publish, commit)

        self.assertIn("cp -p --", registration)
        self.assertIn("mv -f --", registration)
        self.assertNotIn('mkdir -p -- "${registry_dir}"', registration)
        self.assertIn('-w "${registry_file}"', registration)

        registration_verify = preflight.index(
            "kvaf_aicpu_registration_is_exact"
        )
        archive_read = preflight.index(
            'archive_entries=$(tar -tzf "${KVAF_DIRECT_AICPU_ARCHIVE}")'
        )
        self.assertLess(registration_verify, archive_read)
        self.assertIn("sibling AICPU package is invalid", package)
        self.assertIn("sibling AICPU package is invalid", preflight)

    def test_provider_v2_manifest_fails_closed(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            host = build / "libkvaf_urma_provider_host.so"
            device = build / "libkvaf_urma_provider_device.so"
            source_manifest = build / "source.sha256"
            manifest = build / "build.manifest"
            success = build / "build.success"
            host.write_bytes(b"host")
            device.write_bytes(b"device")
            source_manifest.write_text("legacy\n", encoding="utf-8")
            manifest.write_text(
                "schema=kvaf-direct-urma-provider-v2\n",
                encoding="utf-8",
            )
            success.write_text(
                "schema=kvaf-direct-urma-provider-success-v2\n",
                encoding="utf-8",
            )
            command = (
                f'source "{MANIFEST_V3}"; '
                f'KVAF_DIRECT_PROVIDER_BUILD_DIR="{build}"; '
                f'KVAF_DIRECT_HOST_PROVIDER_LIBRARY="{host}"; '
                f'KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY="{device}"; '
                f'KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST="{source_manifest}"; '
                f'KVAF_DIRECT_PROVIDER_BUILD_MANIFEST="{manifest}"; '
                f'KVAF_DIRECT_PROVIDER_SUCCESS_STAMP="{success}"; '
                "kvaf_verify_direct_provider_build_v3"
            )
            result = subprocess.run(
                [bash, "-c", command],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected schema", result.stderr)

    def test_v3_probe_verifier_uses_builder_abi_field_names(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        builder = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        cases = (
            (
                "udma_uapi",
                "kvaf_udma_uapi_probe",
                "KVAF_UDMA_UAPI_ABI=",
                "udma_uapi_abi",
                "udma-public-layout",
            ),
            (
                "udma_private_abi",
                "kvaf_udma_private_abi_probe",
                "KVAF_UDMA_PRIVATE_ABI=",
                "udma_private_abi",
                "udma-private-layout",
            ),
            (
                "ummu_uapi",
                "kvaf_ummu_uapi_probe",
                "KVAF_UMMU_UAPI_ABI=",
                "ummu_uapi_abi",
                "ummu-public-layout",
            ),
        )
        for _, _, _, abi_key, _ in cases:
            self.assertIn(f"{abi_key}=${{", builder)
            self.assertIn(f"{abi_key}_sha256=${{", builder)
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            manifest_lines = []
            success_lines = []
            calls = []
            for prefix, filename, output_prefix, abi_key, abi_value in cases:
                output = f"{output_prefix}{abi_value}"
                probe = build / filename
                probe.write_text(
                    "#!/bin/sh\n" f"printf '%s\\n' '{output}'\n",
                    encoding="utf-8",
                )
                probe.chmod(0o755)
                probe_sha = hashlib.sha256(probe.read_bytes()).hexdigest()
                output_sha = hashlib.sha256(
                    f"{output}\n".encode("utf-8")
                ).hexdigest()
                manifest_lines.extend(
                    (
                        f"{prefix}_probe_filename={filename}",
                        f"{prefix}_probe_sha256={probe_sha}",
                        f"{abi_key}={abi_value}",
                        f"{abi_key}_sha256={output_sha}",
                    )
                )
                success_lines.append(f"{prefix}_probe_sha256={probe_sha}")
                calls.append(
                    "kvaf_v3_verify_probe "
                    f"{prefix} {filename} {output_prefix} {abi_key}"
                )

            manifest = build / "build.manifest"
            success = build / "build.success"
            manifest.write_text(
                "\n".join(manifest_lines) + "\n", encoding="utf-8"
            )
            success.write_text(
                "\n".join(success_lines) + "\n", encoding="utf-8"
            )
            command = (
                f'source "{MANIFEST_V3}"; '
                f'KVAF_DIRECT_PROVIDER_BUILD_DIR="{build}"; '
                f'KVAF_DIRECT_PROVIDER_BUILD_MANIFEST="{manifest}"; '
                f'KVAF_DIRECT_PROVIDER_SUCCESS_STAMP="{success}"; '
                + " && ".join(calls)
            )
            result = subprocess.run(
                [bash, "-c", command],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_aicpu_packaging_requires_source_built_provider_manifest(self) -> None:
        package_source = (
            ROOT / "scripts" / "package_a5_direct_urma_aicpu.sh"
        ).read_text(encoding="utf-8")
        verifier = MANIFEST_V3.read_text(encoding="utf-8")
        self.assertIn("kvaf_verify_direct_provider_build_v3", package_source)
        self.assertNotIn("for prefix in urma_library urma_common_library", verifier)
        for binding in (
            "urma_core_mode:official-umdk-source-linked-no-dynamic-providers",
            "udma_provider_mode:official-umdk-source-built-internal",
            "ummu_provider_mode:paired-libummu-source-built-internal",
        ):
            self.assertIn(binding, verifier)
        self.assertIn("lib(urma_common|urma|ummu|udma|urma-udma)", verifier)
        for entry in (
            "kvaf-direct-urma-umdk-source.sha256",
            "kvaf-direct-urma-libummu-source.sha256",
            "kvaf_udma_uapi_probe",
            "kvaf_udma_private_abi_probe",
            "kvaf_ummu_uapi_probe",
        ):
            self.assertIn(entry, package_source)

    def test_bounded_wait_without_terminal_proof_stays_recoverable(self) -> None:
        start = self.source.index("bounded_wait_expired")
        end = self.source.index("sleep 1 || true", start)
        branch = self.source[start:end]
        self.assertIn("pair_terminal_unproven_process_stuck", branch)
        self.assertIn("exit 5", branch)
        self.assertNotIn("record_quarantine", branch)
        self.assertIn(
            'record_quarantine "unverified_udma_armed_terminal_${pair_outcome}"',
            self.source,
        )

    def test_explicit_unsafe_final_does_not_require_fatal_edge(self) -> None:
        self.assertIn("terminal_explicit_unsafe_proven()", self.source)
        self.assertIn(
            "any_valid_fatal_marker || terminal_explicit_unsafe_proven",
            self.source,
        )
        final_branch = self.source[
            self.source.index("if (( pair_classifier_rc == 4 )); then") :
            self.source.index("if (( pair_classifier_rc == 3 )); then")
        ]
        self.assertIn("record_quarantine", final_branch)
        self.assertNotIn("fatal.host", final_branch)

    def test_fatal_wakeup_is_nonce_validated(self) -> None:
        self.assertIn("valid_fatal_marker()", self.source)
        self.assertIn(
            '"${run_id}" == "${kvaf_run_id}"', self.source
        )

    def test_active_barriers_do_not_share_unsafe_exit_code(self) -> None:
        active_lock = self.source.index(
            "a previous direct URMA worker still owns the active-session lock"
        )
        active_tail = self.source[active_lock : active_lock + 700]
        self.assertIn("exit 5", active_tail)

        quarantine = self.source.index(
            "an unsafe direct URMA session is quarantined"
        )
        quarantine_tail = self.source[quarantine : quarantine + 500]
        self.assertIn("exit 4", quarantine_tail)

    def test_result_bundle_contains_direct_and_provider_provenance(self) -> None:
        for name in (
            "direct-smoke-build.manifest",
            "direct-smoke-source.sha256",
            "provider-build.manifest",
            "provider-source.sha256",
        ):
            self.assertIn(name, self.source)
        self.assertIn("outcome_classifier_sha256", self.source)
        self.assertIn(
            'recorded_classifier="${resolved_result}/'
            'classify_a5_direct_urma_outcome.py"',
            self.source,
        )

    def test_recovery_manifest_keys_are_written_once(self) -> None:
        start = self.source.index("kvaf_commit=$(\n")
        end = self.source.index('} >"${manifest}"', start)
        manifest_writer = self.source[start:end]
        for key in (
            "run_id",
            "teardown_assurance",
            "outcome_classifier_sha256",
        ):
            self.assertEqual(
                manifest_writer.count(f"printf '{key}=%s\\n'"),
                1,
                f"duplicate {key} makes manifest_value fail closed",
            )

    def test_custom_result_path_cannot_escape_recovery_root(self) -> None:
        self.assertIn(
            'KVAF_A5_RESULT_DIR must be one safe direct child of '
            '${kvaf_results_root}',
            self.source,
        )
        self.assertIn("kvaf_configure_direct_urma_state_paths", self.source)
        self.assertNotIn('"${ROOT_DIR}/results/a5_direct_urma_h2d_', self.source)

    def test_quarantine_is_durable_before_active_is_cleared(self) -> None:
        start = self.source.index("record_quarantine()")
        end = self.source.index("retain_unproven_active_session()", start)
        body = self.source[start:end]
        publish = body.index("durably_publish_state")
        clear = body.index("clear_active_session")
        self.assertLess(publish, clear)

    def test_recovery_bundle_is_synced_before_launch_intent(self) -> None:
        bundle_sync = self.source.index(
            'python3 "${KVAF_DURABLE_STATE_HELPER}" sync'
        )
        launch_intent = self.source.index("active_phase=host_launch_intent")
        self.assertLess(bundle_sync, launch_intent)

    def test_fixed_state_root_rejects_split_brain_override(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        helper = ROOT / "scripts" / "a5_direct_urma_legacy_guard.sh"
        command = (
            f"source {helper!s}; "
            "KVAF_DIRECT_QUARANTINE_FILE=/tmp/stale-quarantine; "
            f"kvaf_configure_direct_urma_state_paths {ROOT!s}"
        )
        result = subprocess.run(
            [bash, "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot override", result.stderr)

    def test_machine_state_root_is_checkout_independent(self) -> None:
        self.assertIn(
            "expected_global_state=/data/.kvaf-state",
            (
                ROOT / "scripts" / "a5_direct_urma_legacy_guard.sh"
            ).read_text(encoding="utf-8"),
        )

    def test_persisted_pid_evidence_includes_process_identity(self) -> None:
        for field in (
            "host_start_ticks",
            "device_start_ticks",
            "host_observed_exe",
            "device_observed_exe",
            "worker_expected_exe",
        ):
            self.assertIn(field, self.source)
        self.assertIn(
            'kvaf_result_candidate=$(realpath -m "${KVAF_A5_RESULT_DIR}")',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()

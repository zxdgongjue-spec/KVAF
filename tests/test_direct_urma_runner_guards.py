#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"


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
            ROOT / "scripts" / "run_a5_aiv_urma_h2d_smoke.sh",
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
        guard = self.source.index(
            "kvaf_require_aicpu_affinity_contract || exit $?"
        )
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
            "CONTRACT=unproven-kfc-multilaunch", result.stderr
        )
        self.assertIn(
            "pid/cookie/SO-base/generation sample is diagnostic only",
            result.stderr,
        )
        self.assertIn(
            "No topology/EID/NPU/URMA operation was attempted.",
            result.stderr,
        )

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
        for binding in (
            "udma_provider_mode=source-built-internal",
            "udma_runtime_capability=0x0000000000020001",
            "ummu_provider_mode=source-built-internal",
            "ummu_runtime_capability=0x0000000000010001",
            "checked-retryable-v1",
        ):
            self.assertIn(binding, self.source)
        self.assertIn("ummu_uapi_header_sha256", self.source)
        self.assertIn("lib(urma|ummu|urma-udma)", self.source)

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

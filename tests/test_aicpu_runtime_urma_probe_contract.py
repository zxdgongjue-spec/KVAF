from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ABI = ROOT / "access_operator/a5/provider/include/kvaf_urma_provider_abi.h"
PROVIDER = ROOT / "access_operator/a5/provider/runtime_matched_urma_probe.c"
SMOKE = ROOT / "access_operator/a5/smoke/aicpu_runtime_urma_probe.cpp"
CMAKE = ROOT / "access_operator/a5/CMakeLists.txt"
PROVIDER_CMAKE = ROOT / "access_operator/a5/provider/CMakeLists.txt"
EXPORTS = ROOT / "access_operator/a5/provider/provider_exports.map"
RUNTIME_EXPORTS = ROOT / "access_operator/a5/provider/runtime_probe_exports.map"
BUILD = ROOT / "scripts/build_a5_direct_urma.sh"
BUILD_PROVIDER = ROOT / "scripts/build_a5_direct_urma_provider.sh"
PACKAGE = ROOT / "scripts/package_a5_direct_urma_aicpu.sh"
PREFLIGHT = ROOT / "scripts/a5_direct_urma_runtime_preflight.sh"
MANIFEST_HELPER = ROOT / "scripts/a5_direct_urma_manifest_v3.sh"
RUNNER = ROOT / "scripts/run_a5_aicpu_runtime_urma_probe.sh"
SUPERVISOR = ROOT / "scripts/a5_runtime_probe_supervisor.sh"
STATE_HELPER = ROOT / "scripts/a5_runtime_probe_state.sh"
FAKE_ACL = ROOT / "tests/fake_acl"


def source(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


class AicpuRuntimeUrmaProbeContractTests(unittest.TestCase):
    def test_host_probe_compiles_as_plain_host_cpp(self) -> None:
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(FAKE_ACL),
                "-I",
                str(ABI.parent),
                "-fsyntax-only",
                str(SMOKE),
            ],
            check=True,
        )

    def test_fixed_abi_layout_compiles_in_c_and_cpp(self) -> None:
        unit = textwrap.dedent(
            """
            #include <stddef.h>
            #include "kvaf_urma_provider_abi.h"
            int main(void) {
              struct kvaf_urma_runtime_probe_result_v2 result = {0};
              struct kvaf_urma_aicpu_runtime_probe_args_v2 args = {0};
              return (sizeof(result) == 384 && sizeof(args) == 192 &&
                      KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER == 0 &&
                      KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE == 1 &&
                      KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE == 2 &&
                      offsetof(struct kvaf_urma_aicpu_runtime_probe_args_v2,
                               result_address) == 112 &&
                      offsetof(struct kvaf_urma_aicpu_runtime_probe_args_v2,
                               request_nonce) == 128 &&
                      offsetof(struct kvaf_urma_runtime_probe_result_v2,
                               request_nonce) == 272 &&
                      offsetof(struct kvaf_urma_runtime_probe_result_v2,
                               provider_build_id_sha256) == 312 &&
                      offsetof(struct kvaf_urma_aicpu_runtime_probe_args_v2,
                               expected_provider_build_id_sha256) == 160) ? 0 : 1;
            }
            """
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "probe.c"
            path.write_text(unit, encoding="utf-8")
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Werror",
                    "-I",
                    str(ABI.parent),
                    "-fsyntax-only",
                    str(path),
                ],
                check=True,
            )

    def test_discover_skips_explicit_init_but_dlopen_is_not_assumed_safe(self) -> None:
        provider = source(PROVIDER)
        self.assertIn(
            'dlopen(KVAF_RUNTIME_URMA_SONAME, RTLD_NOW | RTLD_LOCAL)',
            provider,
        )
        self.assertNotIn("RTLD_DEFAULT", provider)
        self.assertNotIn("RTLD_NEXT", provider)
        self.assertNotIn("RTLD_DEEPBIND", provider)
        discover = provider.index(
            "args->mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER"
        )
        init = provider.index("status = api.init(NULL)")
        self.assertLess(discover, init)
        # The branch skips explicit urma_init/create calls, but dlopen above it
        # still executes the vendor DSO constructor.  Runner safety is tested
        # separately and must not classify discover as resource-free.
        self.assertIn("kvaf_runtime_probe_record_library_path", provider)
        self.assertIn("KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS", provider)
        self.assertIn("KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS", provider)

    def test_resource_mode_is_single_invocation_and_fail_closed(self) -> None:
        provider = source(PROVIDER)
        order = [
            "status = api.init(NULL)",
            "device = api.get_device_by_name(device_name)",
            "context = api.create_context(device, args->eid_index)",
            "jfc = api.create_jfc(context, &jfc_config)",
            "jfr = api.create_jfr(context, &jfr_config)",
            "status = api.delete_jfr(jfr)",
            "status = api.delete_jfc(jfc)",
            "status = api.delete_context(context)",
            "status = api.uninit()",
            "status = dlclose(library)",
        ]
        positions = [provider.index(marker) for marker in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("status == URMA_EEXIST", provider)
        self.assertIn(
            "KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED",
            provider,
        )
        self.assertIn(
            "KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN",
            provider,
        )
        self.assertIn("g_kvaf_runtime_probe_poisoned", provider)
        self.assertIn(
            "KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED", provider
        )
        self.assertIn("result->cleanup_required != result->cleanup_completed", provider)
        side_effect_flag = (
            "KVAF_URMA_RUNTIME_PROBE_"
            "OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN"
        )
        self.assertIn(side_effect_flag, provider)
        self.assertLess(
            provider.index("result->flags |=", provider.index(side_effect_flag) - 50),
            provider.index(
                "dlopen(KVAF_RUNTIME_URMA_SONAME, RTLD_NOW | RTLD_LOCAL)"
            ),
        )
        self.assertNotIn("runtime_cleanup_completed", provider)
        self.assertNotIn(
            "~KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN",
            provider.replace("\n", ""),
        )
        self.assertIn("pthread_mutex_trylock", provider)
        self.assertLess(
            provider.index("result->required_symbol_mask = required_symbol_mask"),
            provider.index(
                "dlopen(KVAF_RUNTIME_URMA_SONAME, RTLD_NOW | RTLD_LOCAL)"
            ),
        )

    def test_host_reports_cleanup_and_runner_never_timeout_kills_probe(self) -> None:
        smoke = source(SMOKE)
        runner = source(RUNNER)
        self.assertIn(
            "options.mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER",
            smoke,
        )
        self.assertIn("launch_args.timeout = 0U", smoke)
        self.assertIn(
            "official_loader_constructor_side_effects=unproven", smoke
        )
        self.assertNotIn(
            "timeout --signal=TERM --kill-after=5s 30s",
            runner,
        )
        self.assertNotIn("resource-free discovery log", runner)
        self.assertIn("cleanup_required == result.cleanup_completed", smoke)
        self.assertIn("!official_side_effects_cleanup_unproven", smoke)
        self.assertIn("official_side_effects_cleanup=", smoke)
        self.assertIn("official_public_cleanup=", smoke)
        self.assertIn("!resources.cleanup_proven && !borrowed_runtime", smoke)
        self.assertIn("HoldUnsafeOwner", smoke)
        self.assertIn("CloseChecked", smoke)
        self.assertIn("KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK", smoke)
        self.assertIn(
            "KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE", smoke
        )
        self.assertNotIn(
            "KVAF_A5_RUNTIME_URMA_BASE_SYMBOL_SURFACE_PRESENT", smoke
        )
        self.assertIn("official_abi_handshake=unavailable", smoke)
        self.assertIn(
            "documented_official_aicpu_call_contract=", smoke
        )
        self.assertIn("not-found-in-audited-sources", smoke)
        self.assertIn("local_runtime_compatibility=unproven", smoke)
        self.assertIn("private_queue_abi=unproven", smoke)
        self.assertIn("CapabilityClass", smoke)
        self.assertIn("operation_errno_valid", smoke)
        self.assertIn("required_symbol_mask", smoke)
        self.assertIn("resolved_symbol_mask", smoke)
        self.assertIn("compiled_urma_api_version", smoke)
        self.assertIn("runtime_urma_api_version=unavailable", smoke)
        self.assertIn(
            "candidate_modern_queue_wrapper_symbol_surface=", smoke
        )
        self.assertIn("segment_symbol_surface=", smoke)
        self.assertIn(
            "legacy_jetty_link_wrapper_symbol_surface=", smoke
        )
        self.assertIn("current_direct_ub_symbol_surface=", smoke)
        self.assertIn(
            "audited_pinned_source_snapshot_known_gaps=present", smoke
        )
        self.assertIn("runtime_liburma_source_identity=unproven", smoke)
        self.assertIn("ValidateResultAgainstRequest", smoke)
        self.assertIn("request_binding=verified-v2", smoke)
        self.assertIn("aiv_address_visibility=unproven", smoke)
        self.assertIn("wqe_cqe_private_layout=unproven", smoke)
        self.assertIn(
            '"KVAF_A5_RUNTIME_URMA_PRELAUNCH_SAFE mode="', smoke
        )
        self.assertIn("phase=result_alignment", smoke)
        self.assertIn("offsetof(rtAicpuArgsEx_t, timeout) == 42", smoke)
        self.assertIn("KVAF_A5_RUNTIME_URMA_PROBE_NON_ACCEPTED", smoke)
        self.assertNotIn("KVAF_A5_RUNTIME_URMA_PROBE_INCOMPATIBLE", smoke)
        teardown = smoke.index("KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK")
        terminal = smoke.index(
            "KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE"
        )
        self.assertLess(teardown, terminal)

    def test_complete_symbol_group_does_not_claim_complete_semantics(self) -> None:
        smoke = source(SMOKE)
        start = smoke.index("const char* SymbolGroupState(")
        end = smoke.index("const char* CapabilityClass(", start)
        function = smoke[start:end]

        self.assertIn(
            '? "complete-symbols-only-semantics-unproven"', function
        )
        self.assertIn(': "incomplete"', function)
        self.assertNotIn('? "complete"', function)

    def test_dynamic_probe_is_unconditionally_disabled_after_recovery(self) -> None:
        runner = source(RUNNER)
        recovery_markers = (
            "kvaf_try_recover_rebooted_runtime_probe_blocked || true",
            "kvaf_try_recover_rebooted_runtime_probe_hold || true",
            "kvaf_try_recover_completed_runtime_probe || true",
            "KVAF_A5_RUNTIME_URMA_REBOOT_RECOVERY_ONLY_OK",
            "KVAF_A5_RUNTIME_URMA_HELD_REBOOT_RECOVERY_ONLY_OK",
            "KVAF_A5_RUNTIME_URMA_BLOCKED_RECOVERY_ONLY",
            "KVAF_A5_RUNTIME_URMA_RECOVERY_ONLY_OK",
        )
        gate = runner.index(
            'echo "KVAF_A5_RUNTIME_URMA_DYNAMIC_PROBE_DISABLED"'
        )
        gate_exit = runner.index("builtin exit 2", gate)
        false_branch = runner.index("if [[ 0 == 1 ]]; then", gate_exit)
        ascend = runner.index(
            ': "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"',
            gate,
        )
        provider = runner.index("kvaf_verify_direct_provider_build_v3", gate)
        npu = runner.index("30s npu-smi info", gate)
        endpoint = runner.index('"${KVAF_ENDPOINT_RESOLVER}"', gate)

        for marker in recovery_markers:
            self.assertLess(runner.index(marker), gate, marker)
        for operation in (ascend, provider, npu, endpoint):
            self.assertLess(gate_exit, operation)

        gate_block = runner[gate:ascend]
        self.assertIn(
            "CAPABILITY_OBSERVATION_LEDGER=not-implemented", gate_block
        )
        self.assertIn(
            "DOCUMENTED_OFFICIAL_AICPU_CALL_CONTRACT="
            "not-found-in-audited-sources",
            gate_block,
        )
        self.assertIn("LOCAL_RUNTIME_COMPATIBILITY=unproven", gate_block)
        self.assertIn("OFFICIAL_FUNCTION_SEMANTICS=unproven", gate_block)
        self.assertIn("No ACL/AICPU/URMA operation was attempted.", gate_block)
        self.assertNotIn("${KVAF_", gate_block)
        self.assertNotIn("getenv", gate_block)
        self.assertEqual(gate_block.count("if [[ 0 == 1 ]]; then"), 1)
        self.assertLess(gate_exit, false_branch)

    def test_build_package_and_runtime_preflight_require_the_probe(self) -> None:
        symbol = "aicpu_kvaf_urma_rt_probe_v2"
        self.assertIn(symbol, source(BUILD_PROVIDER))
        for path in (RUNTIME_EXPORTS, PACKAGE, PREFLIGHT):
            self.assertIn(symbol, source(path), str(path))
        self.assertNotIn(symbol, source(EXPORTS))
        self.assertIn("runtime_matched_urma_probe.c", source(PROVIDER_CMAKE))
        cmake = source(CMAKE)
        self.assertIn("kvaf_a5_aicpu_runtime_urma_probe", cmake)
        runtime_target = cmake.index(
            "add_executable(kvaf_a5_aicpu_runtime_urma_probe"
        )
        runtime_tail = cmake[runtime_target:]
        self.assertIn(
            "kvaf_configure_a5_host_target(kvaf_a5_aicpu_runtime_urma_probe)",
            runtime_tail,
        )
        self.assertNotIn(
            "kvaf_configure_a5_target(kvaf_a5_aicpu_runtime_urma_probe)",
            runtime_tail,
        )
        build = source(BUILD)
        self.assertIn("runtime_urma_probe_path=", build)
        self.assertIn("runtime_urma_probe_sha256=", build)

    def test_runtime_probe_provider_is_a_minimal_unique_dso(self) -> None:
        cmake = source(PROVIDER_CMAKE)
        exports = source(EXPORTS)
        runtime_exports = source(RUNTIME_EXPORTS)
        common_start = cmake.index("add_library(kvaf_urma_provider_common OBJECT")
        common_end = cmake.index("set_target_properties(", common_start)
        self.assertNotIn(
            "runtime_matched_urma_probe.c", cmake[common_start:common_end]
        )
        target_start = cmake.index(
            "add_library(kvaf_urma_runtime_probe_provider_device SHARED"
        )
        target_end = cmake.index(
            "add_library(kvaf_public_abi_consumer_probe", target_start
        )
        target = cmake[target_start:target_end]
        self.assertIn("runtime_matched_urma_probe.c", target)
        for forbidden in (
            "kvaf_source_urma_core",
            "kvaf_source_udma_device",
            "kvaf_source_ummu_device",
            "kvaf_urma_provider_common",
        ):
            self.assertNotIn(forbidden, target)
        self.assertIn("runtime_probe_exports.map", target)
        self.assertNotIn("aicpu_kvaf_urma_rt_probe_v2", exports)
        self.assertEqual(
            runtime_exports.count("aicpu_kvaf_urma_rt_probe_v2"), 1
        )
        self.assertIn(
            "canonical provider exports the isolated runtime probe",
            source(BUILD_PROVIDER),
        )
        self.assertIn(
            "canonical Device provider exports the isolated runtime probe",
            source(PACKAGE),
        )

    def test_unique_build_instance_and_source_config_are_bound_end_to_end(self) -> None:
        abi = source(ABI)
        provider = source(PROVIDER)
        smoke = source(SMOKE)
        provider_build = source(BUILD_PROVIDER)
        package = source(PACKAGE)
        preflight = source(PREFLIGHT)
        manifest_helper = source(MANIFEST_HELPER)
        runner = source(RUNNER)
        supervisor = source(SUPERVISOR)
        state = source(STATE_HELPER)

        self.assertIn("expected_provider_build_id_sha256", abi)
        self.assertIn("provider_build_id_sha256", abi)
        self.assertIn("KVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX", provider)
        self.assertIn("KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH", provider)
        self.assertIn('fail("provider_build_id")', smoke)
        self.assertIn(
            "provider_versioned_loader_key=verified-build-instance-v1", smoke
        )
        self.assertIn(
            "kvaf-runtime-probe-provider-source-config-v1", provider_build
        )
        self.assertIn("dd if=/dev/urandom bs=32 count=1", provider_build)
        self.assertIn("random-build-instance-sha256-v1", provider_build)
        self.assertIn("runtime_probe_provider_source_config_sha256", provider_build)
        self.assertIn("builder_script_sha256", provider_build)
        self.assertIn("runtime_probe_provider_builder_script_sha256", provider_build)
        self.assertIn("current_provider_builder_sha", manifest_helper)
        self.assertIn(
            "provider build script changed after the verified provider build",
            manifest_helper,
        )
        self.assertIn('libkvaf_rt_${kvaf_runtime_probe_provider_build_id:0:16}.so', provider_build)
        self.assertIn("runtime_probe_provider_filename", package)
        self.assertIn("runtime_probe_provider_filename", preflight)
        self.assertIn("audit_a5_registered_aicpu_packages.sh", package)
        self.assertIn("audit_a5_registered_aicpu_packages.sh", preflight)
        self.assertGreaterEqual(package.count("run_registered_package_audit"), 3)
        self.assertIn("--candidate-soname", preflight)
        self.assertIn("--replace-archive", preflight)
        self.assertLess(
            package.index("run_registered_package_audit", package.index("# Parse every")),
            package.index("deployment_started=1"),
        )
        self.assertLess(
            package.index("registration_update_started=1"),
            package.rindex("run_registered_package_audit"),
        )
        self.assertIn("--provider-build-id", runner)
        stamp_check = provider_build[
            provider_build.index('[[ "$(manifest_value schema') :
            provider_build.index(
                'die "internal v3 success-stamp validation failed"'
            )
        ]
        self.assertRegex(
            stamp_check,
            re.compile(
                r"manifest_value device_provider_sha256.*?"
                r'== "\$\{device_provider_sha\}"',
                re.DOTALL,
            ),
        )
        self.assertRegex(
            stamp_check,
            re.compile(
                r"manifest_value runtime_probe_provider_artifact_sha256.*?"
                r'== "\$\{runtime_probe_provider_sha\}"',
                re.DOTALL,
            ),
        )
        terminal_blocked = runner.index("elif (( probe_rc == 5 ))")
        terminal_blocked_body = runner[terminal_blocked:]
        self.assertIn('"${runtime_probe_provider_filename}"', terminal_blocked_body)
        self.assertIn('"${runtime_probe_provider_build_id}"', terminal_blocked_body)
        self.assertIn('"${runtime_probe_provider_sha}"', terminal_blocked_body)
        for key in (
            "provider_soname=",
            "provider_build_id_sha256=",
            "expected_packaged_provider_artifact_sha256=",
        ):
            self.assertIn(key, supervisor)
            self.assertIn(key.removesuffix("="), state)

    def test_every_mode_uses_transactional_supervisor(self) -> None:
        runner = source(RUNNER)
        supervisor = source(SUPERVISOR)
        fd_validation = supervisor.index("kvaf_fd8_target=")
        fd8_lock = supervisor.index("flock -n 8")
        fd9_lock = supervisor.index("flock -n 9")
        publication = supervisor.index("# Publication order")
        quarantine = supervisor.index(
            '"${KVAF_RUNTIME_SUPERVISOR_QUARANTINE}" '
            '"${kvaf_quarantine_tmp}"',
            publication,
        )
        active = supervisor.index(
            '"${KVAF_RUNTIME_SUPERVISOR_ACTIVE}" "${kvaf_active_tmp}"',
            publication,
        )
        ready = supervisor.index(
            '"${KVAF_RUNTIME_SUPERVISOR_READY}" "${kvaf_ready_tmp}"',
            publication,
        )
        worker = supervisor.index(
            '"${kvaf_worker_exec_path}" "$@"',
            publication,
        )
        terminal = supervisor.index(
            "kvaf_supervisor_publish_terminal", worker
        )
        self.assertEqual(
            [fd_validation, fd8_lock, fd9_lock, quarantine, active, ready,
             worker, terminal],
            sorted(
                [fd_validation, fd8_lock, fd9_lock, quarantine, active,
                 ready, worker, terminal]
            ),
        )
        quarantine_body = supervisor[
            supervisor.index("kvaf_quarantine_tmp=") :
            supervisor.index("kvaf_active_tmp=")
        ]
        active_body = supervisor[
            supervisor.index("kvaf_active_tmp=") :
            supervisor.index("kvaf_ready_tmp=")
        ]
        self.assertEqual(quarantine_body.count("reason="), 1)
        self.assertIn(
            "reason=runtime_public_probe_supervisor_prelaunch",
            quarantine_body,
        )
        self.assertEqual(active_body.count("reason="), 1)
        self.assertIn(
            "reason=supervisor_owns_locks_and_worker_lifetime", active_body
        )
        self.assertNotIn("${probe_rc} -ne 4", runner)
        state = source(STATE_HELPER)
        self.assertIn("KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE", state)
        self.assertIn("KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK", state)
        self.assertIn("kvaf_safe_terminal", runner)
        self.assertIn('setsid "${kvaf_supervisor_script}"', runner)
        self.assertIn("supervisor_start_ticks=", runner)
        self.assertIn("KVAF_NPU_IDLE_STATUS=idle", runner)
        self.assertIn("KVAF_ENDPOINT_RESOLVER", runner)
        self.assertIn("KVAF_DURABLE_STATE_HELPER", runner)
        self.assertIn("--mode discover|init-device|create-delete", runner)
        self.assertIn("kvaf_try_recover_completed_runtime_probe", runner)
        self.assertIn("worker_sha256=", runner)
        self.assertIn(
            "KVAF_A5_RUNTIME_URMA_SAFE_TERMINAL_RECOVERED", runner
        )
        self.assertIn("supervisor_start_ticks", runner)
        self.assertIn("kvaf_runtime_probe_read_proc_state_and_start_ticks", runner)
        self.assertIn("${kvaf_actual_state} != Z", runner)
        self.assertNotIn("awk '{print $22}'", runner)
        self.assertIn(
            "terminal_contract=runtime-probe-exact-terminal-v2", supervisor
        )
        self.assertIn(
            "lock_contract=inherited-supervisor-worker-v4-held-flock-pinned-worker-fd",
            supervisor,
        )
        self.assertIn('exec 7<"${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}"', supervisor)
        self.assertIn("sha256sum -- /proc/$$/fd/7", supervisor)
        self.assertNotIn(
            '"${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" "$@"', supervisor
        )
        self.assertIn(
            "schema=kvaf-a5-runtime-urma-probe-quarantine-v1", supervisor
        )
        self.assertIn(
            "schema=kvaf-a5-runtime-urma-probe-active-v2", supervisor
        )
        self.assertIn(
            "schema=kvaf-a5-runtime-urma-probe-ready-v1", supervisor
        )
        self.assertIn(
            "schema=kvaf-a5-runtime-urma-probe-terminal-v1", supervisor
        )
        self.assertIn("KVAF_A5_RUNTIME_URMA_RECOVERY_ONLY_OK", runner)
        self.assertIn("kvaf_publish_runtime_probe_blocked", runner)
        self.assertIn("kvaf_runtime_probe_recovered=2", runner)
        self.assertIn(
            "KVAF_A5_RUNTIME_URMA_BLOCKED_TERMINAL_RECOVERED", runner
        )
        self.assertNotIn('[[ ${kvaf_terminal_rc} != 5 ]] || return 1', runner)
        self.assertIn("LC_ALL=C stat -c '%u:%h:%F'", runner)
        recovery_call = runner.index(
            "kvaf_try_recover_completed_runtime_probe || true"
        )
        verification_call = runner.index("kvaf_verify_direct_provider_build_v3")
        self.assertLess(recovery_call, verification_call)
        recovery_body = runner[
            runner.index("kvaf_try_recover_completed_runtime_probe()") :
            runner.index("mode=\n")
        ]
        self.assertNotIn("KVAF_A5_RUNTIME_URMA_PROBE_BINARY", recovery_body)
        self.assertNotIn("actual_probe_sha", recovery_body)
        self.assertNotIn("ASCEND_HOME_PATH", recovery_body)
        self.assertIn("unknown or positional argument", runner)
        self.assertIn("queue-depth|--provider-soname", runner)
        self.assertEqual(runner.count('setsid "${kvaf_supervisor_script}"'), 1)
        self.assertIn("${KVAF_RUNTIME_SUPERVISOR_MODE} == discover", supervisor)
        self.assertNotIn("KVAF_A5_RUNTIME_URMA_BASE_SYMBOL_SURFACE_PRESENT", state)
        self.assertIn("KVAF_A5_RUNTIME_ACCEPTED_INIT_DEVICE", state)
        self.assertIn("kvaf_expected_mode", state)

    def test_symbol_gate_covers_every_required_control_group(self) -> None:
        abi = source(ABI)
        provider = source(PROVIDER)
        smoke = source(SMOKE)
        for group in (
            "KVAF_URMA_RUNTIME_PROBE_CANDIDATE_MODERN_QUEUE_SYMBOLS",
            "KVAF_URMA_RUNTIME_PROBE_CURRENT_ENDPOINT_DISCOVERY_SYMBOLS",
            "KVAF_URMA_RUNTIME_PROBE_CURRENT_RESOURCE_LIFECYCLE_SYMBOLS",
            "KVAF_URMA_RUNTIME_PROBE_CURRENT_CTP_LINK_SYMBOLS",
            "KVAF_URMA_RUNTIME_PROBE_CURRENT_UB_MANAGEMENT_IO_SYMBOLS",
            "KVAF_URMA_RUNTIME_PROBE_CURRENT_CHECKED_TEARDOWN_SYMBOLS",
            "KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS",
        ):
            self.assertIn(group, abi)
            self.assertIn(group, smoke)
        self.assertIn("UINT64_C(0x7fffcb9e000001ff)", abi)
        self.assertIn("official_function_semantics=unproven", smoke)
        # Modern queue APIs are deliberately presence-only at this stage:
        # known public-wrapper failure paths make blind behavioral calls unsafe.
        self.assertIn('KVAF_PROBE_OPTIONAL_SYMBOL("urma_alloc_jfr"', provider)
        for symbol in (
            "urma_str_to_eid",
            "urma_get_device_by_eid",
            "urma_get_eid_list",
            "urma_free_eid_list",
            "urma_alloc_token_id",
            "urma_free_token_id",
            "urma_get_tp_list",
            "urma_import_jetty_ex",
            "urma_post_jetty_send_wr",
            "urma_post_jetty_recv_wr",
            "urma_modify_jetty",
            "urma_query_jetty",
            "urma_modify_jfr",
            "urma_query_jfr",
            "urma_get_async_event",
            "urma_ack_async_event",
        ):
            self.assertIn(f'KVAF_PROBE_OPTIONAL_SYMBOL("{symbol}"', provider)
        self.assertNotIn("api.alloc_jfr(", provider)
        self.assertNotIn("api.alloc_jfs(", provider)

    def test_current_direct_ub_symbol_manifest_is_exact(self) -> None:
        abi = source(ABI)
        provider = source(PROVIDER)
        expected = {
            "INIT": (0, "urma_init"),
            "UNINIT": (1, "urma_uninit"),
            "GET_DEVICE": (2, "urma_get_device_by_name"),
            "CREATE_CONTEXT": (3, "urma_create_context"),
            "DELETE_CONTEXT": (4, "urma_delete_context"),
            "CREATE_JFC": (5, "urma_create_jfc"),
            "DELETE_JFC": (6, "urma_delete_jfc"),
            "CREATE_JFR": (7, "urma_create_jfr"),
            "DELETE_JFR": (8, "urma_delete_jfr"),
            "REGISTER_SEG": (33, "urma_register_seg"),
            "UNREGISTER_SEG": (34, "urma_unregister_seg"),
            "IMPORT_SEG": (35, "urma_import_seg"),
            "UNIMPORT_SEG": (36, "urma_unimport_seg"),
            "POLL_JFC": (39, "urma_poll_jfc"),
            "CREATE_JETTY": (40, "urma_create_jetty"),
            "DELETE_JETTY": (41, "urma_delete_jetty"),
            "UNIMPORT_JETTY": (43, "urma_unimport_jetty"),
            "QUERY_DEVICE": (46, "urma_query_device"),
            "STR_TO_EID": (47, "urma_str_to_eid"),
            "GET_DEVICE_BY_EID": (48, "urma_get_device_by_eid"),
            "GET_EID_LIST": (49, "urma_get_eid_list"),
            "FREE_EID_LIST": (50, "urma_free_eid_list"),
            "ALLOC_TOKEN_ID": (51, "urma_alloc_token_id"),
            "FREE_TOKEN_ID": (52, "urma_free_token_id"),
            "GET_TP_LIST": (53, "urma_get_tp_list"),
            "IMPORT_JETTY_EX": (54, "urma_import_jetty_ex"),
            "POST_JETTY_SEND_WR": (55, "urma_post_jetty_send_wr"),
            "POST_JETTY_RECV_WR": (56, "urma_post_jetty_recv_wr"),
            "MODIFY_JETTY": (57, "urma_modify_jetty"),
            "QUERY_JETTY": (58, "urma_query_jetty"),
            "MODIFY_JFR": (59, "urma_modify_jfr"),
            "QUERY_JFR": (60, "urma_query_jfr"),
            "GET_ASYNC_EVENT": (61, "urma_get_async_event"),
            "ACK_ASYNC_EVENT": (62, "urma_ack_async_event"),
        }
        bit_pattern = re.compile(
            r"#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_(\w+)\s+"
            r"(?:\\\n\s*)?\(UINT64_C\(1\) << (\d+)\)"
        )
        allocated = {
            name: int(bit) for name, bit in bit_pattern.findall(abi)
        }
        self.assertEqual(len(allocated.values()), len(set(allocated.values())))
        self.assertNotIn(63, allocated.values())
        for name, (bit, symbol) in expected.items():
            self.assertEqual(allocated.get(name), bit, name)
            self.assertIn(f'"{symbol}"', provider, symbol)
        mask = sum(1 << bit for bit, _ in expected.values())
        self.assertEqual(mask, 0x7FFFCB9E000001FF)
        self.assertNotIn(" aiv_control_symbol_surface=", source(SMOKE))

    def test_runtime_symbols_are_bound_to_one_root_dso(self) -> None:
        provider = source(PROVIDER)
        self.assertIn("info.dli_fbase != required_dso_base", provider)
        self.assertIn("KVAF_URMA_RUNTIME_PROBE_SYMBOL_ORIGIN_MISMATCH", provider)
        self.assertIn("symbol_resolved_from_different_dso", provider)

    def test_every_mode_requires_idle_and_undocumented_caller_risk(self) -> None:
        runner = source(RUNNER)
        supervisor = source(SUPERVISOR)
        acceptance = (
            "I_ACCEPT_UNDOCUMENTED_AICPU_URMA_"
            "ABI_SOURCE_CALLER_AND_CLEANUP_RISK"
        )
        contract = (
            "undocumented-aicpu-urma-abi-source-caller-cleanup-"
            "risk-accepted-v1"
        )
        self.assertIn(f"KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA={acceptance}", runner)
        self.assertIn(contract, runner)
        self.assertIn(contract, supervisor)
        self.assertLess(
            runner.index("KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA:-"),
            runner.index('setsid "${kvaf_supervisor_script}"'),
        )
        self.assertLess(
            runner.index("KVAF_A5_NPU_IDLE_PROVEN"),
            runner.index("KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA:-"),
        )
        acceptance_gate = runner.index(
            "KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA:-"
        )
        endpoint_gate = runner.index(
            "if [[ ${mode} != discover ]]", acceptance_gate
        )
        self.assertLess(
            acceptance_gate, endpoint_gate
        )

    def test_discover_does_not_require_an_endpoint(self) -> None:
        provider = source(PROVIDER)
        smoke = source(SMOKE)
        runner = source(RUNNER)
        self.assertIn(
            "args->mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER ||",
            provider,
        )
        self.assertIn(
            "options->mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER ||",
            smoke,
        )
        self.assertIn(
            "resource-bearing mode requires exactly one URMA device and EID index",
            runner,
        )

    def test_loader_diagnostics_are_sanitized_and_not_errno_guesses(self) -> None:
        provider = source(PROVIDER)
        self.assertIn("kvaf_runtime_probe_record_loader_error", provider)
        self.assertIn("result->loader_error", provider)
        self.assertIn("KVAF_URMA_RUNTIME_PROBE_LOADER_ERROR_TRUNCATED", provider)
        self.assertIn(
            "KVAF_URMA_RUNTIME_PROBE_STAGE_LOAD_LIBRARY, -1, 0",
            provider,
        )
        self.assertNotIn("operation_errno == 0 ?", provider)


if __name__ == "__main__":
    unittest.main()

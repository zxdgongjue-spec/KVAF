from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "access_operator/a5/smoke/hccp_aiv_channel_probe.cpp"
RUNNER = ROOT / "scripts/run_a5_hccp_aiv_channel_probe.sh"
CMAKE = ROOT / "access_operator/a5/CMakeLists.txt"
BUILD = ROOT / "scripts/build_a5_aiv_urma.sh"
HCOMM_HEADER = (
    ROOT / "access_operator/a5/include/kvaf/access_operator/a5/hcomm_urma_resources.h"
)
HCOMM_SOURCE = ROOT / "access_operator/a5/src/hcomm_urma_resources.cpp"


class HccpAivChannelProbeContractTests(unittest.TestCase):
    def test_probe_scope_is_channel_snapshot_not_data_plane_execution(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("ChannelEngine::kCpu", source)
        self.assertIn("ChannelRole::kClient", source)
        self.assertIn("ChannelEngine::kAiv", source)
        self.assertIn("ChannelRole::kServer", source)
        self.assertIn("PollChannelReady", source)
        self.assertIn("SnapshotDirectChannel", source)
        self.assertIn("snapshot.wqe_count != 0", source)
        self.assertIn("KVAF_A5_HCCP_AIV_CHANNEL_SNAPSHOT_OK", source)
        self.assertIn("WQE_SUBMISSIONS=0", source)
        self.assertNotIn("aclrtCreateStream", source)
        self.assertNotIn("LaunchAiv", source)
        self.assertNotIn("aclrtSynchronizeStream", source)

    def test_device_precedes_host_memory_teardown(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        host = source[source.index("RunResult RunHost") : source.index("RunResult RunDevice")]
        device = source[source.index("RunResult RunDevice") : source.index("bool WriteFinal")]
        self.assertLess(
            host.index('WaitForMarker(options, "device.teardown_ok"'),
            host.index("RunResult result = CloseHost"),
        )
        self.assertLess(
            device.index("RunResult result = CloseDevice"),
            device.index('WriteMarker(options, "device.teardown_ok"'),
        )
        self.assertIn("RetainUnsafeHost", host)
        self.assertIn("host_ddr_exported", host)
        self.assertIn("must_wait_for_device_teardown", source)
        self.assertIn("FailDevice", device)
        self.assertIn("RetainUnsafeDevice", source)
        self.assertIn("try {", source)
        self.assertIn("catch (const std::exception& error)", source)
        self.assertIn("ResourceSignalGuard", source)
        self.assertIn("SIGINT, SIGTERM, SIGHUP, SIGQUIT", source)
        self.assertIn("RetainUnsafeProcess", source)

    def test_nonce_bound_markers_and_channel_name_are_required(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        header = HCOMM_HEADER.read_text(encoding="utf-8")
        hcomm = HCOMM_SOURCE.read_text(encoding="utf-8")
        self.assertIn("run_id", source)
        self.assertIn("MarkerMatches", source)
        self.assertIn("Exactly one optional detail line", source)
        self.assertIn("O_EXCL", source)
        self.assertIn("::fsync", source)
        self.assertIn("const std::string& channel_name = {}", header)
        self.assertIn("std::string channel_name", hcomm)
        self.assertIn("kMaximumChannelNameBytes = 191U", hcomm)
        self.assertIn("SetChannelName(&desc, impl_->channel_name)", hcomm)
        self.assertIn("impl_->channel_name.clear();", hcomm)
        self.assertIn("impl_->channel = channel;", hcomm)
        self.assertIn("partially initialized channel", hcomm)
        self.assertIn("HCOMM may return a partially created endpoint", hcomm)
        self.assertIn("reserving HCOMM registration tracking failed", hcomm)
        self.assertIn("reserving HCOMM import tracking failed", hcomm)
        self.assertIn("HcommMemUnimport", hcomm)
        self.assertIn("unproven_resource_operation", hcomm)
        self.assertIn("did not prove that a failed create/import", hcomm)
        self.assertNotIn("SetChannelName(&desc, channel_name.c_str())", hcomm)

    def test_old_header_anonymous_mode_is_explicit_and_hardware_gated(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        main = source[source.index("int main(") :]

        self.assertIn("--print-capabilities", source)
        self.assertIn(
            '<< "usage: " << program << " --print-capabilities\\n"', source
        )
        self.assertIn("KVAF_HCCP_CHANNEL_NAME_SUPPORT=", source)
        self.assertIn("channel-name-unsupported", main)
        self.assertIn("legacy-anonymous-guard-fd-absent", main)
        self.assertIn("HasOpenResourceGuardFd", source)
        self.assertIn("legacy-anonymous-risk-ack", source)
        self.assertIn(
            "const std::string channel_name = supports_channel_name ? options.run_id : \"\";",
            source,
        )
        self.assertLess(
            main.index("if (!supports_channel_name && !options.allow_anonymous_channel_fallback)"),
            main.index("aclInit"),
        )

        self.assertIn("KVAF_HCCP_AIV_CHANNEL_LEGACY_ANONYMOUS", runner)
        self.assertIn(
            "I_ACCEPT_LEGACY_ANONYMOUS_CHANNEL_CROSS_PROCESS_PAIRING_RISK",
            runner,
        )
        self.assertIn("KVAF_HCCP_CHANNEL_NAME_SUPPORT=0", runner)
        self.assertIn("--print-capabilities", runner)
        self.assertLess(
            runner.index("channel_capability_output"),
            runner.index("query_a5_topology.py"),
        )
        self.assertIn("CONTRACT=best-effort-exclusive-legacy-anonymous-v1", runner)
        self.assertIn("exec 10>", runner)
        self.assertIn("flock -n 10", runner)
        self.assertIn("[k]vaf_a5_hccp_aiv_channel_probe", runner)
        self.assertIn("/data/.kvaf-state/KVAF", runner)
        self.assertIn("hccp-aiv-legacy-anonymous.active", runner)
        self.assertIn("marker_has_safe_terminal", runner)
        self.assertIn('--channel-name-mode "${channel_name_mode}"', runner)
        self.assertIn('--resource-guard-fd "${resource_guard_fd}"', runner)
        self.assertNotIn("legacy-anonymous-risk-ack)\n    channel_name_support=1", runner)

    def test_legacy_hcomm_qos_does_not_alias_ubc_queue_depth(self) -> None:
        hcomm = HCOMM_SOURCE.read_text(encoding="utf-8")
        header = HCOMM_HEADER.read_text(encoding="utf-8")
        source = SOURCE.read_text(encoding="utf-8")
        main = source[source.index("int main(") :]

        self.assertIn("void SetDefaultQos(T* desc)", hcomm)
        self.assertIn(
            "desc->remoteEndpoint.protocol == COMM_PROTOCOL_HCCS", hcomm
        )
        self.assertIn("desc->hccsAttr.qos = kDefaultQos;", hcomm)
        self.assertIn("Status ValidateLegacyUbcQueueDefaults", hcomm)
        self.assertIn("Status PrepareChannelDescriptor", hcomm)
        self.assertIn("ValidateChannelDescriptor", header)
        self.assertIn("COMM_PROTOCOL_UBC_CTP", hcomm)
        self.assertIn("COMM_PROTOCOL_UBC_TP", hcomm)
        self.assertIn(
            "first_union_word != std::numeric_limits<std::uint32_t>::max()",
            hcomm,
        )
        self.assertIn("return ValidateLegacyUbcQueueDefaults(*desc);", hcomm)
        self.assertIn("refusing HcommChannelCreate before an HCOMM", hcomm)
        self.assertIn('"resource operation"', hcomm)
        self.assertNotIn(
            "else if constexpr (HasHccsQos<T>::value) {\n"
            "    desc->hccsAttr.qos = kDefaultQos;\n"
            "  }",
            hcomm,
        )
        self.assertIn("channel-descriptor-preflight-failed", main)
        self.assertLess(
            main.index("ValidateChannelDescriptor"), main.index("aclInit")
        )

    def test_runner_has_strict_safety_gates_but_no_worker_kill_path(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "I_ACKNOWLEDGE_HCCP_AIV_CHANNEL_ZERO_WQE_PROBE", runner
        )
        self.assertIn("preflight_a5_hccp_hal_capabilities.sh", runner)
        self.assertIn("check_a5_npu_idle.py", runner)
        self.assertIn("resolve_a5_urma_eids.py", runner)
        self.assertIn("query_a5_topology.py", runner)
        self.assertIn("flock -n 9", runner)
        self.assertIn("workers_retained=1", runner)
        self.assertIn("mapfile -t lines", runner)
        self.assertIn("worker_running", runner)
        self.assertIn("RUNTIME_CLEANUP_INCOMPLETE", runner)
        self.assertIn("WQE_SUBMISSIONS=0", runner)
        self.assertNotIn("timeout --kill-after", runner)
        self.assertNotIn("kill ", runner)
        self.assertNotIn("aclrt", runner)
        self.assertNotIn("npu-smi set", runner)

    def test_legacy_anonymous_guard_is_scoped_to_npu_and_eid_pair(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")

        self.assertIn("legacy_workers_for_current_npu", runner)
        self.assertIn(
            "hccp-aiv-channel.npu${physical_device_id}.lock", runner
        )
        self.assertIn('lock_dir="${legacy_state_dir}"', runner)
        self.assertIn(
            "KVAF_GLOBAL_STATE_DIR to select this lock would let another container evade",
            runner,
        )
        self.assertIn(
            "hccp-aiv-anonymous-channel.${legacy_scope}.lock", runner
        )
        self.assertIn(
            "hccp-aiv-legacy-anonymous.${legacy_scope}.active", runner
        )
        self.assertIn("--physical-device-id ${physical_device_id}", runner)
        self.assertIn("KVAF_A5_HCCP_LEGACY_ANONYMOUS_OTHER_NPU_RETAINED", runner)
        self.assertIn("Never remove the old file during this migration", runner)
        self.assertLess(
            runner.rindex("legacy_workers_for_current_npu"),
            runner.index("legacy_boot_id=$(cat /proc/sys/kernel/random/boot_id)"),
        )

    def test_runner_is_shell_syntax_valid(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(RUNNER)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_target_is_built_by_the_hcomm_build_script(self) -> None:
        cmake = CMAKE.read_text(encoding="utf-8")
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("kvaf_a5_hccp_aiv_channel_probe", cmake)
        self.assertIn("smoke/hccp_aiv_channel_probe.cpp", cmake)
        self.assertIn("${KVAF_HCOMM_LIBRARY}", cmake)
        self.assertIn("kvaf_a5_hccp_aiv_channel_probe", build)


if __name__ == "__main__":
    unittest.main()

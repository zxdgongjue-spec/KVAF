from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "access_operator/a5/smoke/hccp_resource_lifecycle_probe.cpp"
)
RUNNER = ROOT / "scripts/run_a5_hccp_resource_lifecycle_probe.sh"
CMAKE = ROOT / "access_operator/a5/CMakeLists.txt"
BUILD = ROOT / "scripts/build_a5_aiv_urma.sh"
HCOMM_RESOURCES = ROOT / "access_operator/a5/src/hcomm_urma_resources.cpp"


class HccpResourceLifecycleProbeContractTests(unittest.TestCase):
    def test_runner_has_explicit_ack_idle_and_static_abi_gates(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "I_ACKNOWLEDGE_HCCP_RESOURCE_LIFECYCLE_PROBE", source
        )
        self.assertIn("preflight_a5_hccp_hal_capabilities.sh", source)
        self.assertIn("KVAF_HCCP_HCOMM_RESOURCE_SURFACE=complete-symbols-only", source)
        self.assertIn("check_a5_npu_idle.py", source)
        self.assertIn("KVAF_A5_NPU_IDLE_PROVEN", source)
        self.assertIn("resolve_a5_urma_eids.py", source)
        self.assertIn("flock -n 9", source)
        self.assertIn("WQE_SUBMISSIONS=0", source)
        self.assertNotIn("aclrt", source)
        self.assertNotIn("npu-smi set", source)
        self.assertNotIn("kill ", source)

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

    def test_binary_uses_resource_before_memory_reverse_teardown(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("HcommUrmaResources", source)
        self.assertIn("HostDdrDeviceMapping", source)
        self.assertIn("OpenEndpoint", source)
        self.assertIn("RegisterMemory", source)
        self.assertIn("ExportMemory", source)
        self.assertIn("CloseWithRetry", source)
        self.assertIn("retain-process-owned-resources", source)
        self.assertIn("retain-host-ddr-mapping", source)
        self.assertIn("KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN", source)
        self.assertIn("KVAF_A5_HCCP_RESOURCE_PROBE_FINALIZE_SKIPPED", source)
        self.assertIn("aclrtResetDevice(clean process context)", source)
        self.assertLess(
            source.index("if (!result.safe_to_finalize)"),
            source.index("aclrtResetDevice(options.logical_device_id)"),
        )

        host = source[source.index("RunResult RunHostDdr") : source.index("RunResult RunDeviceHbm")]
        self.assertLess(host.index("resources->Close()"), host.index("mapping->Close()"))

    def test_probe_is_built_with_the_legacy_hcomm_resource_backend(self) -> None:
        cmake = CMAKE.read_text(encoding="utf-8")
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("kvaf_a5_hccp_resource_lifecycle_probe", cmake)
        self.assertIn("smoke/hccp_resource_lifecycle_probe.cpp", cmake)
        self.assertIn("${KVAF_HCOMM_LIBRARY}", cmake)
        self.assertIn(
            "kvaf_configure_a5_target(kvaf_a5_hccp_resource_lifecycle_probe)",
            cmake,
        )
        self.assertIn("kvaf_a5_hccp_resource_lifecycle_probe", build)

    def test_hcomm_close_retains_dependency_state_but_drains_safe_siblings(self) -> None:
        source = HCOMM_RESOURCES.read_text(encoding="utf-8")
        close = source[source.index("Status HcommUrmaResources::Close()") :]
        self.assertIn("ChannelHandle destroying_channel", close)
        self.assertIn("Status first_failure = Status::OK()", close)
        self.assertIn("remember_failure(\"HcommMemUnimport\"", close)
        self.assertIn("remember_failure(\"HcommMemUnreg\"", close)
        self.assertIn("impl_->imported_descriptors.erase", close)
        self.assertIn("impl_->registrations.erase", close)
        self.assertIn("if (!impl_->imported_descriptors.empty())", close)
        self.assertIn("if (!impl_->registrations.empty())", close)
        self.assertLess(
            close.index("HcommChannelDestroy"),
            close.index("HcommMemUnimport"),
        )
        self.assertLess(
            close.index("HcommMemUnimport"),
            close.index("HcommMemUnreg"),
        )
        self.assertLess(
            close.index("HcommMemUnreg"),
            close.index("HcommEndpointDestroy"),
        )


if __name__ == "__main__":
    unittest.main()

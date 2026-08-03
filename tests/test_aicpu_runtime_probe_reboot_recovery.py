from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_a5_aicpu_runtime_urma_probe.sh"
AUDIT = ROOT / "docs/a5_official_urma_capability_audit.md"


class RuntimeProbeRebootRecoveryTests(unittest.TestCase):
    def test_blocked_state_is_recovered_only_after_exact_reboot_evidence(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        start = runner.index(
            "kvaf_try_recover_rebooted_runtime_probe_blocked()"
        )
        end = runner.index(
            "kvaf_try_recover_completed_runtime_probe()", start
        )
        body = runner[start:end]
        self.assertIn("kvaf-a5-runtime-urma-probe-blocked-v1", body)
        self.assertIn("kvaf_runtime_probe_file_has_exact_keys", body)
        self.assertIn("kvaf_runtime_probe_terminal_evidence_matches", body)
        self.assertIn("${kvaf_boot_id} != \"${kvaf_current_boot_id}\"", body)
        self.assertIn("${kvaf_terminal_rc} == 5", body)
        self.assertIn("worker_exited_log_synced", body)
        self.assertIn("KVAF_A5_RUNTIME_URMA_PROCESS_CONTAMINATED", body)
        self.assertIn("KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK", body)
        self.assertIn("remove \"${kvaf_quarantine}\"", body)
        self.assertNotIn("setsid", body)
        self.assertNotIn("KVAF_A5_RUNTIME_URMA_PROBE_BINARY", body)

        recovery_call = runner.index(
            "kvaf_try_recover_rebooted_runtime_probe_blocked || true"
        )
        probe_recovery_call = runner.index(
            "kvaf_try_recover_completed_runtime_probe || true"
        )
        verification_call = runner.index("kvaf_verify_direct_provider_build_v3")
        self.assertLess(recovery_call, probe_recovery_call)
        self.assertLess(probe_recovery_call, verification_call)
        self.assertIn("KVAF_A5_RUNTIME_URMA_REBOOT_RECOVERY_ONLY_OK", runner)
        self.assertIn("exit 6", runner[recovery_call:probe_recovery_call])

    def test_old_boot_held_owner_recovery_does_not_invent_a_terminal(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        start = runner.index("kvaf_try_recover_rebooted_runtime_probe_hold()")
        end = runner.index("kvaf_try_recover_completed_runtime_probe()", start)
        body = runner[start:end]
        self.assertIn("kvaf-a5-runtime-urma-probe-quarantine-v1", body)
        self.assertIn("kvaf-a5-runtime-urma-probe-active-v2", body)
        self.assertIn("kvaf-a5-runtime-urma-probe-ready-v1", body)
        self.assertIn("provider_build_id_sha256", body)
        self.assertIn("expected_packaged_provider_artifact_sha256", body)
        self.assertIn("${kvaf_boot_id} != \"${kvaf_current_boot_id}\"", body)
        self.assertIn("! -e ${kvaf_terminal_file}", body)
        self.assertIn("remove \"${kvaf_active}\"", body)
        self.assertIn("remove \"${kvaf_ready_file}\"", body)
        self.assertIn("remove \"${kvaf_quarantine}\"", body)
        self.assertNotIn("kvaf_runtime_probe_log_is_safe_terminal", body)
        self.assertNotIn("setsid", body)
        call = runner.index(
            "kvaf_try_recover_rebooted_runtime_probe_hold || true"
        )
        completed = runner.index(
            "kvaf_try_recover_completed_runtime_probe || true"
        )
        self.assertLess(call, completed)
        self.assertIn(
            "KVAF_A5_RUNTIME_URMA_HELD_REBOOT_RECOVERY_ONLY_OK",
            runner[call:completed],
        )

    def test_documentation_does_not_equate_dlclose_with_cleanup_proof(self) -> None:
        audit = AUDIT.read_text(encoding="utf-8")
        self.assertIn("dlclose", audit)
        self.assertIn("不能证明 constructor", audit)
        self.assertIn("boot ID", audit)
        self.assertIn("不会用操作者的风险接受文本替代 cleanup 证据", audit)


if __name__ == "__main__":
    unittest.main()

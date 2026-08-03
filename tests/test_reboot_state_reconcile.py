from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reconcile_a5_direct_urma_after_reboot.sh"


class RebootStateReconcileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_uses_one_fixed_machine_wide_state_transaction(self):
        self.assertIn("state_dir=/data/.kvaf-state/KVAF", self.source)
        self.assertIn("boot_id_file=/proc/sys/kernel/random/boot_id", self.source)
        self.assertNotIn("KVAF_DIRECT_STATE_DIR:-", self.source)
        self.assertNotIn("KVAF_GLOBAL_STATE_DIR:-", self.source)

    def test_prepare_proof_is_one_shot_and_durable(self):
        self.assertIn(
            'an unconsumed reboot proof already exists', self.source
        )
        self.assertIn(
            'kvaf-a5-direct-reboot-proof-v1', self.source
        )
        self.assertIn(
            '"${KVAF_DURABLE_STATE_HELPER}" publish', self.source
        )
        self.assertIn("reboot-reconcile.lock", self.source)
        self.assertIn("flock -n 5", self.source)
        self.assertIn("mktemp", self.source)
        self.assertIn("kvaf_git_head=%s", self.source)
        self.assertIn("status --porcelain", self.source)
        self.assertIn(
            "KVAF checkout must be clean for reboot reconciliation",
            self.source,
        )
        self.assertIn("reconcile_script_sha256=%s", self.source)
        self.assertIn("durable_helper_sha256=%s", self.source)
        self.assertIn(
            "KVAF checkout changed between reboot prepare and retire",
            self.source,
        )

    def test_retire_requires_changed_boot_and_free_owner_locks(self):
        changed = self.source.index(
            'if [[ "${before_boot_id}" == "${current_boot_id}" ]]'
        )
        deploy = self.source.index('flock -n 9')
        active = self.source.index('flock -n 8')
        archive = self.source.index('archive_dir=${retired_root}')
        self.assertLess(changed, deploy)
        self.assertLess(deploy, active)
        self.assertLess(active, archive)
        self.assertIn("OS reboot is not proven", self.source)

    def test_retire_is_host_only_and_serializes_cross_npu_launchers(self):
        self.assertIn('if [[ -e /.dockerenv ]]', self.source)
        self.assertIn(
            "run reboot state reconciliation on Server B host", self.source
        )
        self.assertIn("experimental/cross-npu-risk.lock", self.source)
        self.assertIn(
            "experimental/npu-2-reuse/reuse-risk.lock", self.source
        )
        self.assertIn("flock -n 7", self.source)
        self.assertIn("flock -n 6", self.source)

    def test_retire_holds_legacy_deploy_locks_and_rejects_old_markers(self):
        self.assertIn("/run/lock/kvaf-a5-direct-urma.lock", self.source)
        self.assertIn("flock -n 4", self.source)
        self.assertIn("flock -n 3", self.source)
        self.assertIn("flock -n 11", self.source)
        self.assertIn("legacy_state_files=(", self.source)
        self.assertIn(
            "legacy active/quarantine state must be resolved separately",
            self.source,
        )

    def test_retire_matches_exact_pre_reboot_marker_snapshot(self):
        self.assertIn("marker_%s_present=0", self.source)
        self.assertIn("marker_%s_stat=%s", self.source)
        self.assertIn("marker_%s_sha256=%s", self.source)
        compare = self.source.index(
            'cmp -s "${expected_snapshot}" "${actual_snapshot}"'
        )
        archive = self.source.index('archive_dir=${retired_root}')
        self.assertLess(compare, archive)
        self.assertIn(
            "canonical state changed after reboot proof was prepared",
            self.source,
        )

    def test_process_observation_errors_fail_closed(self):
        self.assertIn("pgrep python3", self.source)
        self.assertIn('case "${pgrep_rc}" in', self.source)
        self.assertIn(
            "could not inspect KVAF processes after reboot", self.source
        )

    def test_archive_is_durable_before_canonical_state_removal(self):
        archive_sync = self.source.index(
            '"${archive_dir}/retirement.manifest"'
        )
        removal_comment = self.source.index(
            "# Remove only after the archive and manifest are durable."
        )
        durable_remove = self.source.index(
            '"${KVAF_DURABLE_STATE_HELPER}" remove', removal_comment
        )
        self.assertLess(archive_sync, removal_comment)
        self.assertLess(removal_comment, durable_remove)
        self.assertNotIn('rm -f "${active_file}"', self.source)
        self.assertNotIn('rm -f "${quarantine_file}"', self.source)
        self.assertIn(
            'archived_marker_paths+=("${archive_dir}/${name}")',
            self.source,
        )
        self.assertIn('"${archived_marker_paths[@]}"', self.source)
        self.assertIn('"${retired_root}"\n  "${state_dir}"', self.source)

    def test_results_locks_and_experiment_evidence_are_not_retired(self):
        self.assertIn(
            "Lock files,\n# results and every experimental/risk artifact "
            "are deliberately retained.",
            self.source,
        )
        self.assertNotIn("results/*", self.source)
        self.assertNotIn("experimental/*", self.source)


if __name__ == "__main__":
    unittest.main()

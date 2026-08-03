from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_a5_cross_npu2_reuse_detached.sh"


class CrossNpu2ReuseHostLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_invalid_invocation_stops_before_host_or_docker_checks(self) -> None:
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage:", completed.stderr)

    def test_start_requires_an_explicit_short_risk_flag(self) -> None:
        self.assertIn(
            "start --accept-npu2-reuse-risk", self.source
        )
        start = self.source.index("start_experiment()")
        worker = self.source.index("record_worker_exit()")
        self.assertIn(
            '[[ "$1" == --accept-npu2-reuse-risk ]]',
            self.source[start:worker],
        )
        self.assertIn(
            "I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK",
            self.source,
        )

    def test_identity_is_fixed_and_source_must_be_clean(self) -> None:
        for required in (
            "vllm-ascend-cs-kvaf",
            "cc3cc0e8e08bf4882c290dc3ed0a5b40a45312ce",
            "KVAF-cross-npu2-${KVAF_EXPERIMENT_COMMIT}",
            "530f5998a0b646d48db858170a9996ff",
            "30ef7336d212459a9317fd6647f6fa15",
            "0000000000000002",
            "attempt.selected",
            "A5_AIV_URMA_cs",
            "diff-index --quiet HEAD --",
            "ls-files --others --exclude-standard",
        ):
            self.assertIn(required, self.source)

    def test_launch_is_claim_guarded_atomic_and_detached(self) -> None:
        start = self.source.index("start_experiment()")
        worker = self.source.index("record_worker_exit()")
        body = self.source[start:worker]
        claim = body.index('[[ ! -e "${KVAF_CLAIM}"')
        preflight = body.index("run_read_only_preflight")
        publish = body.index("reserve_launch_generation")
        detached = body.index("docker exec -d")
        self.assertLess(claim, preflight)
        self.assertLess(preflight, publish)
        self.assertLess(claim, publish)
        self.assertLess(publish, detached)
        self.assertIn('ln "${temporary}" "${path}"', self.source)
        self.assertIn("KVAF_DETACHED_LAUNCHER_COMMIT", body)
        self.assertIn("KVAF_RESERVATION", self.source)

    def test_dispatch_failure_is_not_a_coordinator_result(self) -> None:
        start = self.source.index("record_dispatch_unproven()")
        end = self.source.index("start_experiment()")
        body = self.source[start:end]
        self.assertIn("KVAF_DISPATCH_UNPROVEN", body)
        self.assertIn("coordinator_completion=unproven", body)
        self.assertNotIn("KVAF_WORKER_RC", body)

    def test_worker_revalidates_identity_and_runs_only_zero_wqe(self) -> None:
        worker = self.source.index("run_worker()")
        status = self.source.index("show_status()")
        body = self.source[worker:status]
        for required in (
            "worker mode is container-only",
            "detached worker authorization is missing",
            "trap record_worker_exit EXIT",
            "read_reservation",
            "require_log_identity",
            "launcher checkout moved before worker start",
            "rev-parse HEAD",
            "prepare_pinned_worktree",
            "run_a5_cross_npu2_reuse_experiment.sh",
            "zero-wqe",
        ):
            self.assertIn(required, body)
        self.assertLess(
            body.index("trap record_worker_exit EXIT"),
            body.index("require_canonical_state_root"),
        )
        self.assertLess(
            body.index("trap record_worker_exit EXIT"),
            body.index("exec >>"),
        )

    def test_worker_result_is_atomic_and_strictly_parsed(self) -> None:
        self.assertIn('publish_file_once "${KVAF_WORKER_RC}" 400', self.source)
        parser = self.source.index("read_canonical_worker_rc()")
        status = self.source.index("show_status()")
        body = self.source[parser:status]
        self.assertIn("mapfile -t kvaf_rc_lines", body)
        self.assertIn("${#kvaf_rc_lines[@]} == 1", body)
        self.assertIn("[1-9][0-9]{0,2}", body)
        self.assertIn("<= 255", body)

    def test_log_and_reservation_have_exact_identity(self) -> None:
        for required in (
            "kvaf-a5-cross-npu2-detached-launch-v2",
            "kvaf-a5-cross-npu2-detached-log-v2",
            "stat -c '%h:%a'",
            "1:${mode}",
            "kvaf_expected_reservation_sha",
            "kvaf_actual_reservation_sha",
            "kvaf_expected_log_header_sha",
            "kvaf_actual_log_header_sha",
            'head -n 8 "${KVAF_LOG}"',
        ):
            self.assertIn(required, self.source)

    def test_pinned_coordinator_commit_is_rechecked_externally(self) -> None:
        worker = self.source.index("run_worker()")
        parser = self.source.index("read_canonical_worker_rc()")
        body = self.source[worker:parser]
        self.assertIn(
            "KVAF_CROSS_NPU2_REUSE_EXPECTED_SOURCE_COMMIT", body
        )
        immediate = body.index(
            "pinned experiment commit changed immediately before coordinator start"
        )
        coordinator = body.index(
            "run_a5_cross_npu2_reuse_experiment.sh zero-wqe"
        )
        self.assertLess(immediate, coordinator)

    def test_exactly_one_nonce_bound_worker_can_own_the_generation(self) -> None:
        for required in (
            "KVAF_WORKER_SELECTED",
            "KVAF_DETACHED_LAUNCH_NONCE",
            "launch_nonce=${KVAF_LAUNCH_NONCE}",
            'publish_file_once "${KVAF_WORKER_SELECTED}" 400',
            "another detached worker already selected this generation",
            "KVAF_WORKER_OWNS_GENERATION=1",
        ):
            self.assertIn(required, self.source)

    def test_unfinished_status_is_never_success(self) -> None:
        status = self.source.index("show_status()")
        watch = self.source.index("watch_status()")
        body = self.source[status:watch]
        running = body.index("KVAF_DETACHED_STATUS=running-or-unproven")
        self.assertIn("return 5", body[running:])
        self.assertIn("KVAF_NPU2_REUSE_ZERO_WQE_RC=0", body)
        self.assertIn("KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=PASS", body)

    def test_status_and_watch_are_observation_only(self) -> None:
        status = self.source.index("show_status()")
        dispatch = self.source.index("(( $# >= 1 )) || usage")
        observation = self.source[status:dispatch]
        for forbidden in (
            r"\bkill\b",
            r"\bpkill\b",
            r"\brm\b",
            r"\breset\b",
            r"\bunlink\b",
            r"docker exec",
        ):
            self.assertIsNone(re.search(forbidden, observation), forbidden)
        self.assertIn("Ctrl+C does not signal the detached worker", observation)
        self.assertIn("read_reservation", observation)
        self.assertNotIn("rev-parse HEAD", observation)


if __name__ == "__main__":
    unittest.main()

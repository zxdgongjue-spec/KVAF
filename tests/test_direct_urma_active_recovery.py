#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"
CLASSIFIER = ROOT / "scripts" / "classify_a5_direct_urma_outcome.py"
RUN_ID = "0123456789abcdef0123456789abcdef"


def final_marker(role: str, outcome: str, *, armed: str = "1") -> str:
    return "\n".join(
        (
            "schema=kvaf-a5-direct-final-v1",
            f"run_id={RUN_ID}",
            f"role={role}",
            f"outcome={outcome}",
            "reason=recovery_test",
            f"resource_armed={armed}",
            "stream_quiesced=1",
            "queue_drained=1",
            "provider_closed=1",
            "backing_released=1",
            "host_mapping_unregistered=1",
            "allocations_released=1",
            "",
        )
    )


class ActiveRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("manifest_value()\n")
        end = source.index("\nacquire_persistent_session_guards()\n")
        cls.production_functions = source[start:end]

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / "repo"
        self.result_root = self.temp / "persistent-state" / "results"
        self.result = self.result_root / "run"
        self.exchange = self.result / "exchange"
        self.exchange.mkdir(parents=True)
        self.state = self.temp / "direct-urma.active"
        self.quarantine = self.temp / "direct-urma.quarantine"
        self.bundle = self.result / CLASSIFIER.name
        shutil.copyfile(CLASSIFIER, self.bundle)
        self.classifier_sha = hashlib.sha256(
            self.bundle.read_bytes()
        ).hexdigest()
        self.manifest = self.result / "manifest.txt"
        self.write_manifest("verified")

        # macOS /usr/bin/realpath lacks GNU -e/-m. Recovery targets Linux, but
        # this tiny test shim lets the production shell function run unchanged.
        self.bin = self.temp / "bin"
        self.bin.mkdir()
        realpath = self.bin / "realpath"
        realpath.write_text(
            "#!/bin/sh\n"
            "mode=$1\n"
            "case $mode in -e|-m) shift ;; *) mode=plain ;; esac\n"
            "python3 - \"$mode\" \"$1\" <<'PY'\n"
            "import os, sys\n"
            "mode, path = sys.argv[1:]\n"
            "if mode == '-e' and not os.path.exists(path):\n"
            "    raise SystemExit(1)\n"
            "print(os.path.realpath(path))\n"
            "PY\n",
            encoding="utf-8",
        )
        realpath.chmod(0o755)

    def write_manifest(self, assurance: str) -> None:
        self.manifest.write_text(
            "\n".join(
                (
                    f"run_id={RUN_ID}",
                    f"teardown_assurance={assurance}",
                    f"outcome_classifier_sha256={self.classifier_sha}",
                    "",
                )
            ),
            encoding="utf-8",
        )

    def write_active(
        self, phase: str, host_pid: int, device_pid: int
    ) -> None:
        self.state.write_text(
            "\n".join(
                (
                    "schema=kvaf-a5-direct-active-v2",
                    f"run_id={RUN_ID}",
                    f"phase={phase}",
                    "logical_device_id=2",
                    f"host_pid={host_pid}",
                    f"device_pid={device_pid}",
                    "host_start_ticks=0",
                    "device_start_ticks=0",
                    "host_observed_exe=none",
                    "device_observed_exe=none",
                    "worker_expected_exe=/tmp/kvaf-test-worker",
                    f"result_dir={self.result}",
                    "updated_at=2026-07-31T00:00:00Z",
                    "",
                )
            ),
            encoding="utf-8",
        )

    def recover(self) -> subprocess.CompletedProcess[str]:
        script = "\n".join(
            (
                "set -u",
                f"SCRIPT_DIR={shlex_quote(str(ROOT / 'scripts'))}",
                f"ROOT_DIR={shlex_quote(str(self.root))}",
                "KVAF_DIRECT_RESULT_ROOT="
                f"{shlex_quote(str(self.result_root))}",
                "KVAF_DIRECT_ACTIVE_SESSION_FILE="
                f"{shlex_quote(str(self.state))}",
                "KVAF_DIRECT_QUARANTINE_FILE="
                f"{shlex_quote(str(self.quarantine))}",
                "KVAF_DURABLE_STATE_HELPER="
                f"{shlex_quote(str(ROOT / 'scripts' / 'durable_state_file.py'))}",
                self.production_functions,
                "recover_stale_active_session",
            )
        )
        environment = os.environ.copy()
        environment["PATH"] = f"{self.bin}:{environment['PATH']}"
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_before_launch_recovers_without_endpoint_marker(self) -> None:
        self.write_active("before_host_launch", 0, 0)
        result = self.recover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.state.exists())

    def test_host_started_requires_safe_host_marker(self) -> None:
        self.write_active("host_started", 101, 0)
        (self.exchange / "host.final").write_text(
            final_marker("host", "FAIL_SAFE"), encoding="utf-8"
        )
        result = self.recover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.state.exists())

    def test_device_intent_requires_both_endpoint_markers(self) -> None:
        self.write_active("device_launch_intent", 101, 0)
        (self.exchange / "host.final").write_text(
            final_marker("host", "FAIL_SAFE"), encoding="utf-8"
        )
        result = self.recover()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.state.exists())

    def test_device_started_recovers_only_with_safe_pair(self) -> None:
        for phase in (
            "device_started",
            "safe_terminal_process_stuck",
            "pair_terminal_unproven_process_stuck",
        ):
            with self.subTest(phase=phase):
                self.write_active(phase, 101, 202)
                (self.exchange / "host.final").write_text(
                    final_marker("host", "FAIL_SAFE"), encoding="utf-8"
                )
                (self.exchange / "device.final").write_text(
                    final_marker("device", "FAIL_SAFE"), encoding="utf-8"
                )
                result = self.recover()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(self.state.exists())

    def test_phase_pid_inconsistency_fails_closed(self) -> None:
        self.write_active("host_started", 101, 202)
        result = self.recover()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.state.exists())

    def test_unverified_armed_terminal_promotes_to_quarantine(self) -> None:
        self.write_active("host_started", 101, 0)
        (self.exchange / "host.final").write_text(
            final_marker("host", "FAIL_SAFE", armed="1"), encoding="utf-8"
        )
        self.write_manifest("experimental-unverified")
        result = self.recover()
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertFalse(self.state.exists())
        self.assertTrue(self.quarantine.exists())
        self.assertIn(
            "reason=stale_unverified_udma_armed_terminal_FAIL_SAFE",
            self.quarantine.read_text(encoding="utf-8"),
        )

    def test_one_valid_unsafe_marker_promotes_to_quarantine(self) -> None:
        self.write_active("device_started", 101, 202)
        (self.exchange / "host.final").write_text(
            final_marker("host", "FAIL_UNSAFE"), encoding="utf-8"
        )
        result = self.recover()
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertFalse(self.state.exists())
        self.assertTrue(self.quarantine.exists())
        self.assertIn(
            "reason=stale_explicit_fail_unsafe",
            self.quarantine.read_text(encoding="utf-8"),
        )

    def test_changed_classifier_keeps_active_barrier(self) -> None:
        self.write_active("host_started", 101, 0)
        (self.exchange / "host.final").write_text(
            final_marker("host", "FAIL_SAFE", armed="0"), encoding="utf-8"
        )
        self.write_manifest("verified")
        self.bundle.write_text("raise SystemExit(0)\n", encoding="utf-8")
        result = self.recover()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.state.exists())

    def test_recovery_evidence_survives_repository_reclone(self) -> None:
        self.write_active("device_started", 101, 202)
        (self.exchange / "host.final").write_text(
            final_marker("host", "FAIL_SAFE"), encoding="utf-8"
        )
        (self.exchange / "device.final").write_text(
            final_marker("device", "FAIL_SAFE"), encoding="utf-8"
        )
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir()

        result = self.recover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.state.exists())


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    unittest.main()

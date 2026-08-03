#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import fcntl
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "validate_a5_cross_npu2_reuse_risk.py"
STATE_HELPER = ROOT / "scripts" / "a5_cross_npu2_reuse_state.sh"
SPEC = importlib.util.spec_from_file_location("validate_npu2_reuse", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


class CrossNpu2ReuseRiskValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "KVAF"
        self.state.mkdir()
        self.state = self.state.resolve()
        self.proc = self.root / "proc"
        self.proc.mkdir()
        self.proc = self.proc.resolve()
        self.token = "0123456789abcdef"

        source_result = self.state / "results" / "source"
        source_result.mkdir(parents=True)
        self.source_evidence = types.SimpleNamespace(
            result_dir=source_result,
            host=types.SimpleNamespace(pid=VALIDATOR.SOURCE_HOST_PID),
            device=types.SimpleNamespace(pid=VALIDATOR.SOURCE_DEVICE_PID),
        )

        self.source_lock = self.root / "source.lock"
        self.source_lock.touch(mode=0o600)
        holder_program = (
            "import fcntl,sys,time; "
            "f=open(sys.argv[1],'r+'); "
            "fcntl.flock(f,fcntl.LOCK_EX); "
            "print('READY',flush=True); time.sleep(120)"
        )
        self.holder = subprocess.Popen(
            [sys.executable, "-c", holder_program, str(self.source_lock)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self._stop_holder)
        assert self.holder.stdout is not None
        self.assertEqual(self.holder.stdout.readline().strip(), "READY")
        for pid in (VALIDATOR.SOURCE_HOST_PID, VALIDATOR.SOURCE_DEVICE_PID):
            fd_dir = self.proc / str(pid) / "fd"
            fd_dir.mkdir(parents=True)
        (self.proc / str(VALIDATOR.SOURCE_HOST_PID) / "fd" / "7").symlink_to(
            self.source_lock
        )

        worker, lane, result = VALIDATOR._expected_paths(self.state)
        worker.parent.mkdir(parents=True)
        worker.write_bytes(b"old experimental worker\n")
        self.lane = lane
        self.result = result
        exchange = result / "exchange"
        exchange.mkdir(parents=True)
        (lane / "direct-urma.active").write_bytes(
            VALIDATOR._expected_active_bytes(self.state)
        )
        (lane / "direct-urma.active.lock").touch(mode=0o600)
        self._write_manifest()
        (result / "host.log").write_text(
            "KVAF_UMMU_CONTEXT_READY fd_mode=process_fd_dup owner_pid=116040\n"
            "KVAF_URMA_MGMT_READY role=server jetty_id=502 local_eid=known\n"
            "KVAF_URMA_MGMT_ERROR stage=server_handshake raw_errno=110\n",
            encoding="utf-8",
        )
        (result / "device.log").write_bytes(b"")
        self._write_prior_process(
            VALIDATOR.PRIOR_HOST_PID,
            VALIDATOR.PRIOR_HOST_START_TICKS,
            "Z",
        )
        self._write_prior_process(
            VALIDATOR.PRIOR_DEVICE_PID,
            VALIDATOR.PRIOR_DEVICE_START_TICKS,
            "X",
        )

    def _stop_holder(self) -> None:
        if self.holder.poll() is None:
            self.holder.terminate()
            try:
                self.holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.holder.kill()
                self.holder.wait(timeout=5)
        if self.holder.stdout is not None:
            self.holder.stdout.close()
        if self.holder.stderr is not None:
            self.holder.stderr.close()

    def _write_manifest(self, **updates: str) -> None:
        values = {
            "kvaf_commit": VALIDATOR.PRIOR_SOURCE_COMMIT,
            "aicpu_affinity_run_mode": "risk-accepted-cross-npu2-v1",
            "execution_safety_contract": "experimental-risk-accepted-cross-npu2-v1",
            "cross_npu_artifact_id": VALIDATOR.PRIOR_ARTIFACT_ID,
            "cross_npu_source_run_id": VALIDATOR.SOURCE_RUN_ID,
            "cross_npu_source_quarantine_sha256": VALIDATOR.SOURCE_QUARANTINE_SHA256,
            "run_id": VALIDATOR.PRIOR_TARGET_RUN_ID,
            "logical_device_id": "2",
            "physical_device_id": "2",
            "host_eid": "00000000007f040000100000df0a0b00",
            "device_eid": "000000000000020000100000df004100",
            "management_mode": "ub",
            "management_jetty_id": "502",
            "h2d_transfer_count": "0",
            "h2d_bytes_per_transfer": "1024",
            "h2d_batch_iterations": "1",
        }
        values.update(updates)
        (self.result / "manifest.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def _write_prior_process(self, pid: int, ticks: int, state: str) -> None:
        process = self.proc / str(pid)
        process.mkdir(parents=True, exist_ok=True)
        fields = [state, *("0" for _ in range(18)), str(ticks)]
        (process / "stat").write_text(
            f"{pid} (old worker) {' '.join(fields)}\n", encoding="utf-8"
        )

    def validate(self):
        with mock.patch.object(
            VALIDATOR, "_validate_source_incident", return_value=self.source_evidence
        ):
            return VALIDATOR.validate_reuse_evidence(
                artifact_token=self.token,
                acceptance=VALIDATOR.ACCEPTANCE_PHRASE,
                state_root=self.state,
                proc_root=self.proc,
                source_lock=self.source_lock,
            )

    def test_accepts_exact_zombie_pair_and_emits_sealed_kvaf_evidence(self) -> None:
        evidence = self.validate()
        self.assertEqual(evidence.prior_host.state, "Z")
        self.assertEqual(evidence.prior_device.state, "X")
        self.assertEqual(evidence.artifact_token, self.token)
        self.assertEqual(
            evidence.state_dir,
            self.state
            / "experimental"
            / "npu-2-reuse"
            / VALIDATOR.SOURCE_RUN_ID
            / VALIDATOR.PRIOR_TARGET_RUN_ID
            / self.token,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            VALIDATOR._emit(evidence)
        lines = output.getvalue().splitlines()
        self.assertTrue(lines)
        self.assertTrue(all(line.startswith("KVAF_") for line in lines))
        self.assertIn(
            f"KVAF_REUSE_SOURCE_RUN_ID={VALIDATOR.SOURCE_RUN_ID}", lines
        )
        self.assertIn(
            f"KVAF_REUSE_PRIOR_TARGET_RUN_ID={VALIDATOR.PRIOR_TARGET_RUN_ID}",
            lines,
        )
        self.assertTrue(
            any(line.startswith("KVAF_REUSE_PRIOR_ACTIVE_SHA256=") for line in lines)
        )
        self.assertIn(
            f"KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN={self.token}", lines
        )
        self.assertTrue(any(line.startswith("KVAF_REUSE_STATE_DIR=") for line in lines))

    def test_accepts_absent_prior_pids_but_rejects_pid_reuse_and_live_pid(self) -> None:
        shutil.rmtree(self.proc / str(VALIDATOR.PRIOR_HOST_PID))
        shutil.rmtree(self.proc / str(VALIDATOR.PRIOR_DEVICE_PID))
        evidence = self.validate()
        self.assertEqual(evidence.prior_host.state, "absent")
        self.assertEqual(evidence.prior_device.state, "absent")

        self._write_prior_process(
            VALIDATOR.PRIOR_HOST_PID,
            VALIDATOR.PRIOR_HOST_START_TICKS + 1,
            "Z",
        )
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "PID was reused"):
            self.validate()
        self._write_prior_process(
            VALIDATOR.PRIOR_HOST_PID, VALIDATOR.PRIOR_HOST_START_TICKS, "S"
        )
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "still live"):
            self.validate()

    def test_rejects_changed_active_final_and_nonzero_manifest(self) -> None:
        active = self.lane / "direct-urma.active"
        active.write_bytes(active.read_bytes() + b"override=yes\n")
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "contents changed"):
            self.validate()
        active.write_bytes(VALIDATOR._expected_active_bytes(self.state))

        (self.result / "exchange" / "host.final").write_text(
            "unsafe\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "host final appeared"):
            self.validate()
        (self.result / "exchange" / "host.final").unlink()

        self._write_manifest(h2d_transfer_count="1")
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "h2d_transfer_count"):
            self.validate()

    def test_rejects_wrong_acceptance_token_unheld_lock_and_missing_fd(self) -> None:
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "exact phrase"):
            with mock.patch.object(
                VALIDATOR,
                "_validate_source_incident",
                return_value=self.source_evidence,
            ):
                VALIDATOR.validate_reuse_evidence(
                    artifact_token=self.token,
                    acceptance="wrong",
                    state_root=self.state,
                    proc_root=self.proc,
                    source_lock=self.source_lock,
                )
        for token in ("a" * 15, "A" * 16, "a" * 17):
            with self.subTest(token=token):
                with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "16 lowercase"):
                    with mock.patch.object(
                        VALIDATOR,
                        "_validate_source_incident",
                        return_value=self.source_evidence,
                    ):
                        VALIDATOR.validate_reuse_evidence(
                            artifact_token=token,
                            acceptance=VALIDATOR.ACCEPTANCE_PHRASE,
                            state_root=self.state,
                            proc_root=self.proc,
                            source_lock=self.source_lock,
                        )

        source_fd = self.proc / str(VALIDATOR.SOURCE_HOST_PID) / "fd" / "7"
        source_fd.unlink()
        with self.assertRaisesRegex(
            VALIDATOR.ReuseRiskError, "not open in either pinned worker"
        ):
            self.validate()
        source_fd.symlink_to(self.source_lock)

        self._stop_holder()
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "unexpectedly unlocked"):
            self.validate()

    def test_rejects_symlinked_generation_and_does_not_mutate_old_evidence(self) -> None:
        active = self.lane / "direct-urma.active"
        before = (active.read_bytes(), active.stat().st_ino)
        generation_parent = (
            self.state
            / "experimental"
            / "npu-2-reuse"
            / VALIDATOR.SOURCE_RUN_ID
            / VALIDATOR.PRIOR_TARGET_RUN_ID
        )
        generation_parent.mkdir(parents=True)
        (generation_parent / self.token).symlink_to(self.root / "escape")
        with self.assertRaisesRegex(VALIDATOR.ReuseRiskError, "symlink"):
            self.validate()
        self.assertEqual((active.read_bytes(), active.stat().st_ino), before)

    def test_state_helper_is_fixed_generation_and_rejects_override(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        syntax = subprocess.run(
            [bash, "-n", str(STATE_HELPER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        program = (
            f"source {STATE_HELPER!s}; "
            "kvaf_configure_cross_npu2_reuse_state_paths /tmp/root "
            f"{VALIDATOR.SOURCE_RUN_ID} {VALIDATOR.PRIOR_TARGET_RUN_ID} {self.token}; "
            "printf '%s\\n' \"$KVAF_DIRECT_STATE_DIR\""
        )
        result = subprocess.run(
            [bash, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "/data/.kvaf-state/KVAF/experimental/npu-2-reuse/"
            f"{VALIDATOR.SOURCE_RUN_ID}/{VALIDATOR.PRIOR_TARGET_RUN_ID}/{self.token}",
        )
        override = subprocess.run(
            [
                bash,
                "-c",
                f"set -e; export KVAF_DIRECT_STATE_DIR=/tmp/escape; {program}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(override.returncode, 0)
        self.assertIn("cannot override", override.stderr)


if __name__ == "__main__":
    unittest.main()

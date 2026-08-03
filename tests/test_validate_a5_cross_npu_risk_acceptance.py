#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "validate_a5_cross_npu_risk_acceptance.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_a5_cross_npu_risk_acceptance", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)

RUN_ID = "530f5998a0b646d48db858170a9996ff"
HOST_PID = 103721
DEVICE_PID = 105213
HOST_START_TICKS = 110500001
DEVICE_START_TICKS = 110500777


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CrossNpuRiskAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.result_root = self.root / "state" / "results"
        self.result = self.result_root / "a5_direct_urma_h2d_known_incident"
        self.exchange = self.result / "exchange"
        self.exchange.mkdir(parents=True)
        self.proc_root = self.root / "proc"
        self.proc_root.mkdir()

        self.worker = self.root / "kvaf_a5_aiv_direct_urma_h2d_smoke"
        self.worker.write_bytes(b"known pinned direct URMA worker\n")
        self.worker.chmod(0o755)

        self.manifest = self.result / "manifest.txt"
        self.write_manifest()
        self.device_final = self.exchange / "device.final"
        self.write_device_final()
        self.write_process(
            HOST_PID, HOST_START_TICKS, "host", state="S"
        )
        self.write_process(
            DEVICE_PID, DEVICE_START_TICKS, "device", state="S"
        )

        self.quarantine = self.root / "direct-urma.quarantine"
        self.write_quarantine()

    def write_manifest(self, **replacements: str) -> None:
        values = {
            "kvaf_commit": VALIDATOR.KNOWN_SOURCE_COMMIT,
            "run_id": RUN_ID,
            "logical_device_id": "7",
            "physical_device_id": "7",
            "npu_idle_preflight_override": "0",
            "host_eid": VALIDATOR.KNOWN_HOST_EID,
            "device_eid": VALIDATOR.KNOWN_DEVICE_EID,
            "host_urma_device": VALIDATOR.KNOWN_HOST_URMA_DEVICE,
            "host_eid_index": "0",
            "device_urma_device": VALIDATOR.KNOWN_DEVICE_URMA_DEVICE,
            "device_eid_index": "0",
            "management_mode": "ub",
            "management_jetty_id": "507",
            "h2d_workload_mode": "explicit-transfer-shape",
            "h2d_transfer_count": "0",
            "h2d_batch_iterations": "1",
            "h2d_total_bytes_per_batch": "0",
            "h2d_total_bytes": "0",
            "binary_sha256": sha256(self.worker),
            "teardown_assurance": "verified",
        }
        values.update(replacements)
        self.manifest.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def write_device_final(self, **replacements: str) -> None:
        values = {
            "schema": "kvaf-a5-direct-final-v1",
            "run_id": RUN_ID,
            "role": "device",
            "pid": str(DEVICE_PID),
            "outcome": "FAIL_UNSAFE",
            "reason": "device_provider_open_cleanup_failed",
            "resource_armed": "1",
            "stream_quiesced": "1",
            "queue_drained": "1",
            "provider_closed": "0",
            "backing_released": "0",
            "host_mapping_unregistered": "1",
            "allocations_released": "0",
        }
        values.update(replacements)
        self.device_final.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def process_arguments(self, role: str) -> list[str]:
        return [
            str(self.worker),
            "--role",
            role,
            "--run-id",
            RUN_ID,
            "--logical-device-id",
            "7",
            "--transfer-count",
            "0",
            "--host-eid",
            VALIDATOR.KNOWN_HOST_EID,
            "--device-eid",
            VALIDATOR.KNOWN_DEVICE_EID,
            "--host-urma-device",
            VALIDATOR.KNOWN_HOST_URMA_DEVICE,
            "--device-urma-device",
            VALIDATOR.KNOWN_DEVICE_URMA_DEVICE,
            "--management-jetty-id",
            "507",
        ]

    def write_process(
        self,
        pid: int,
        start_ticks: int,
        role: str,
        *,
        state: str,
    ) -> None:
        process = self.proc_root / str(pid)
        process.mkdir(exist_ok=True)
        stat_tail = [state, *("0" for _ in range(18)), str(start_ticks)]
        (process / "stat").write_text(
            f"{pid} (kvaf worker) {' '.join(stat_tail)}\n",
            encoding="utf-8",
        )
        exe = process / "exe"
        if exe.exists() or exe.is_symlink():
            exe.unlink()
        exe.symlink_to(self.worker)
        arguments = self.process_arguments(role)
        (process / "cmdline").write_bytes(
            b"\x00".join(value.encode("utf-8") for value in arguments)
            + b"\x00"
        )

    def write_quarantine(self, **replacements: str) -> None:
        values = {
            "schema": "kvaf-a5-direct-quarantine-v1",
            "reason": "unsafe_provider_teardown",
            "run_id": RUN_ID,
            "logical_device_id": "7",
            "host_pid": str(HOST_PID),
            "device_pid": str(DEVICE_PID),
            "host_start_ticks": str(HOST_START_TICKS),
            "device_start_ticks": str(DEVICE_START_TICKS),
            "host_observed_exe": str(self.worker),
            "device_observed_exe": str(self.worker),
            "worker_expected_exe": str(self.worker),
            "result_dir": str(self.result),
            "created_at": "2026-08-03T05:01:42Z",
        }
        values.update(replacements)
        self.quarantine.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def validate(self, **replacements: object):
        arguments = {
            "quarantine_path": self.quarantine,
            "expected_quarantine_sha256": sha256(self.quarantine),
            "expected_run_id": RUN_ID,
            "source_logical_device_id": 7,
            "result_root": self.result_root,
            "proc_root": self.proc_root,
        }
        arguments.update(replacements)
        return VALIDATOR.validate_risk_acceptance(**arguments)

    def test_accepts_exact_known_live_incident(self) -> None:
        evidence = self.validate()
        self.assertEqual(evidence.run_id, RUN_ID)
        self.assertEqual(evidence.source_logical_device_id, 7)
        self.assertEqual(evidence.host.pid, HOST_PID)
        self.assertEqual(evidence.device.pid, DEVICE_PID)
        self.assertEqual(evidence.quarantine_sha256, sha256(self.quarantine))
        self.assertEqual(evidence.manifest_sha256, sha256(self.manifest))
        self.assertEqual(
            evidence.device_final_sha256, sha256(self.device_final)
        )

    def test_cli_emits_digest_bound_evidence(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--quarantine",
                str(self.quarantine),
                "--quarantine-sha256",
                sha256(self.quarantine),
                "--run-id",
                RUN_ID,
                "--source-logical-device-id",
                "7",
                "--result-root",
                str(self.result_root),
                "--proc-root",
                str(self.proc_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "KVAF_CROSS_NPU_RISK_ACCEPTANCE=verified\n", result.stdout
        )
        self.assertIn(
            f"KVAF_ACCEPTED_QUARANTINE_SHA256={sha256(self.quarantine)}\n",
            result.stdout,
        )
        self.assertEqual(result.stderr, "")

    def test_rejects_wrong_digest_run_id_and_source_npu(self) -> None:
        with self.subTest("digest"):
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "SHA256 does not match"
            ):
                self.validate(expected_quarantine_sha256="0" * 64)
        with self.subTest("run_id"):
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "run_id"
            ):
                self.validate(expected_run_id="1" * 32)
        with self.subTest("source_npu"):
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "only source logical NPU 7"
            ):
                self.validate(source_logical_device_id=2)

    def test_quarantine_parser_rejects_unknown_and_duplicate_keys(self) -> None:
        original = self.quarantine.read_text(encoding="utf-8")
        with self.subTest("unknown"):
            self.quarantine.write_text(
                original + "override=yes\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "unexpected keys"
            ):
                self.validate()
        with self.subTest("duplicate"):
            self.quarantine.write_text(
                original + "run_id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "duplicate key"
            ):
                self.validate()

    def test_rejects_nonzero_wqe_manifest(self) -> None:
        for key, value in (
            ("h2d_transfer_count", "1"),
            ("h2d_total_bytes_per_batch", "1024"),
            ("h2d_total_bytes", "1024"),
        ):
            with self.subTest(key=key):
                self.write_manifest(**{key: value})
                with self.assertRaisesRegex(
                    VALIDATOR.RiskAcceptanceError, key
                ):
                    self.validate()
                self.write_manifest()

    def test_rejects_any_other_device_failure_shape(self) -> None:
        for key, value in (
            ("reason", "some_other_failure"),
            ("stream_quiesced", "0"),
            ("queue_drained", "0"),
            ("provider_closed", "1"),
        ):
            with self.subTest(key=key):
                self.write_device_final(**{key: value})
                with self.assertRaisesRegex(
                    VALIDATOR.RiskAcceptanceError, key
                ):
                    self.validate()
                self.write_device_final()

    def test_rejects_pid_start_ticks_executable_and_zombie_changes(self) -> None:
        with self.subTest("start_ticks"):
            self.write_process(
                DEVICE_PID, DEVICE_START_TICKS + 1, "device", state="S"
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "start_ticks changed"
            ):
                self.validate()
        self.write_process(
            DEVICE_PID, DEVICE_START_TICKS, "device", state="S"
        )

        with self.subTest("executable"):
            replacement = self.root / "replacement-worker"
            replacement.write_bytes(b"replacement\n")
            (self.proc_root / str(DEVICE_PID) / "exe").unlink()
            (self.proc_root / str(DEVICE_PID) / "exe").symlink_to(
                replacement
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "executable changed"
            ):
                self.validate()
        self.write_process(
            DEVICE_PID, DEVICE_START_TICKS, "device", state="S"
        )

        with self.subTest("zombie"):
            self.write_process(
                DEVICE_PID, DEVICE_START_TICKS, "device", state="Z"
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "not live"
            ):
                self.validate()

        with self.subTest("stopped"):
            self.write_process(
                DEVICE_PID, DEVICE_START_TICKS, "device", state="T"
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "not live"
            ):
                self.validate()

    def test_rejects_cmdline_identity_change(self) -> None:
        cmdline = self.proc_root / str(DEVICE_PID) / "cmdline"
        arguments = self.process_arguments("device")
        role_index = arguments.index("--run-id") + 1
        arguments[role_index] = "a" * 32
        cmdline.write_bytes(
            b"\x00".join(value.encode("utf-8") for value in arguments)
            + b"\x00"
        )
        with self.assertRaisesRegex(
            VALIDATOR.RiskAcceptanceError, "--run-id"
        ):
            self.validate()

    def test_rejects_final_pid_change_and_new_host_final(self) -> None:
        with self.subTest("final_pid"):
            self.write_device_final(pid=str(DEVICE_PID + 1))
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "pid does not match"
            ):
                self.validate()
        self.write_device_final()

        with self.subTest("host_final"):
            (self.exchange / "host.final").write_text(
                "unexpected\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                VALIDATOR.RiskAcceptanceError, "must not have Host"
            ):
                self.validate()

    def test_rejects_result_dir_outside_pinned_root(self) -> None:
        outside = self.root / "outside" / "run"
        (outside / "exchange").mkdir(parents=True)
        self.write_quarantine(result_dir=str(outside))
        with self.assertRaisesRegex(
            VALIDATOR.RiskAcceptanceError, "direct child"
        ):
            self.validate()

    def test_rejects_symlink_quarantine(self) -> None:
        target = self.root / "quarantine-target"
        target.write_bytes(self.quarantine.read_bytes())
        self.quarantine.unlink()
        self.quarantine.symlink_to(target)
        with self.assertRaisesRegex(
            VALIDATOR.RiskAcceptanceError, "non-symlink"
        ):
            self.validate(
                expected_quarantine_sha256=sha256(target)
            )


if __name__ == "__main__":
    unittest.main()

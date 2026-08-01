#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "classify_a5_direct_urma_outcome.py"
SPEC = importlib.util.spec_from_file_location(
    "classify_a5_direct_urma_outcome", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
CLASSIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLASSIFIER
SPEC.loader.exec_module(CLASSIFIER)

RUN_ID = "0123456789abcdef0123456789abcdef"


def marker_text(
    role: str,
    outcome: str,
    *,
    run_id: str = RUN_ID,
    resource_armed: str = "1",
    stream_quiesced: str = "1",
    queue_drained: str = "1",
    provider_closed: str = "1",
    backing_released: str = "1",
    host_mapping_unregistered: str = "1",
    allocations_released: str = "1",
) -> str:
    return "\n".join(
        (
            "schema=kvaf-a5-direct-final-v1",
            f"run_id={run_id}",
            f"role={role}",
            f"outcome={outcome}",
            "reason=test_reason",
            f"resource_armed={resource_armed}",
            f"stream_quiesced={stream_quiesced}",
            f"queue_drained={queue_drained}",
            f"provider_closed={provider_closed}",
            f"backing_released={backing_released}",
            f"host_mapping_unregistered={host_mapping_unregistered}",
            f"allocations_released={allocations_released}",
            "",
        )
    )


class ClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.host = self.root / "host.final"
        self.device = self.root / "device.final"

    def write_pair(
        self,
        host_outcome: str = "PASS",
        device_outcome: str = "PASS",
    ) -> None:
        self.host.write_text(
            marker_text("host", host_outcome), encoding="utf-8"
        )
        self.device.write_text(
            marker_text("device", device_outcome), encoding="utf-8"
        )

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--run-id",
                RUN_ID,
                "--host-final",
                str(self.host),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_pair_pass(self) -> None:
        self.write_pair()
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS\n")
        self.assertEqual(result.stderr, "")

    def test_pair_safe_failure(self) -> None:
        self.write_pair(device_outcome="FAIL_SAFE")
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(result.stdout, "FAIL_SAFE\n")
        self.assertEqual(result.stderr, "")

    def test_explicit_unsafe_outcome_wins(self) -> None:
        self.write_pair(host_outcome="FAIL_UNSAFE")
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 4)
        self.assertEqual(result.stdout, "FAIL_UNSAFE\n")

    def test_explicit_unsafe_outcome_wins_over_missing_peer(self) -> None:
        self.host.write_text(
            marker_text("host", "FAIL_UNSAFE"), encoding="utf-8"
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(result.stdout, "FAIL_UNSAFE\n")

    def test_explicit_unsafe_outcome_wins_over_malformed_peer(self) -> None:
        self.host.write_text("broken\n", encoding="utf-8")
        self.device.write_text(
            marker_text("device", "FAIL_UNSAFE"), encoding="utf-8"
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(result.stdout, "FAIL_UNSAFE\n")

    def test_safe_outcome_requires_every_safety_proof(self) -> None:
        self.host.write_text(marker_text("host", "PASS"), encoding="utf-8")
        self.device.write_text(
            marker_text("device", "FAIL_SAFE", queue_drained="0"),
            encoding="utf-8",
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertEqual(result.stdout, "UNPROVEN\n")
        self.assertIn("lacks safety proof", result.stderr)

    def test_pass_requires_transport_to_have_been_armed(self) -> None:
        self.host.write_text(
            marker_text("host", "PASS", resource_armed="0"),
            encoding="utf-8",
        )
        self.device.write_text(
            marker_text("device", "PASS"), encoding="utf-8"
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("PASS requires resource_armed=1", result.stderr)

    def test_pass_requires_complete_local_cleanup(self) -> None:
        self.host.write_text(
            marker_text("host", "PASS", backing_released="0"),
            encoding="utf-8",
        )
        self.device.write_text(
            marker_text("device", "PASS"), encoding="utf-8"
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("backing_released", result.stderr)

        self.host.write_text(
            marker_text("host", "PASS"), encoding="utf-8"
        )
        self.device.write_text(
            marker_text(
                "device", "PASS", allocations_released="0"
            ),
            encoding="utf-8",
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("allocations_released", result.stderr)

    def test_safe_outcome_requires_role_specific_release(self) -> None:
        self.host.write_text(
            marker_text(
                "host", "FAIL_SAFE", host_mapping_unregistered="0"
            ),
            encoding="utf-8",
        )
        self.device.write_text(
            marker_text("device", "PASS"), encoding="utf-8"
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("host_mapping_unregistered", result.stderr)

        self.host.write_text(
            marker_text("host", "PASS"), encoding="utf-8"
        )
        self.device.write_text(
            marker_text(
                "device",
                "FAIL_SAFE",
                backing_released="0",
                allocations_released="0",
            ),
            encoding="utf-8",
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 3, result.stderr)

    def test_duplicate_and_malformed_lines_are_unproven(self) -> None:
        self.write_pair()
        with self.host.open("a", encoding="utf-8") as output:
            output.write("role=host\n")
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("duplicate key", result.stderr)

        self.host.write_text(
            marker_text("host", "PASS") + "malformed\n",
            encoding="utf-8",
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("expected non-empty key=value", result.stderr)

    def test_missing_field_is_unproven(self) -> None:
        self.write_pair()
        text = self.device.read_text(encoding="utf-8")
        self.device.write_text(
            text.replace("provider_closed=1\n", ""), encoding="utf-8"
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("missing required keys", result.stderr)

    def test_schema_run_id_and_role_must_match(self) -> None:
        cases = (
            marker_text("host", "PASS").replace(
                "schema=kvaf-a5-direct-final-v1",
                "schema=kvaf-a5-direct-final-v0",
            ),
            marker_text("host", "PASS", run_id="f" * 32),
            marker_text("device", "PASS"),
        )
        for text in cases:
            with self.subTest(text=text.splitlines()[:3]):
                self.write_pair()
                self.host.write_text(text, encoding="utf-8")
                result = self.run_cli("--device-final", str(self.device))
                self.assertEqual(result.returncode, 6)
                self.assertEqual(result.stdout, "UNPROVEN\n")

    def test_missing_marker_is_unproven(self) -> None:
        self.host.write_text(marker_text("host", "PASS"), encoding="utf-8")
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("cannot stat", result.stderr)

    def test_single_host_accepts_safe_pre_device_failure(self) -> None:
        self.host.write_text(
            marker_text(
                "host", "FAIL_SAFE", resource_armed="0"
            ),
            encoding="utf-8",
        )
        result = self.run_cli("--single-role", "host")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(result.stdout, "FAIL_SAFE\n")
        self.assertEqual(result.stderr, "")

    def test_single_host_pass_is_safe_but_not_pair_acceptance(self) -> None:
        self.host.write_text(marker_text("host", "PASS"), encoding="utf-8")
        result = self.run_cli("--single-role", "host")
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, "FAIL_SAFE\n")

    def test_non_boolean_value_and_symlink_are_unproven(self) -> None:
        self.write_pair()
        self.host.write_text(
            marker_text("host", "PASS", stream_quiesced="true"),
            encoding="utf-8",
        )
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("must be 0 or 1", result.stderr)

        target = self.root / "target.final"
        target.write_text(marker_text("host", "PASS"), encoding="utf-8")
        self.host.unlink()
        self.host.symlink_to(target)
        result = self.run_cli("--device-final", str(self.device))
        self.assertEqual(result.returncode, 6)
        self.assertIn("non-symlink", result.stderr)


if __name__ == "__main__":
    unittest.main()

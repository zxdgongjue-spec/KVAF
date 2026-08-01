#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "resolve_a5_urma_endpoints.py"
SPEC = importlib.util.spec_from_file_location(
    "resolve_a5_urma_endpoints", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
RESOLVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESOLVER
SPEC.loader.exec_module(RESOLVER)

HOST_EID = "00000000003f050000100000df080b00"
DEVICE_EID = "000000000000020000100000df00e100"
OTHER_EID = "000000000041050000100000df00f207"


def displayed(eid: str) -> str:
    return ":".join(eid[offset : offset + 4] for offset in range(0, 32, 4))


def hccn_table(*rows: tuple[str, int, str]) -> str:
    lines = [
        "+-----------------+-----------+-----------------------------------------+",
        "| Name            | EID Index | EID                                     |",
        "+-----------------+-----------+-----------------------------------------+",
    ]
    previous = None
    for device, index, eid in rows:
        name = device if device != previous else ""
        lines.append(f"| {name:<15} | {index:<9} | {displayed(eid):<39} |")
        lines.append(
            "+-----------------+-----------+-----------------------------------------+"
        )
        previous = device
    return "\n".join(lines) + "\n"


class InventoryParseTests(unittest.TestCase):
    def test_npu7_device_endpoint_is_exact_d0e2_match(self) -> None:
        snapshot = hccn_table(
            ("udmac0d0e2", 0, DEVICE_EID),
            ("udmac0d1e5", 0, OTHER_EID),
            ("udmac0d1e5", 2, "000000000047050000100000df00f807"),
        )
        inventory = RESOLVER.parse_hccn_inventory(snapshot)
        endpoint = RESOLVER.choose_exact_endpoint(
            inventory, DEVICE_EID, "Device"
        )
        self.assertEqual(endpoint.device, "udmac0d0e2")
        self.assertEqual(endpoint.eid_index, 0)

    def test_duplicate_eid_is_rejected(self) -> None:
        snapshot = hccn_table(
            ("udmac0d0e2", 0, DEVICE_EID),
            ("udmac0d1e2", 1, DEVICE_EID),
        )
        with self.assertRaisesRegex(
            RESOLVER.EndpointResolutionError, "duplicate Device EID"
        ):
            RESOLVER.parse_hccn_inventory(snapshot)

    def test_missing_closing_border_is_rejected_as_truncated(self) -> None:
        snapshot = hccn_table(("udmac0d0e2", 0, DEVICE_EID))
        snapshot = snapshot.rsplit("\n", 2)[0] + "\n"
        with self.assertRaisesRegex(
            RESOLVER.EndpointResolutionError, "truncated|closing border"
        ):
            RESOLVER.parse_hccn_inventory(snapshot)

    def test_duplicate_table_header_is_rejected(self) -> None:
        snapshot = hccn_table(("udmac0d0e2", 0, DEVICE_EID)) * 2
        with self.assertRaisesRegex(
            RESOLVER.EndpointResolutionError, "exactly one EID table header"
        ):
            RESOLVER.parse_hccn_inventory(snapshot)

    def test_exact_lookup_rejects_missing_target(self) -> None:
        inventory = RESOLVER.parse_hccn_inventory(
            hccn_table(("udmac0d1e5", 0, OTHER_EID))
        )
        with self.assertRaisesRegex(
            RESOLVER.EndpointResolutionError, "found 0"
        ):
            RESOLVER.choose_exact_endpoint(inventory, DEVICE_EID, "Device")

    def test_host_sysfs_exact_match_and_duplicate_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "udmac0d1e5" / "eids"
            first.mkdir(parents=True)
            (first / "eid0").write_text(displayed(HOST_EID), encoding="utf-8")
            (first / "eid1").write_text("0" * 32, encoding="utf-8")
            inventory = RESOLVER.read_host_sysfs_inventory(root)
            endpoint = RESOLVER.choose_exact_endpoint(
                inventory, HOST_EID, "Host"
            )
            self.assertEqual(
                (endpoint.device, endpoint.eid_index), ("udmac0d1e5", 0)
            )

            second = root / "udmac1d1e5" / "eids"
            second.mkdir(parents=True)
            (second / "eid3").write_text(displayed(HOST_EID), encoding="utf-8")
            duplicate_inventory = RESOLVER.read_host_sysfs_inventory(root)
            with self.assertRaisesRegex(
                RESOLVER.EndpointResolutionError, "found 2"
            ):
                RESOLVER.choose_exact_endpoint(
                    duplicate_inventory, HOST_EID, "Host"
                )


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        eids = self.root / "sysfs" / "udmac0d1e5" / "eids"
        eids.mkdir(parents=True)
        (eids / "eid0").write_text(displayed(HOST_EID), encoding="utf-8")
        self.tool = self.root / "hccn_tool"
        self.tool.write_text(
            "#!/bin/sh\n"
            "test \"$1 $2 $3 $4\" = \"-g -dev_info -i 7\" || exit 9\n"
            "printf '%s' \"$KVAF_TEST_HCCN_OUTPUT\"\n",
            encoding="utf-8",
        )
        self.tool.chmod(0o755)

    def run_resolver(self, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["KVAF_TEST_HCCN_OUTPUT"] = hccn_table(
            ("udmac0d0e2", 0, DEVICE_EID),
            ("udmac0d1e5", 0, OTHER_EID),
        )
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--physical-device-id",
                "7",
                "--host-eid",
                HOST_EID,
                "--device-eid",
                DEVICE_EID,
                "--ubcore-root",
                str(self.root / "sysfs"),
                "--hccn-tool",
                str(self.tool),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_shell_output_contains_only_fixed_assignments(self) -> None:
        result = self.run_resolver("--shell")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "KVAF_HOST_URMA_DEVICE=udmac0d1e5",
                "KVAF_HOST_EID_INDEX=0",
                "KVAF_DEVICE_URMA_DEVICE=udmac0d0e2",
                "KVAF_DEVICE_EID_INDEX=0",
            ],
        )

    def test_json_output_names_both_endpoints(self) -> None:
        result = self.run_resolver()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "device_eid_index": 0,
                "device_urma_device": "udmac0d0e2",
                "host_eid_index": 0,
                "host_urma_device": "udmac0d1e5",
            },
        )

    def test_bounded_capture_rejects_oversized_output(self) -> None:
        result = self.run_resolver("--max-hccn-output-bytes", "16")
        self.assertEqual(result.returncode, 2)
        self.assertIn("bounded capture size", result.stderr)


if __name__ == "__main__":
    unittest.main()

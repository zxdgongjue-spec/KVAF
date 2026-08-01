#!/usr/bin/env python3

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "resolve_a5_urma_eids.py"
RUNNER = ROOT / "scripts" / "run_a5_aiv_urma_h2d_smoke.sh"
SPEC = importlib.util.spec_from_file_location("resolve_a5_urma_eids", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RESOLVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESOLVER
SPEC.loader.exec_module(RESOLVER)

HOST_EID_0 = "00112233445566778899aabbccddeeff"
DEVICE_EID_0 = "ffeeddccbbaa99887766554433221100"
HOST_EID_1 = "102132435465768798a9bacbdcedfe0f"
DEVICE_EID_1 = "f0e0d0c0b0a090807060504030201000"


def channel_block(
    channel: int,
    *,
    flag: int = 1,
    hop: int = 0,
    host_eid: str | None = HOST_EID_0,
    device_eid: str | None = DEVICE_EID_0,
) -> str:
    lines = [f"chan_id = {channel}; flag = {flag}; hop = {hop}"]
    if host_eid is not None:
        lines.append(f"local_eid: {host_eid}")
    if device_eid is not None:
        lines.append(f"remote_eid: {device_eid}")
    return "\n".join(lines)


def pair_info(
    *blocks: str,
    device_id: int = 0,
    slot_id: int = 3,
    module_id: int = 7,
    declared_channels: int | None = None,
) -> str:
    count = len(blocks) if declared_channels is None else declared_channels
    header = (
        "Dev id info. "
        f"(dev_id = {device_id}; slot_id = {slot_id}; "
        f"module_id = {module_id}; chan_num = {count})"
    )
    return "\n".join((header, *blocks, ""))


class ParseTests(unittest.TestCase):
    def test_normalize_eid(self) -> None:
        displayed = "0011:2233:4455:6677:8899:AABB:CCDD:EEFF"
        self.assertEqual(RESOLVER.normalize_eid(displayed), HOST_EID_0)
        with self.assertRaises(RESOLVER.EidResolutionError):
            RESOLVER.normalize_eid("0" * 32)
        with self.assertRaises(RESOLVER.EidResolutionError):
            RESOLVER.normalize_eid("not-an-eid")

    def test_single_active_pair_preserves_roles(self) -> None:
        snapshot = pair_info(
            channel_block(0, flag=0, host_eid=None, device_eid=None),
            channel_block(4, hop=2),
        )
        pairs = RESOLVER.parse_active_pairs(snapshot, 0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].channel, 4)
        self.assertEqual(pairs[0].host_eid, HOST_EID_0)
        self.assertEqual(pairs[0].device_eid, DEVICE_EID_0)

    def test_explicit_channel_disambiguates_multiple_active_pairs(self) -> None:
        snapshot = pair_info(
            channel_block(2),
            channel_block(
                5, host_eid=HOST_EID_1, device_eid=DEVICE_EID_1
            ),
        )
        pairs = RESOLVER.parse_active_pairs(snapshot, 0)
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"channels=2,5"
        ):
            RESOLVER.choose_pair(pairs, None)
        self.assertEqual(RESOLVER.choose_pair(pairs, 5).host_eid, HOST_EID_1)
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"active channels=2,5"
        ):
            RESOLVER.choose_pair(pairs, 9)

    def test_active_block_cannot_borrow_eid_from_next_block(self) -> None:
        snapshot = pair_info(
            channel_block(1, device_eid=None),
            channel_block(
                2, flag=0, host_eid=HOST_EID_1, device_eid=DEVICE_EID_1
            ),
        )
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"active channel 1"
        ):
            RESOLVER.parse_active_pairs(snapshot, 0)

    def test_stale_and_malformed_snapshots_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"stale pair_info"
        ):
            RESOLVER.parse_active_pairs(
                pair_info(channel_block(0), device_id=3), 0
            )
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"declares chan_num=2"
        ):
            RESOLVER.parse_active_pairs(
                pair_info(channel_block(0), declared_channels=2), 0
            )
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"duplicate chan_id=0"
        ):
            RESOLVER.parse_active_pairs(
                pair_info(channel_block(0), channel_block(0)), 0
            )

    def test_identical_endpoint_eids_are_rejected(self) -> None:
        snapshot = pair_info(
            channel_block(0, host_eid=HOST_EID_0, device_eid=HOST_EID_0)
        )
        with self.assertRaisesRegex(
            RESOLVER.EidResolutionError, r"identical Host and Device"
        ):
            RESOLVER.parse_active_pairs(snapshot, 0)

    def test_transient_invalid_snapshot_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            snapshot = pair_info(channel_block(4))
            (proc_root / "pair_info").write_text(
                snapshot, encoding="utf-8"
            )
            parsed = RESOLVER.parse_active_pairs(snapshot, 0)
            transient = RESOLVER.EidResolutionError("transient snapshot")
            with mock.patch.object(
                RESOLVER,
                "parse_active_pairs",
                side_effect=(transient, parsed),
            ) as parser:
                selected = RESOLVER.read_driver_pair(
                    proc_root,
                    0,
                    None,
                    root / "selector.lock",
                    1.0,
                    root / "legacy.lock",
                )
            self.assertEqual(selected.channel, 4)
            self.assertEqual(parser.call_count, 2)


class CommandLineTests(unittest.TestCase):
    def run_resolver(
        self,
        proc_root: Path,
        lock_file: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--physical-device-id",
                "0",
                "--proc-root",
                str(proc_root),
                "--lock-file",
                str(lock_file),
                "--legacy-lock-file",
                str(lock_file.with_name("legacy-selector.lock")),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_shell_output_contains_only_fixed_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(channel_block(6, hop=3)), encoding="utf-8"
            )
            result = self.run_resolver(
                proc_root, root / "selector.lock", "--shell"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(
                result.stdout.splitlines(),
                [
                    f"KVAF_HOST_EID={HOST_EID_0}",
                    f"KVAF_DEVICE_EID={DEVICE_EID_0}",
                    "KVAF_PAIR_CHANNEL=6",
                ],
            )
            self.assertEqual(
                (proc_root / "dev_id").read_text(encoding="utf-8"), "0\n"
            )

    def test_default_output_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(channel_block(1)), encoding="utf-8"
            )
            result = self.run_resolver(proc_root, root / "selector.lock")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["channel"], 1)
            self.assertEqual(payload["host_eid"], HOST_EID_0)
            self.assertEqual(payload["device_eid"], DEVICE_EID_0)

    def test_multiple_active_channels_fail_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(
                    channel_block(2),
                    channel_block(
                        8, host_eid=HOST_EID_1, device_eid=DEVICE_EID_1
                    ),
                ),
                encoding="utf-8",
            )
            result = self.run_resolver(
                proc_root, root / "selector.lock", "--shell"
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("channels=2,8", result.stderr)
            self.assertIn("KVAF_PAIR_CHANNEL", result.stderr)

    def test_explicit_channel_selects_requested_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(
                    channel_block(2),
                    channel_block(
                        8, host_eid=HOST_EID_1, device_eid=DEVICE_EID_1
                    ),
                ),
                encoding="utf-8",
            )
            result = self.run_resolver(
                proc_root,
                root / "selector.lock",
                "--pair-channel",
                "8",
                "--shell",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.splitlines(),
                [
                    f"KVAF_HOST_EID={HOST_EID_1}",
                    f"KVAF_DEVICE_EID={DEVICE_EID_1}",
                    "KVAF_PAIR_CHANNEL=8",
                ],
            )

    def test_lock_timeout_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(channel_block(0)), encoding="utf-8"
            )
            lock_path = root / "selector.lock"
            with lock_path.open("a+", encoding="utf-8") as held_lock:
                fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX)
                result = self.run_resolver(
                    proc_root,
                    lock_path,
                    "--lock-timeout-seconds",
                    "0.05",
                )
            self.assertEqual(result.returncode, 2)
            self.assertIn("timed out", result.stderr)

    def test_hard_linked_lock_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(channel_block(0)), encoding="utf-8"
            )
            lock_path = root / "selector.lock"
            lock_path.write_text("", encoding="utf-8")
            os.link(lock_path, root / "selector.hardlink")
            result = self.run_resolver(proc_root, lock_path)
            self.assertEqual(result.returncode, 2)
            self.assertIn("exactly one hard link", result.stderr)


class LockTopologyTests(unittest.TestCase):
    def test_default_lock_is_shared_beside_the_physical_repository(self) -> None:
        self.assertEqual(
            RESOLVER.DEFAULT_LOCK_FILE,
            Path("/data/.kvaf-state/asdrv-ub-pair.lock"),
        )

    def test_persistent_lock_is_held_before_legacy_lock(self) -> None:
        persistent = Path("/persistent/selector.lock")
        legacy = Path("/legacy/selector.lock")
        events: list[str] = []

        @contextmanager
        def fake_lock(path: Path, _timeout: float):
            events.append(f"enter:{path}")
            try:
                yield object()
            finally:
                events.append(f"exit:{path}")

        with mock.patch.object(RESOLVER, "acquire_lock", fake_lock):
            with RESOLVER.acquire_selector_locks(
                persistent, 1.0, legacy
            ):
                events.append("critical")

        self.assertEqual(
            events,
            [
                f"enter:{persistent}",
                f"enter:{legacy}",
                "critical",
                f"exit:{legacy}",
                f"exit:{persistent}",
            ],
        )

    def test_lock_parent_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(
                RESOLVER.EidResolutionError, "not a real directory"
            ):
                RESOLVER.acquire_lock(linked_parent / "selector.lock", 0.1)


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_auto_resolves_pair_before_binary_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            proc_root.mkdir()
            (proc_root / "dev_id").write_text("", encoding="utf-8")
            (proc_root / "pair_info").write_text(
                pair_info(channel_block(3)), encoding="utf-8"
            )
            environment = os.environ.copy()
            environment.pop("KVAF_HOST_EID", None)
            environment.pop("KVAF_DEVICE_EID", None)
            environment.pop("KVAF_PAIR_CHANNEL", None)
            environment.update(
                {
                    "KVAF_LOGICAL_DEVICE_ID": "0",
                    "KVAF_PHYSICAL_DEVICE_ID": "0",
                    "KVAF_SUPER_DEVICE_ID": "0",
                    "KVAF_SERVER_INDEX": "0",
                    "KVAF_SUPER_POD_INDEX": "4294967295",
                    "KVAF_EID_PROC_ROOT": str(proc_root),
                    "KVAF_EID_LOCK_FILE": str(root / "selector.lock"),
                    "KVAF_EID_LEGACY_LOCK_FILE": str(
                        root / "legacy-selector.lock"
                    ),
                    "KVAF_A5_BINARY": str(root / "missing-smoke-binary"),
                }
            )
            result = subprocess.run(
                ["bash", str(RUNNER)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "KVAF_A5_EID_RESOLVED "
                "source=driver_pair_info physical_device=0 channel=3",
                result.stdout,
            )
            self.assertIn("smoke binary is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()

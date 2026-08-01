#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "durable_state_file.py"
SPEC = importlib.util.spec_from_file_location("durable_state_file", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DURABLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DURABLE
SPEC.loader.exec_module(DURABLE)


class DurableStateFileTests(unittest.TestCase):
    def test_publish_orders_file_fsync_before_replace_and_directory_fsync(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "state.tmp"
            target = root / "state"
            source.write_text("new\n", encoding="utf-8")
            target.write_text("old\n", encoding="utf-8")

            events: list[str] = []
            real_fsync = os.fsync
            real_replace = os.replace

            def traced_fsync(descriptor: int) -> None:
                metadata = os.fstat(descriptor)
                events.append(
                    "fsync_dir"
                    if stat.S_ISDIR(metadata.st_mode)
                    else "fsync_file"
                )
                real_fsync(descriptor)

            def traced_replace(*args, **kwargs) -> None:
                events.append("replace")
                real_replace(*args, **kwargs)

            with mock.patch.object(DURABLE.os, "fsync", traced_fsync), \
                 mock.patch.object(DURABLE.os, "replace", traced_replace):
                DURABLE.durable_publish(source, target)

            self.assertEqual(
                events, ["fsync_file", "replace", "fsync_dir"]
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
            self.assertFalse(source.exists())

    def test_publish_rejects_symlink_and_hard_link_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.write_text("state\n", encoding="utf-8")
            symlink = root / "state.tmp"
            symlink.symlink_to(real)
            with self.assertRaises(DURABLE.StateFileError):
                DURABLE.durable_publish(symlink, root / "state")

            source = root / "fresh.tmp"
            source.write_text("fresh\n", encoding="utf-8")
            os.link(source, root / "fresh.hardlink")
            with self.assertRaises(DURABLE.StateFileError):
                DURABLE.durable_publish(source, root / "state")

    def test_publish_failure_before_replace_preserves_old_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "state.tmp"
            target = root / "state"
            source.write_text("new\n", encoding="utf-8")
            target.write_text("old\n", encoding="utf-8")

            with mock.patch.object(
                DURABLE.os, "fsync", side_effect=OSError("injected fsync")
            ):
                with self.assertRaises(OSError):
                    DURABLE.durable_publish(source, target)

            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")
            self.assertTrue(source.exists())

    def test_remove_and_sync_are_idempotent_and_no_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "state"
            target.write_text("value\n", encoding="utf-8")
            DURABLE.durable_sync((target, root))
            DURABLE.durable_remove(target)
            DURABLE.durable_remove(target)
            self.assertFalse(target.exists())

            real = root / "real"
            real.write_text("value\n", encoding="utf-8")
            symlink = root / "link"
            symlink.symlink_to(real)
            with self.assertRaises(DURABLE.StateFileError):
                DURABLE.durable_sync((symlink,))


if __name__ == "__main__":
    unittest.main()

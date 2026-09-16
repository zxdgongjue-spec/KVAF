"""Portable fake-tool tests for the read-only URMA snapshot collector."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "collect_urma_dfx", Path(__file__).resolve().parents[1] / "collect_urma_dfx.py")
dfx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dfx)


class UrmaDfxTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands.jsonl"
        fake = self.bin / "urma_admin"
        fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys, time
args = sys.argv[1:]
with open(os.environ["DFX_TEST_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[-1] == os.environ.get("DFX_TEST_TIMEOUT"):
    print("partial output", flush=True)
    time.sleep(60)
if args[-1] == os.environ.get("DFX_TEST_FAILURE"):
    print("unsupported resource", file=sys.stderr)
    sys.exit(17)
print("fake read-only output")
''')
        fake.chmod(0o755)
        self.proc = self.root / "proc"
        (self.proc / "123").mkdir(parents=True)
        (self.proc / "modules").write_text("ubcore 1 0 - Live 0x0\n")
        (self.proc / "123" / "maps").write_text(
            "000 /lib/liburma.so\n001 /lib/libra.so\n002 /lib/libhccp.so\n"
            "003 /lib/libudma.so\n004 /vendor/urma/libvendor.so\n005 /unrelated/private.bin\n")
        (self.proc / "123" / "status").write_text("Name:\tfake-worker\n")
        (self.proc / "123" / "numa_maps").write_text("000 default anon=1 N0=1\n")
        self.sysfs = self.root / "sys"
        device = self.sysfs / "class" / "ubcore" / "dev0"
        device.mkdir(parents=True)
        (device / "numa_node").write_text("0\n")
        uburma = self.sysfs / "class" / "uburma" / "urma0"
        uburma.mkdir(parents=True)
        (uburma / "ubdev_name").write_text("dev0\n")
        (uburma / "dangerous_unknown_register").write_text("must never be read\n")
        self.environ = patch.dict(os.environ, {
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            "DFX_TEST_LOG": str(self.log), "DFX_TEST_FAILURE": "", "DFX_TEST_TIMEOUT": "",
        })
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def collect(self, *options):
        args = dfx.parse_args(["--output-dir", str(self.root / "capture"),
                               "--device", "dev0", "--pid", "123", *options])
        snapshot, code = dfx.collect(args, self.proc, self.sysfs)
        return snapshot, code

    def test_only_expected_read_commands_and_files(self):
        snapshot, code = self.collect()
        self.assertEqual(code, 0)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(commands, [
            ["show", "dev", "dev0", kind] for kind in ("jetty", "jfr", "jfc", "tp", "tpreuse")
        ] + [["--version"], ["--help"], ["show", "--all", "--whole"],
             ["show", "topo"], ["show_res", "--help"], ["show_stats", "--help"]])
        manifest = json.loads((snapshot.output_dir / "manifest.json").read_text())
        self.assertFalse(manifest["hardware_stall_counters_collected"])
        rows = manifest["entries"]
        self.assertTrue(rows[0]["source"].endswith("/123/maps"))
        maps = (snapshot.output_dir / rows[0]["output_file"]).read_text()
        self.assertIn("liburma.so", maps)
        self.assertIn("libra.so", maps)
        self.assertIn("libhccp.so", maps)
        self.assertIn("libudma.so", maps)
        self.assertIn("/vendor/urma/libvendor.so", maps)
        self.assertNotIn("private.bin", maps)
        sources = [row.get("source", "") for row in rows]
        self.assertFalse(any("dangerous_unknown" in source for source in sources))
        for row in rows:
            self.assertLessEqual(row["start"]["monotonic_ns"], row["end"]["monotonic_ns"])
            self.assertTrue(row["start"]["utc"].endswith("+00:00"))
            self.assertTrue((snapshot.output_dir / row["output_file"]).exists())

    def test_failures_missing_resources_and_permission_are_recorded(self):
        (self.proc / "123" / "numa_maps").unlink()
        original = Path.open

        def selective_permission(source, *args, **kwargs):
            if source == self.proc / "123" / "status":
                raise PermissionError("denied by test")
            return original(source, *args, **kwargs)

        with patch.dict(os.environ, {"DFX_TEST_FAILURE": "jfc"}), patch.object(Path, "open", selective_permission):
            snapshot, code = self.collect()
        self.assertEqual(code, 1)
        rows = snapshot.manifest["entries"]
        self.assertEqual(rows[1]["status"], "permission_denied")
        self.assertEqual(rows[2]["status"], "missing")
        failed = next(row for row in rows if row.get("argv", [])[-1:] == ["jfc"])
        self.assertEqual(failed["exitcode"], 17)
        self.assertEqual(failed["status"], "nonzero_exit")
        self.assertIn("unsupported resource", (snapshot.output_dir / failed["output_file"]).read_text())
        self.assertEqual(snapshot.manifest["status"], "complete_with_errors")

    def test_command_timeout_retains_partial_output_and_continues(self):
        with patch.dict(os.environ, {"DFX_TEST_TIMEOUT": "jetty"}):
            snapshot, code = self.collect("--command-timeout", "1")
        self.assertEqual(code, 1)
        timed_out = [row for row in snapshot.manifest["entries"] if row["status"] == "timeout"]
        self.assertEqual(len(timed_out), 1)
        self.assertIsNone(timed_out[0]["exitcode"])
        self.assertIn("partial output", (snapshot.output_dir / timed_out[0]["output_file"]).read_text())
        self.assertTrue(any(row.get("argv") == ["urma_admin", "show", "topo"]
                            and row["status"] == "ok" for row in snapshot.manifest["entries"]))

    def test_missing_executable_is_not_silently_skipped(self):
        (self.bin / "urma_admin").unlink()
        with patch.dict(os.environ, {"PATH": str(self.bin)}):
            snapshot, code = self.collect()
        self.assertEqual(code, 1)
        commands = [row for row in snapshot.manifest["entries"] if row["kind"] == "command"]
        self.assertTrue(all(row["status"] == "missing" for row in commands))

    def test_reject_existing_output_and_unsafe_arguments(self):
        output = self.root / "capture"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual(sentinel.read_text(), "keep")
        for options in (("--device", "../dev0"), ("--device=--all",),
                        ("--pid", "0"), ("--command-timeout", "nan"),
                        ("--command-timeout", "0")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    dfx.parse_args(["--output-dir", str(output), *options])


if __name__ == "__main__":
    unittest.main()

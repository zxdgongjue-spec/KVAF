#!/usr/bin/env python3
"""Collect read-only URMA resource/environment snapshots, not hardware stall counters.

Run while the CPU service and NPU workers are alive. Unsupported commands and
unavailable files are preserved as failures in manifest.json; no sudo, device
configuration changes, counter resets or profiling modes are used.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import stat
import subprocess
import time


LIBRARY_PATTERN = re.compile(
    r"liburma|libuburma|libudma|/urma/|provider|libascendcl|libhccp|libra(?:[._-]|$)",
    re.IGNORECASE)
SYSFS_ATTRIBUTES = (
    "numa_node", "ue_id", "guid", "ubdev", "ubdev_name",
    "device/numa_node", "device/vendor", "device/device", "device/revision",
    "device/class", "device/ue_id",
)
SYSFS_LINKS = ("device", "driver", "device/driver", "subsystem")
MAX_FILE_BYTES = 16 * 1024 * 1024


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns()}


def error_status(error):
    if isinstance(error, FileNotFoundError):
        return "missing"
    if isinstance(error, PermissionError):
        return "permission_denied"
    return "error"


def text_output(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")


class Snapshot:
    def __init__(self, output_dir, timeout):
        self.output_dir = Path(output_dir)
        # Never mix a new capture with old output, including an existing empty directory.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.timeout = timeout
        self.manifest = dict(
            schema_version=1, scope="read_only_resource_and_environment_snapshot",
            hardware_stall_counters_collected=False, start=stamp(),
            command_timeout_seconds=timeout, status="collecting", entries=[])
        self.save()

    def save(self):
        (self.output_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def record(self, row, contents, suffix="txt"):
        filename = f"{len(self.manifest['entries']):03d}.{suffix}"
        (self.output_dir / filename).write_text(contents, encoding="utf-8")
        row.update(end=stamp(), output_file=filename)
        self.manifest["entries"].append(row)
        self.save()

    def command(self, argv):
        row = dict(kind="command", argv=argv, start=stamp(), exitcode=None)
        try:
            result = subprocess.run(argv, capture_output=True, timeout=self.timeout, check=False)
            row.update(exitcode=result.returncode, status="ok" if result.returncode == 0 else "nonzero_exit")
            stdout, stderr = text_output(result.stdout), text_output(result.stderr)
        except subprocess.TimeoutExpired as error:
            row.update(status="timeout", error=str(error))
            stdout, stderr = text_output(error.stdout), text_output(error.stderr)
        except OSError as error:
            row.update(status=error_status(error), error=str(error))
            stdout, stderr = "", str(error)
        self.record(row, json.dumps(dict(stdout=stdout, stderr=stderr), indent=2) + "\n", "json")

    def file(self, source, filter_libraries=False):
        row = dict(kind="file", source=str(source), start=stamp())
        try:
            with source.open("rb") as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
            truncated = len(data) > MAX_FILE_BYTES
            contents = data[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
            if filter_libraries:
                contents = "".join(line for line in contents.splitlines(keepends=True)
                                   if LIBRARY_PATTERN.search(line))
                row["filter"] = LIBRARY_PATTERN.pattern
            row.update(status="truncated" if truncated else "ok", max_bytes=MAX_FILE_BYTES)
        except OSError as error:
            row.update(status=error_status(error), error=str(error))
            contents = str(error) + "\n"
        self.record(row, contents)

    def sysfs_node(self, source):
        row = dict(kind="sysfs_node", source=str(source), start=stamp())
        details = {}
        attributes = []
        try:
            details["realpath"] = str(source.resolve(strict=True))
            details["links"] = {}
            for relative in SYSFS_LINKS:
                link = source / relative
                if link.is_symlink():
                    details["links"][relative] = str(link.resolve(strict=False))
            row["status"] = "ok"
            details["attributes"] = {}
            for relative in SYSFS_ATTRIBUTES:
                attribute = source / relative
                try:
                    mode = attribute.stat().st_mode
                    details["attributes"][relative] = "file" if stat.S_ISREG(mode) else "not_a_file"
                    if stat.S_ISREG(mode):
                        attributes.append(attribute)
                except FileNotFoundError:
                    details["attributes"][relative] = "absent"
                except OSError as error:
                    details["attributes"][relative] = dict(status=error_status(error), error=str(error))
                    row["status"] = error_status(error)
        except OSError as error:
            row.update(status=error_status(error), error=str(error))
        self.record(row, json.dumps(details, indent=2) + "\n", "json")
        # Optional attributes vary across providers. Never recurse through sysfs
        # or read arbitrary driver files.
        for attribute in attributes:
            self.file(attribute)

    def sysfs_class(self, source):
        row = dict(kind="sysfs_class", source=str(source), start=stamp())
        children = []
        try:
            children = sorted(source.iterdir())
            row["status"] = "ok"
        except OSError as error:
            row.update(status=error_status(error), error=str(error))
        self.record(row, json.dumps([child.name for child in children], indent=2) + "\n", "json")
        for child in children:
            self.sysfs_node(child)

    def finish(self):
        failures = sum(row["status"] != "ok" for row in self.manifest["entries"])
        self.manifest.update(end=stamp(), failed_entries=failures,
                             status="complete_with_errors" if failures else "complete")
        self.save()
        return 1 if failures else 0


def collect(args, proc_root=Path("/proc"), sys_root=Path("/sys")):
    snapshot = Snapshot(args.output_dir, args.command_timeout)
    snapshot.manifest.update(devices=args.device, pids=args.pid)
    snapshot.save()
    # Dynamic resources can disappear when a worker exits: capture these first.
    for pid in args.pid:
        root = proc_root / str(pid)
        for filename in ("maps", "status", "numa_maps"):
            snapshot.file(root / filename, filter_libraries=filename == "maps")
    for device in args.device:
        for resource in ("jetty", "jfr", "jfc", "tp", "tpreuse"):
            snapshot.command(["urma_admin", "show", "dev", device, resource])
        snapshot.sysfs_node(sys_root / "class" / "ubcore" / device)
    for argv in (
        ["uname", "-a"], ["urma_admin", "--version"], ["urma_admin", "--help"],
        ["urma_admin", "show", "--all", "--whole"], ["urma_admin", "show", "topo"],
        ["urma_admin", "show_res", "--help"], ["urma_admin", "show_stats", "--help"],
    ):
        snapshot.command(argv)
    snapshot.file(proc_root / "modules")
    snapshot.sysfs_class(sys_root / "class" / "uburma")
    return snapshot, snapshot.finish()


def device_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*", value) or value in (".", ".."):
        raise argparse.ArgumentTypeError("device must be a single device name, not an option or path")
    return value


def positive_pid(value):
    try:
        pid = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("pid must be a positive integer") from None
    if pid <= 0:
        raise argparse.ArgumentTypeError("pid must be a positive integer")
    return pid


def positive_timeout(value):
    try:
        timeout = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("command timeout must be a positive finite number") from None
    if not 0 < timeout < float("inf"):
        raise argparse.ArgumentTypeError("command timeout must be a positive finite number")
    return timeout


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory; existing paths are rejected")
    parser.add_argument("--device", action="append", type=device_name, default=[], help="URMA device name (repeatable)")
    parser.add_argument("--pid", action="append", type=positive_pid, default=[], help="live service/worker PID (repeatable)")
    parser.add_argument("--command-timeout", type=positive_timeout, default=5.0, help="timeout in seconds for each external command (default: 5)")
    args = parser.parse_args(argv)
    args.device = list(dict.fromkeys(args.device))
    args.pid = list(dict.fromkeys(args.pid))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        snapshot, exitcode = collect(args)
    except OSError as error:
        print(f"Snapshot failed: {error}")
        return 2
    print(f"{snapshot.manifest['status']}: {args.output_dir / 'manifest.json'} "
          f"({snapshot.manifest['failed_entries']} unavailable/failed entries)")
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())

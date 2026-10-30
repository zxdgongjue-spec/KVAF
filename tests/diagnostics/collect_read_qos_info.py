#!/usr/bin/env python3
"""Collect CPU-side Linux URMA/UBASE information and QoS configuration.

No QoS writes, counter resets, driver reloads, debugfs mounts or traffic generation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


QOS_FILES = (
    "adev_qos", "sl_vl_map", "ets_tc", "ets_tcg", "ets_port",
    "tm_queue", "tm_qset", "tm_pri", "tm_pg", "tm_port",
)
SOURCE_NAMES = {
    "udma_jetty.c", "udma_jfs.c", "udma_dfx.c", "udma_u_jetty.c",
    "ubase_qos_hw.c", "ubase_qos_debugfs.c", "ubase_debugfs.c",
    "ubase_stats.c", "ubase_tp.c", "udma_ctrlq_tp.c",
    "urma_types.h", "ubcore_types.h", "ubase_comm_dev.h",
    "admin_parameters.h", "admin_cmd_show_res.c",
}
SOURCE_PATTERN = re.compile(
    r"priority|\bsl\b|resp.*vl|req.*vl|perf_stats|tx_vl_bw|rx_vl_bw|tp_type", re.I
)


def normalize_eid(value):
    value = value.strip().replace(":", "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("EID must be exactly 32 hexadecimal digits (colons allowed)")
    return value


def parse_devices(text):
    """Parse the installed urma_admin show --all --whole format, not device names."""
    devices, current = {}, None
    for line in text.splitlines():
        match = re.match(r"\s*name\s*:\s*(\S+)\s*$", line)
        if match:
            current = match[1]
            devices.setdefault(current, [])
        match = re.match(r"\s*eid\d+\s*:\s*([0-9a-fA-F:]+)\s*$", line)
        if match and current:
            try:
                devices[current].append(normalize_eid(match[1]))
            except ValueError:
                pass
    return devices


def select_device(devices, eid, requested=None):
    matches = [name for name, eids in devices.items() if eid in eids]
    if requested:
        if requested not in matches:
            raise ValueError(f"{requested} is not verified to own EID {eid}; "
                             f"device_eids={devices.get(requested, [])}; matches={matches}")
        return requested
    if len(matches) != 1:
        raise ValueError(f"Need exactly one local EID match; found {matches}")
    return matches[0]


def resolve_debug_dir(device_path, debug_root):
    """Use the device's immediate UB controller, never an arbitrary ancestor."""
    resolved = device_path.resolve(strict=True)
    if resolved.parent.name != "ubcore":
        raise ValueError(f"Unrecognized ubcore sysfs layout: {resolved}")
    candidate = debug_root / resolved.parent.parent.name
    if not candidate.is_dir():
        raise ValueError(f"Controller debugfs unavailable: {candidate}")
    return candidate


def configured_response_pairs(text):
    req = re.search(r"^ctp_req_vl:\s*([^\n]*)", text, re.M)
    offset = re.search(r"^ctp_resp_vl_offset:\s*(\d+)\s*$", text, re.M)
    if not req or not offset:
        return None
    values = req[1].split()
    if not values or not all(v.isdecimal() for v in values):
        return None
    return [[int(v), int(v) + int(offset[1])] for v in values]


class Collector:
    def __init__(self, output, timeout):
        self.output, self.timeout, self.records = output, timeout, []
        output.mkdir(parents=True, exist_ok=False)

    def run(self, name, argv, save=True):
        start = time.time()
        try:
            proc = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=self.timeout,
                                  check=False)
            data, status = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as exc:
            data, status = (exc.stdout or b"") + b"\nTIMEOUT\n", "timeout"
        except OSError as exc:
            data, status = str(exc).encode(), "unavailable"
        limit = 2 * 1024 * 1024
        truncated = len(data) > limit
        text = data[:limit].decode("utf-8", errors="replace")
        if save:
            (self.output / (name + ".txt")).write_text(text, encoding="utf-8")
        self.records.append(dict(name=name, argv=[str(x) for x in argv], status=status,
                                 start_unix=start, end_unix=time.time(), truncated=truncated))
        return status, text

    def read(self, name, path):
        # cat in a timed subprocess prevents a debugfs read from blocking Python forever.
        return self.run(name, ["cat", str(path)])


def collect_sources(collector, roots):
    """Bounded source excerpts. Supplied source is not automatically the running build."""
    collected, seen = [], set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            collected.append(dict(root=str(root), error="not a directory"))
            continue
        for parent, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in {".git", "node_modules", "__pycache__"})
            for name in sorted(names):
                if name not in SOURCE_NAMES:
                    continue
                path = Path(parent) / name
                if path in seen:
                    continue
                seen.add(path)
                if len(seen) > 150:
                    return collected + [dict(error="source file limit reached")]
                try:
                    if path.stat().st_size > 2 * 1024 * 1024:
                        raise ValueError("source file exceeds 2 MiB")
                    raw = path.read_bytes()
                    lines = raw.decode("utf-8", errors="replace").splitlines()
                    selected = set()
                    for index, line in enumerate(lines):
                        if SOURCE_PATTERN.search(line):
                            selected.update(range(max(0, index - 4), min(len(lines), index + 5)))
                    indices = sorted(selected)
                    target = f"source_{len(collected):03d}_{name}.txt"
                    excerpts = "\n".join(f"{i + 1}: {lines[i]}" for i in indices[:1800])
                    (collector.output / target).write_text(excerpts, encoding="utf-8")
                    collected.append(dict(path=str(path), file=target,
                                          sha256=hashlib.sha256(raw).hexdigest(),
                                          truncated=len(indices) > 1800,
                                          running_build_match="unverified"))
                except (OSError, ValueError) as exc:
                    collected.append(dict(path=str(path), error=str(exc)))
    return collected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid", required=True, help="Local CPU-side URMA EID")
    parser.add_argument("--urma-device", help="Optional CPU-side device name; its EID must match")
    parser.add_argument("--source-root", type=Path, action="append", default=[], help="Matching CPU-side kernel/driver/UMDK source tree")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args(argv)
    try:
        eid = normalize_eid(args.eid)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.timeout < float("inf"):
        parser.error("timeout must be finite and > 0")
    if sys.platform != "linux":
        parser.error("Collection runs on the Linux server; parsers can be tested on any OS")
    collector = Collector(args.output, args.timeout)
    report = dict(role="cpu", eid=eid, hostname=platform.node(),
                  kernel=platform.release(), warnings=[])
    collector.read("os_release", "/etc/os-release")
    collector.run("urma_version", ["urma_admin", "--version"])
    collector.run("urma_help", ["urma_admin", "--help"])
    status, device_text = collector.run("urma_devices", ["urma_admin", "show", "--all", "--whole"])
    devices = parse_devices(device_text) if status == 0 else {}
    report["local_device_eids"] = devices
    for module in ("ubase", "udma", "ubcore", "uburma"):
        collector.run("modinfo_" + module, ["modinfo", module])
        collector.read("loaded_" + module + "_srcversion", f"/sys/module/{module}/srcversion")
    debug = None
    stage = "device_identity"
    try:
        device = select_device(devices, eid, args.urma_device)
        report["local_device"] = device
        stage = "debugfs_resolution"
        device_path = Path("/sys/class/ubcore") / device
        report["sysfs_device"] = str(device_path.resolve())
        debug = resolve_debug_dir(device_path, Path("/sys/kernel/debug/ubase"))
        report["debugfs_device"] = str(debug)
        for name in ("caps_info", "all_adev_name"):
            collector.read(name, debug / name)
        for name in QOS_FILES:
            code, output = collector.read("qos_" + name, debug / "qos" / name)
            if name == "adev_qos" and code == 0:
                report["configured_ctp_req_resp_vl_pairs"] = configured_response_pairs(output)
    except (OSError, ValueError) as exc:
        report["local_device_error"] = dict(stage=stage, reason=str(exc))
        print(f"INFO_UNAVAILABLE stage={stage} reason={exc}", flush=True)
        if stage == "device_identity":
            print(f"DEVICE_DETAILS={args.output.resolve() / 'urma_devices.txt'}", flush=True)
        report["warnings"].append(str(exc))
        report["warnings"].append("No local debugfs association; local QoS information is unavailable.")
    debug_root = Path("/sys/kernel/debug/ubase")
    try:
        report["available_debug_controllers"] = sorted(p.name for p in debug_root.iterdir() if p.is_dir())
    except OSError as exc:
        report["warnings"].append(str(exc))
    report["source_excerpts"] = collect_sources(collector, args.source_root)
    report["warnings"] += [
        "Configured request/response VL pairs describe capabilities, not observed READ packet routing.",
        "CPU-side snapshots must be refreshed after device, kernel, driver, UMDK or QoS changes.",
    ]
    report["commands"] = collector.records
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={args.output.resolve() / 'report.json'}")
    return 0 if debug is not None else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        sys.exit(1)

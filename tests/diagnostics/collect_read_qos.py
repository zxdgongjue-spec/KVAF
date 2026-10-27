#!/usr/bin/env python3
"""Collect READ QoS evidence on Linux; never infer response SL from a Jetty priority.

No QoS writes, counter resets, driver reloads, debugfs mounts or traffic generation.
Optional --device-id invokes the existing HCCP query (creates a temporary context).
"""
import argparse
import csv
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
    r"priority|\bsl\b|\bSL\b|resp.*vl|req.*vl|perf_stats|tx_vl_bw|rx_vl_bw|"
    r"Ra\w*(?:Query|Get|Dfx|Stats)|tpType|tp_type", re.I
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
            raise ValueError(f"{requested} is not verified to own EID {eid}; matches={matches}")
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


def parse_perf_stats(text):
    """Only accept documented per-VL bandwidth rows; these are NOT counters."""
    rows, port, in_table = [], None, False
    for line in text.splitlines():
        match = re.fullmatch(r"\s*port_id:\s*(\d+)\s*", line)
        if match:
            port, in_table = int(match[1]), False
        elif re.fullmatch(r"\s*vl\s+tx_bw\(kbps\)\s+rx_bw\(kbps\)\s*", line):
            in_table = True
        elif in_table:
            match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\d+)\s*", line)
            if match and port is not None:
                rows.append(dict(port=port, vl=int(match[1]),
                                 tx_kbps=int(match[2]), rx_kbps=int(match[3])))
            else:
                in_table = False
    return rows


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

    def run(self, name, argv, timeout=None):
        start = time.time()
        try:
            proc = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=timeout or self.timeout,
                                  check=False)
            data, status = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as exc:
            data, status = (exc.stdout or b"") + b"\nTIMEOUT\n", "timeout"
        except OSError as exc:
            data, status = str(exc).encode(), "unavailable"
        limit = 2 * 1024 * 1024
        truncated = len(data) > limit
        text = data[:limit].decode("utf-8", errors="replace")
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
                if name not in SOURCE_NAMES and not (name.startswith("hccp") and name.endswith(".h")):
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
    parser.add_argument("--role", choices=("npu", "cpu"), required=True)
    parser.add_argument("--eid", required=True)
    parser.add_argument("--urma-device", help="Optional local device name; its EID must match")
    parser.add_argument("--device-id", type=int, help="NPU logical ID; enables the existing HCCP mapping query")
    parser.add_argument("--build-dir", type=Path, default=Path("build-npu"))
    parser.add_argument("--hccp-include", type=Path)
    parser.add_argument("--jetty-id", type=int, action="append", default=[], help="Live local Jetty ID; repeatable")
    parser.add_argument("--source-root", type=Path, action="append", default=[], help="Matching kernel/UMDK/SDK source tree")
    parser.add_argument("--samples", type=int, default=1, help="Number of per-VL perf_stats reads")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between reads (read itself also takes time)")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--label", default="snapshot", help="e.g. idle, p1-only, p3-only")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args(argv)
    try:
        eid = normalize_eid(args.eid)
    except ValueError as exc:
        parser.error(str(exc))
    if args.samples < 1 or args.interval < 0.1 or args.timeout <= 0 or any(i < 0 for i in args.jetty_id):
        parser.error("samples >= 1, interval >= 0.1, timeout > 0 and nonnegative Jetty IDs required")
    if args.device_id is not None and (args.role != "npu" or args.device_id < 0):
        parser.error("--device-id requires --role npu and a nonnegative logical ID")
    if sys.platform != "linux":
        parser.error("Collection runs on the Linux server; parsers can be tested on any OS")
    collector = Collector(args.output, args.timeout)
    report = dict(role=args.role, eid=eid, label=args.label, hostname=platform.node(),
                  kernel=platform.release(), warnings=[], request_packet_sl="unverified",
                  response_packet_sl="unverified", response_inherits_request_sl="unverified")
    collector.read("os_release", "/etc/os-release")
    collector.read("modules", "/proc/modules")
    collector.run("urma_version", ["urma_admin", "--version"])
    collector.run("urma_help", ["urma_admin", "--help"])
    status, device_text = collector.run("urma_devices", ["urma_admin", "show", "--all", "--whole"])
    devices = parse_devices(device_text) if status == 0 else {}
    for module in ("ubase", "udma", "ubcore", "uburma"):
        collector.run("modinfo_" + module, ["modinfo", module])
        collector.read("loaded_" + module + "_srcversion", f"/sys/module/{module}/srcversion")
    collector.run("npu_smi", ["npu-smi", "info"])
    collector.read("ascend_driver_version", "/usr/local/Ascend/driver/version.info")
    roots = list(args.source_root)
    if args.device_id is not None:
        helper = Path(__file__).with_name("query_npu_priority_sl.py")
        command = [sys.executable, helper, "--device", str(args.device_id), "--eid", eid,
                   "--build-dir", args.build_dir]
        if args.hccp_include:
            command += ["--hccp-include", args.hccp_include]
        code, output = collector.run("npu_hccp_mapping", command, timeout=180)
        report["hccp_query_status"] = code
        match = re.search(r"^HCCP_HEADER=(.+)$", output, re.M)
        if match:
            roots.append(Path(match[1].strip()).parent)
        for key in ("HCCP_LIBRARY", "KVAF_LIBRARY"):
            match = re.search(r"^" + key + r"=(.+)$", output, re.M)
            if match:
                library = Path(match[1].strip())
                collector.run(key.lower() + "_sha256", ["sha256sum", library])
                if key == "HCCP_LIBRARY":
                    collector.run("hccp_exported_symbols", ["nm", "-D", "--defined-only", library])
        # The helper uses HDC; local Host modules/sysfs may not describe the NPU driver.
        report["warnings"].append("Host modinfo is not proof of the NPU embedded driver/firmware version.")
    debug = None
    try:
        device = select_device(devices, eid, args.urma_device)
        report["local_device"] = device
        device_path = Path("/sys/class/ubcore") / device
        report["sysfs_device"] = str(device_path.resolve())
        collector.run("jetty_list", ["urma_admin", "show", "dev", device, "jetty"])
        collector.run("tp_list", ["urma_admin", "show", "dev", device, "tp"])
        for jetty in args.jetty_id:
            code, _ = collector.run(f"jetty_{jetty}", ["urma_admin", "show", "dev", device, "jetty", str(jetty)])
            if code != 0:
                # Public UMDK TOOL_RES_KEY_JETTY == 7 (not stats key type 6).
                collector.run(f"jetty_{jetty}_legacy", ["urma_admin", "show_res", "-d", device,
                              "--resource_type", "7", "--key", str(jetty)])
        report["warnings"].append("Jetty IDs must belong to this live workload; terminated-process IDs can be reused.")
        debug = resolve_debug_dir(device_path, Path("/sys/kernel/debug/ubase"))
        report["debugfs_device"] = str(debug)
        for name in ("caps_info", "all_adev_name"):
            collector.read(name, debug / name)
        for name in QOS_FILES:
            code, output = collector.read("qos_" + name, debug / "qos" / name)
            if name == "adev_qos" and code == 0:
                report["configured_ctp_req_resp_vl_pairs"] = configured_response_pairs(output)
    except (OSError, ValueError) as exc:
        report["warnings"].append(str(exc))
        report["warnings"].append("No local debugfs association. An HDC NPU name must NOT be substituted with a same-named Host device.")
    debug_root = Path("/sys/kernel/debug/ubase")
    try:
        report["available_debug_controllers"] = sorted(p.name for p in debug_root.iterdir() if p.is_dir())
    except OSError as exc:
        report["warnings"].append(str(exc))
    report["source_excerpts"] = collect_sources(collector, roots)
    report["warnings"] += [
        "Jetty query pri/priority can be hardware SL in UDMA; confirm against the installed driver implementation.",
        "Configured request/response VL pairs describe capabilities, not observed READ packet routing.",
        "VL bandwidth is aggregate traffic, not opcode/EID/Jetty-filtered; one VL may correspond to multiple SLs.",
    ]
    fields = ["sample", "start_unix", "end_unix", "port", "vl", "tx_kbps", "rx_kbps"]
    count = 0
    with (args.output / "vl_bandwidth.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        if debug is not None:
            print(f"SAMPLING_READY role={args.role} label={args.label} path={debug / 'perf_stats'}", flush=True)
            for sample in range(args.samples):
                start = time.time()
                code, output = collector.read(f"perf_{sample:04d}", debug / "perf_stats")
                end = time.time()
                rows = parse_perf_stats(output) if code == 0 else []
                if not rows:
                    report["warnings"].append(f"Sample {sample}: no recognized per-VL bandwidth; see raw output.")
                    break
                for row in rows:
                    writer.writerow(dict(sample=sample, start_unix=start, end_unix=end, **row))
                count += 1
                file.flush()
                if sample + 1 < args.samples:
                    time.sleep(args.interval)
    report["valid_perf_samples"] = count
    report["commands"] = collector.records
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={args.output.resolve() / 'report.json'} valid_perf_samples={count}")
    print("READ response SL inheritance remains UNVERIFIED; use matching driver/firmware rules or decoded packet evidence.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        sys.exit(1)

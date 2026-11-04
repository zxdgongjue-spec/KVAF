#!/usr/bin/env python3
"""Sample CPU-side UBASE perf_stats concurrently for one or more URMA devices.

Outputs <device>/perf_*.txt and <device>/sampling.log under a new output directory.
No QoS writes, counter resets, driver reloads, debugfs mounts or traffic generation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import platform
import re
import sys
import time

from collect_read_qos_info import (
    Collector, normalize_eid, parse_devices, resolve_debug_dir, select_device,
)


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


def log_message(path, message, console=True):
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")
    if console:
        print(message, flush=True)


def command_record(record):
    return (f"COMMAND name={record['name']} status={record['status']} "
            f"start_unix={record['start_unix']:.6f} end_unix={record['end_unix']:.6f} "
            f"truncated={record['truncated']}")


def valid_device_name(name):
    return re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None and name not in {".", ".."}


def parse_port_group(text):
    """Identify the physical port group from required caps_info fields."""
    values = []
    for field in ("chip_id", "die_id", "logic_port_bitmap"):
        matches = re.findall(rf"^[ \t]*{field}:[ \t]*(.*)$", text, re.M)
        if len(matches) != 1:
            raise ValueError(f"caps_info must contain exactly one {field}")
        value = matches[0].strip()
        if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", value):
            raise ValueError(f"Invalid caps_info {field}: {value!r}")
        values.append(int(value, 16 if value.lower().startswith("0x") else 10))
    if values[2] == 0:
        raise ValueError("caps_info logic_port_bitmap must select at least one port")
    return tuple(values)


def sample_port_group(debug, targets, args):
    """Read each port group once per sample and fan out to its URMA devices."""
    collector = targets[0]["collector"]
    for sample in range(args.samples):
        name = f"perf_{sample:04d}"
        code, output = collector.run(name, ["cat", debug / "perf_stats"], save=False)
        record = collector.records[-1]
        truncated = record["truncated"]
        rows = parse_perf_stats(output) if code == 0 and not truncated else []
        reason = (f"Read failed with status={code}" if code != 0 else
                  "Raw output was truncated" if truncated else
                  "Read succeeded but no recognized per-VL bandwidth table")
        stage = "perf_stats_read" if code != 0 or truncated else "perf_stats_parse"
        for target in targets:
            if target["failed"]:
                continue
            device = target["device"]
            path = target["collector"].output / f"{name}.txt"
            try:
                path.write_text(output, encoding="utf-8")
                log_message(target["log"], command_record(record), console=False)
                if rows:
                    target["count"] += 1
                else:
                    target["failed"] = True
                    log_message(target["log"], f"SAMPLING_FAILED device={device} stage={stage} "
                                f"reason={reason} details={path.resolve()}")
            except OSError as exc:
                target["failed"] = True
                message = f"SAMPLING_FAILED device={device} stage=output_write reason={exc}"
                try:
                    log_message(target["log"], message)
                except OSError:
                    print(message, flush=True)
        if not rows or all(target["failed"] for target in targets):
            break
        if sample + 1 < args.samples:
            time.sleep(args.interval)

    results = {}
    for target in targets:
        complete = not target["failed"] and target["count"] == args.samples
        status = "complete" if complete else "incomplete"
        try:
            log_message(target["log"], f"SAMPLING_DONE status={status} device={target['device']} "
                        f"valid_perf_samples={target['count']} requested_perf_samples={args.samples}")
        except OSError as exc:
            complete = False
            print(f"SAMPLING_FAILED device={target['device']} stage=output_write reason={exc}", flush=True)
        results[target["device"]] = complete
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid", required=True, nargs="+", action="append",
                        help="Local CPU-side EIDs, paired with --urma-device values in order; repeatable")
    parser.add_argument("--urma-device", nargs="+", action="append", default=[],
                        help="CPU-side device names paired with EIDs in order; repeatable; otherwise infer from EIDs")
    parser.add_argument("--samples", type=int, default=1, help="Number of perf_stats reads per chip/die/port-bitmap group")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between reads on each port group")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--label", default="snapshot", help="e.g. idle, test-A, test-B")
    parser.add_argument("--output", type=Path, required=True, help="New parent output directory; one subdirectory per device")
    args = parser.parse_args(argv)
    try:
        eids = [normalize_eid(value) for values in args.eid for value in values]
    except ValueError as exc:
        parser.error(str(exc))
    requested = [name for names in args.urma_device for name in names]
    if requested and len(requested) != len(eids):
        parser.error("--urma-device and --eid must contain the same number of values, paired in order")
    if len(set(requested)) != len(requested):
        parser.error("Each --urma-device must be listed only once")
    if any(not valid_device_name(name) for name in requested):
        parser.error("Device names must be simple directory names using letters, digits, _, . or -")
    if (args.samples < 1 or not 0.1 <= args.interval < float("inf") or
            not 0 < args.timeout < float("inf")):
        parser.error("samples >= 1, finite interval >= 0.1 and finite timeout > 0 required")
    if sys.platform != "linux":
        parser.error("Collection runs on the Linux server; parsers can be tested on any OS")

    collector = Collector(args.output, args.timeout)
    root_log = args.output / "sampling.log"
    metadata = (f"role=cpu label={args.label!r} hostname={platform.node()} kernel={platform.release()} "
                f"requested_perf_samples={args.samples} interval={args.interval} timeout={args.timeout}")
    log_message(root_log, f"COLLECTION_START devices={len(eids)} {metadata}")
    code, device_text = collector.run("urma_devices", ["urma_admin", "show", "--all", "--whole"], save=False)
    log_message(root_log, command_record(collector.records[-1]), console=False)
    if code != 0 or collector.records[-1]["truncated"]:
        log_message(root_log, f"SAMPLING_BLOCKED stage=device_identity reason=Device query failed "
                    f"status={code} truncated={collector.records[-1]['truncated']}; {device_text[-4000:].strip()}")
        log_message(root_log, f"COLLECTION_DONE status=incomplete completed_devices=0 requested_devices={len(eids)}")
        return 1
    devices = parse_devices(device_text)
    groups, selected = {}, set()
    for eid, name in zip(eids, requested or [None] * len(eids)):
        stage = "device_identity"
        device = name
        try:
            device = select_device(devices, eid, name)
            if not valid_device_name(device):
                raise ValueError(f"Invalid device directory name: {device}")
            if device in selected:
                raise ValueError(f"Multiple EIDs select {device}; list each device only once")
            selected.add(device)
            stage = "debugfs_resolution"
            device_path = Path("/sys/class/ubcore") / device
            debug = resolve_debug_dir(device_path, Path("/sys/kernel/debug/ubase")).resolve()
            stage = "port_group_identity"
            code, caps = collector.run(f"caps_info_{device}", ["cat", debug / "caps_info"], save=False)
            record = collector.records[-1]
            log_message(root_log, f"{command_record(record)} path={debug / 'caps_info'}", console=False)
            if code != 0 or record["truncated"]:
                raise ValueError(f"caps_info read failed: status={code} truncated={record['truncated']}; "
                                 f"{caps[-4000:].strip()}")
            group = parse_port_group(caps)
            stage = "output_setup"
            target_collector = Collector(args.output / device, args.timeout)
            log = target_collector.output / "sampling.log"
            log_message(log, f"SAMPLING_START device={device} eid={eid} {metadata}")
            log_message(log, f"DEVICE local_device={device} sysfs_device={device_path.resolve()} "
                        f"perf_stats={debug / 'perf_stats'}", console=False)
            log_message(log, "NOTE: Bandwidth is aggregated per chip/die/port/VL, not per URMA device.", console=False)
            groups.setdefault(group, []).append(dict(device=device, debug=debug, collector=target_collector,
                                                      log=log, count=0, failed=False))
        except (OSError, ValueError) as exc:
            log_message(root_log, f"SAMPLING_BLOCKED device={device or 'auto'} eid={eid} stage={stage} reason={exc}")
            if stage == "device_identity":
                log_message(root_log, f"DEVICE_EIDS={devices}", console=False)

    completed = 0
    if groups:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {}
            for (chip_id, die_id, bitmap), targets in groups.items():
                debug = targets[0]["debug"]
                names = ",".join(target["device"] for target in targets)
                group_info = (f"chip_id={chip_id} die_id={die_id} logic_port_bitmap={bitmap:#x} "
                              f"source_device={targets[0]['device']} path={debug / 'perf_stats'}")
                log_message(root_log, f"SAMPLING_READY devices={names} {group_info}")
                tag = "SHARED_PORT_GROUP" if len(targets) > 1 else "PORT_GROUP"
                for target in targets:
                    log_message(target["log"], f"{tag} devices={names} {group_info}", console=False)
                futures[executor.submit(sample_port_group, debug, targets, args)] = names
            for future in as_completed(futures):
                try:
                    results = future.result()
                    for device, complete in results.items():
                        completed += int(complete)
                        status = "complete" if complete else "incomplete"
                        log_message(root_log, f"DEVICE_DONE device={device} status={status} "
                                    f"details={args.output.resolve() / device / 'sampling.log'}", console=False)
                except (OSError, ValueError) as exc:
                    log_message(root_log, f"SAMPLING_FAILED devices={futures[future]} stage=collection reason={exc}")
    status = "complete" if completed == len(eids) else "incomplete"
    log_message(root_log, f"COLLECTION_DONE status={status} completed_devices={completed} requested_devices={len(eids)}")
    return 0 if completed == len(eids) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        sys.exit(1)

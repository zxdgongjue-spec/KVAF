#!/usr/bin/env python3
"""Sample A5 NPU UB ports through hccn_tool, independently of the benchmark.

Command/output reference: Ascend/mind-cluster, component/ascend-common/
devmanager/hccn/hccn_tool.go (GetNpuDevNetPortInfo, GetNPUInterfaceTrafficNpu,
GetNPUUbStatInfo). See npu_port_stat.md for measurement scope and usage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import re
import shutil
import subprocess
import sys
import time


COUNTERS = {
    "tx_ctp_pps": "ub_umoc_ctph_cnt_tx",
    "rx_ctp_pps": "ub_umoc_ctph_cnt_rx",
    "tx_flit_ps": "tx_busi_flit_num",
    "rx_flit_ps": "rx_busi_flit_num",
}


def run_hccn(tool, arguments, timeout):
    command = [tool, *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}: rc={result.returncode}; "
                           f"{(result.stderr + result.stdout).strip()[-1500:]}")
    return result.stdout


def parse_ports(text):
    """Read the first UdieID/PortID table; exclude ETH/UBoE interfaces."""
    ports, columns, in_rows = {}, None, False
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            if columns is not None and line.strip().startswith("+-"):
                if in_rows:
                    break
                in_rows = True
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        names = [re.sub(r"[\s_]", "", cell).lower() for cell in cells]
        if "udieid" in names and "portid" in names:
            columns = {name: names.index(name) for name in
                       ("udieid", "portid", "porttype", "linkstatus") if name in names}
            if len(columns) != 4:
                raise ValueError("Unrecognized -dev_info table header")
            continue
        if columns is None or len(cells) <= max(columns.values()):
            continue
        if cells[columns["porttype"]].upper() != "UB":
            continue
        udie, port = cells[columns["udieid"]], cells[columns["portid"]]
        if not udie.isdecimal() or not port.isdecimal():
            raise ValueError(f"Invalid UB port row: {line}")
        ports[int(udie), int(port)] = cells[columns["linkstatus"]].upper()
    if not ports:
        raise ValueError("No UB ports found in -dev_info; check A5 hccn_tool version/output")
    return ports


def parse_bandwidth(text):
    rates = {}
    for direction, value in re.findall(
            r"^\s*Bandwidth\s+(TX|RX):\s*(\S+)\s+MB/(?:sec|s)\s*$", text, re.M):
        number = float(value)
        if not math.isfinite(number) or number < 0 or direction in rates:
            raise ValueError("Invalid or duplicate bandwidth reading")
        rates[direction] = number
    if set(rates) != {"TX", "RX"}:
        raise ValueError(f"Missing TX/RX bandwidth (MB/sec): {text.strip()[-1000:]}")
    return rates["TX"], rates["RX"]


def parse_counters(text):
    stats = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not separator or key not in COUNTERS.values():
            continue
        try:
            number = int(value, 16 if value.lower().startswith("0x") else 10)
        except ValueError:
            continue
        if number >= 0:
            if key in stats:
                raise ValueError(f"Duplicate counter: {key}")
            stats[key] = number
    if not stats:
        raise ValueError(f"No recognized UB counters: {text.strip()[-1000:]}")
    return stats


def counter_rates(previous, current):
    """Missing/reset counters stay unavailable, never zero or negative rates."""
    if previous is None or current is None:
        return {}
    dt = current[0] - previous[0]
    if dt <= 0:
        return {}
    rates = {"counter_dt": dt}
    for output, field in COUNTERS.items():
        before, after = previous[1].get(field), current[1].get(field)
        if before is not None and after is not None and after >= before:
            rates[output] = (after - before) / dt
    return rates


def collect_port(args, key, baseline=False):
    start = time.monotonic()
    udie, port = key
    selection = ["-i", args.device_id, "-u", udie, "-p", port]
    row = dict(device=args.device_id, udie=udie, port=port)
    errors, snapshot = [], None
    if args.counters:
        try:
            raw = run_hccn(args.hccn_tool, ["-g", "-stat", *selection], args.timeout)
            snapshot = (time.monotonic(), parse_counters(raw))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(f"stat: {exc}")
    if not baseline:
        try:
            # Same hardware measurement setting as the official NPU Exporter.
            raw = run_hccn(args.hccn_tool,
                           ["-g", "-bandwidth", *selection, "-time", 100], args.timeout)
            row["tx_MBps"], row["rx_MBps"] = parse_bandwidth(raw)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(f"bandwidth: {exc}")
    row["query_seconds"] = time.monotonic() - start
    row["error"] = "; ".join(errors)
    return key, row, snapshot


def sample(args, keys):
    previous, failed, last_start = {}, False, None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        if args.counters:
            last_start = time.monotonic()
            baseline = list(executor.map(lambda k: collect_port(args, k, True), keys))
            for key, row, snapshot in baseline:
                previous[key] = snapshot
                if row["error"]:
                    failed = True
                    print(f"[WARN] baseline {key}: {row['error']}", file=sys.stderr)
        iteration = 0
        while args.count == 0 or iteration < args.count:
            if last_start is not None:
                remaining = args.interval - (time.monotonic() - last_start)
                if remaining > 0:
                    time.sleep(remaining)
            start = time.monotonic()
            results = list(executor.map(lambda k: collect_port(args, k), keys))
            collect_seconds = time.monotonic() - start
            actual_interval = "--" if last_start is None else f"{start - last_start:.3f}s"
            last_start = start
            iteration += 1
            lines = [f"===== sample #{iteration} {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"target={args.interval:g}s start_dt={actual_interval} "
                     f"collect={collect_seconds:.3f}s workers={args.workers} ====="]
            header = " NPU Udie Port      Query(s)    TxBW(MB/s)    RxBW(MB/s)"
            if args.counters:
                header += "         dt(s)       TxCTP/s       RxCTP/s      TxFlit/s      RxFlit/s"
            lines.append(header)
            warnings = []
            for key, row, snapshot in results:
                row.update(counter_rates(previous.get(key), snapshot))
                previous[key] = snapshot
                values = ["query_seconds", "tx_MBps", "rx_MBps"]
                if args.counters:
                    values += ["counter_dt", *COUNTERS]
                numbers = " ".join(f"{row[name]:13,.3f}" if name in row else f"{'N/A':>13}"
                                   for name in values)
                lines.append(f"{args.device_id:4} {key[0]:4} {key[1]:4} {numbers}")
                if row["error"]:
                    failed = True
                    warnings.append(f"[WARN] {key}: {row['error']}")
            print("\n".join(lines) + "\n", flush=True)
            if warnings:
                print("\n".join(warnings), file=sys.stderr, flush=True)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description="采集 A5 NPU 的 UB 端口收发带宽与可选计数器")
    parser.add_argument("--device-id", type=int, required=True, help="hccn_tool 的 NPU logic ID")
    parser.add_argument("--ports", help="Udie:Port 列表，如 0:4,1:9；默认选择 UP 的 UB 端口")
    parser.add_argument("--list", action="store_true", help="仅列出实际发现的 UB 端口")
    parser.add_argument("-i", "--interval", type=float, default=1,
                        help="两轮开始之间的目标间隔秒数，默认 1；采集耗时超过间隔时不额外等待")
    parser.add_argument("-n", "--count", type=int, default=1, help="采样轮数，0 表示持续采集")
    parser.add_argument("--counters", action="store_true", help="同时查询 CTP 包率与原始 flit/s")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行查询端口数，默认 8；整轮采集完成后统一打印")
    parser.add_argument("--timeout", type=float, default=10, help="每条 hccn_tool 命令的超时秒数")
    parser.add_argument("--hccn-tool", default=shutil.which("hccn_tool") or
                        "/usr/local/Ascend/driver/tools/hccn_tool")
    args = parser.parse_args(argv)
    if (args.device_id < 0 or args.count < 0 or args.workers < 1 or
            not math.isfinite(args.interval) or args.interval <= 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("device-id/count 必须非负，workers/interval/timeout 必须为正数")
    try:
        raw = run_hccn(args.hccn_tool, ["-g", "-dev_info", "-i", args.device_id], args.timeout)
        ports = parse_ports(raw)
        for (udie, port), state in sorted(ports.items()):
            print(f"NPU={args.device_id} Udie={udie} Port={port} Type=UB Link={state}",
                  file=sys.stderr)
        if args.list:
            return 0
        if args.ports:
            if not re.fullmatch(r"\d+:\d+(,\d+:\d+)*", args.ports):
                raise ValueError("--ports 格式应为 0:4,1:9")
            keys = sorted({tuple(map(int, item.split(":"))) for item in args.ports.split(",")})
            if any(key not in ports for key in keys):
                raise ValueError("--ports 包含本 NPU -dev_info 未报告的 UB 端口")
        else:
            keys = sorted(key for key, state in ports.items() if state == "UP")
        if not keys:
            raise ValueError("没有 UP 的 UB 端口；请检查 --list 输出")
        return sample(args, keys)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

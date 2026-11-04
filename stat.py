#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_BYTES_PER_FLIT = 20


def run_ubctl(chip, port, timeout=10):
    cmd = [
        "ubctl", "-m", "dl",
        "-p", str(port),
        "-f", "pkt_stats",
        "-d", "1",
        "-c", str(chip),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cmd={' '.join(cmd)} rc={result.returncode}, err={result.stderr.strip()}"
        )
    return result.stdout


def parse_stats(text):
    """解析 ubctl 输出为字典，支持 0x 十六进制，同名字段只取第一次出现的值。"""
    stats = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("---") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            num = int(val, 0)
        except ValueError:
            continue
        if key not in stats:
            stats[key] = num
    return stats


def collect_one(chip, port):
    """
    采集单个端口，返回 ((chip,port), stats, t_start, t_end, err)
    """
    t_start = time.monotonic()
    try:
        text = run_ubctl(chip, port)
        t_end = time.monotonic()
        return (chip, port), parse_stats(text), t_start, t_end, None
    except Exception as e:
        return (chip, port), None, t_start, time.monotonic(), str(e)


def collect_all(chips, ports, workers=8):
    """
    并行采集所有 (chip, port)。
    返回 data, t_before, t_after 三个字典，键为 (chip, port)。
    """
    data = {}
    t_before = {}
    t_after = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(collect_one, c, p) for c in chips for p in ports]
        for fut in as_completed(futures):
            key, stats, t1, t2, err = fut.result()
            if err:
                print(f"[WARN] chip={key[0]} port={key[1]} 采集失败: {err}", file=sys.stderr)
            data[key] = stats
            t_before[key] = t1
            t_after[key] = t2
    return data, t_before, t_after


def precise_sleep(target_interval, slice_sec=0.05):
    start = time.monotonic()
    while True:
        remaining = target_interval - (time.monotonic() - start)
        if remaining <= 0:
            break
        time.sleep(min(slice_sec, remaining))
    return time.monotonic() - start


def fmt_pkt_rate(v):
    """包率：整数部分加千分位，保留 1 位小数"""
    if v < 0:
        return f"-{abs(v):,.1f}"
    return f"{v:,.1f}"


def fmt_bw_MBps(v):
    """带宽统一为 MB/s，加千分位，保留 3 位小数"""
    mb = v / 1e6
    if mb < 0:
        return f"-{abs(mb):,.3f} MB/s"
    return f"{mb:,.3f} MB/s"


def main():
    parser = argparse.ArgumentParser(
        description="统计各端口收发包率与收发带宽（逐端口并行，使用每端口独立采样时刻）"
    )
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="目标采样间隔秒数，默认 1.0")
    parser.add_argument("-c", "--chips", default="0,1",
                        help="芯片列表，逗号分隔，默认 0,1")
    parser.add_argument("-p", "--ports", default="0,1,2,3,4,5,6,7",
                        help="端口列表，逗号分隔，默认 0-7")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="采样次数，默认 1；设为 0 表示无限循环")
    parser.add_argument("--bytes-per-flit", type=int, default=DEFAULT_BYTES_PER_FLIT,
                        help=f"每个 flit 的字节数，默认 {DEFAULT_BYTES_PER_FLIT}")
    parser.add_argument("--sleep-slice", type=float, default=0.05,
                        help="精确等待粒度，默认 0.05s")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行采集线程数，默认 8")
    args = parser.parse_args()

    chips = [int(x) for x in args.chips.split(",") if x.strip()]
    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    bpflit = args.bytes_per_flit

    print(
        f"开始采集: chips={chips}, ports={ports}, interval={args.interval}s, "
        f"bytes_per_flit={bpflit}, workers={args.workers}",
        file=sys.stderr,
    )

    prev, t_prev_before, t_prev_after = collect_all(chips, ports, workers=args.workers)

    iteration = 0
    while True:
        iteration += 1
        real_wait = precise_sleep(args.interval, args.sleep_slice)
        curr, t_curr_before, t_curr_after = collect_all(chips, ports, workers=args.workers)

        dts = []
        for key in curr.keys():
            if key in prev and prev[key] is not None and curr[key] is not None:
                dt_k = t_curr_after[key] - t_prev_after[key]
                if dt_k > 0:
                    dts.append(dt_k)

        if not dts:
            print("[WARN] 本轮没有可用的端口数据，跳过", file=sys.stderr)
            prev, t_prev_before, t_prev_after = curr, t_curr_before, t_curr_after
            continue

        dt_min, dt_max = min(dts), max(dts)
        dt_avg = sum(dts) / len(dts)

        print(f"===== 采样 #{iteration} {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"目标间隔={args.interval}s 实际间隔: min={dt_min:.3f}s "
              f"avg={dt_avg:.3f}s max={dt_max:.3f}s (sleep耗时={real_wait:.3f}s) =====")
        print(f"{'Chip':>4} {'Port':>4} "
              f"{'dt(s)':>7} "
              f"{'TxPkt/s':>14} {'RxPkt/s':>14} "
              f"{'TxBW':>16} {'RxBW':>16}")

        sum_tx_pps = 0.0
        sum_rx_pps = 0.0
        sum_tx_Bps = 0.0
        sum_rx_Bps = 0.0
        has_data = False

        for key in sorted(curr.keys()):
            chip, port = key
            s1 = prev.get(key)
            s2 = curr.get(key)
            if s1 is None or s2 is None:
                print(f"{chip:>4} {port:>4}  采集失败/缺失")
                continue

            dt_k = t_curr_after[key] - t_prev_after[key]
            if dt_k <= 0:
                print(f"{chip:>4} {port:>4}  dt无效({dt_k:.3f}s)")
                continue

            tx_pkt_d = s2.get("tx_busi_pkt_num", 0) - s1.get("tx_busi_pkt_num", 0)
            rx_pkt_d = s2.get("rx_busi_pkt_num", 0) - s1.get("rx_busi_pkt_num", 0)
            tx_flit_d = s2.get("tx_busi_flit_num", 0) - s1.get("tx_busi_flit_num", 0)
            rx_flit_d = s2.get("rx_busi_flit_num", 0) - s1.get("rx_busi_flit_num", 0)

            tx_pps = tx_pkt_d / dt_k
            rx_pps = rx_pkt_d / dt_k
            tx_Bps = tx_flit_d * bpflit / dt_k
            rx_Bps = rx_flit_d * bpflit / dt_k

            sum_tx_pps += tx_pps
            sum_rx_pps += rx_pps
            sum_tx_Bps += tx_Bps
            sum_rx_Bps += rx_Bps
            has_data = True

            print(f"{chip:>4} {port:>4} "
                  f"{dt_k:>7.3f} "
                  f"{fmt_pkt_rate(tx_pps):>14} {fmt_pkt_rate(rx_pps):>14} "
                  f"{fmt_bw_MBps(tx_Bps):>16} {fmt_bw_MBps(rx_Bps):>16}")

        if has_data:
            print(f"{'ALL':>4} {'--':>4} {'--':>7} "
                  f"{fmt_pkt_rate(sum_tx_pps):>14} {fmt_pkt_rate(sum_rx_pps):>14} "
                  f"{fmt_bw_MBps(sum_tx_Bps):>16} {fmt_bw_MBps(sum_rx_Bps):>16}")

        print()
        prev, t_prev_before, t_prev_after = curr, t_curr_before, t_curr_after

        if args.count > 0 and iteration >= args.count:
            break


if __name__ == "__main__":
    main()
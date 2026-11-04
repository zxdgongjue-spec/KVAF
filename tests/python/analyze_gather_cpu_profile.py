#!/usr/bin/env python3
"""Summarize CPU_GATHER_PROFILE timing fields from multiple logs into one CSV."""
import argparse
from collections import defaultdict
import csv
import glob
from pathlib import Path
import re
import statistics
import sys


PROFILE = re.compile(r"\bCPU_GATHER_PROFILE\s+(.*)")
COLUMNS = ["log_file", "token_bytes", "sgl", "lane", "rows", "active_threads", "field",
           "samples", "mean_us", "p50_us", "p95_us", "p99_us", "min_us", "max_us",
           "stddev_us", "variance_us2", "cv_pct"]


def input_files(inputs):
    paths = set()
    for item in inputs:
        literal = Path(item).expanduser()
        matches = [literal] if literal.exists() else [Path(p) for p in glob.glob(str(literal), recursive=True)]
        if not matches:
            raise ValueError(f"没有匹配的文件：{item}")
        for path in matches:
            if path.is_dir():
                paths.update(p.resolve() for p in path.rglob("*.log") if p.is_file())
            else:
                paths.add(path.resolve())
    if not paths:
        raise ValueError("没有找到日志文件")
    return sorted(paths)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_name(name):
    return name.removesuffix("_ns").removesuffix("_us") + "_ns"


def collect(paths, fields=None, first_sequence=None, last_sequence=None, sgl=None):
    groups = defaultdict(list)
    selected_fields = {timing_name(name) for name in fields} if fields else None
    request_count = 0
    for path in paths:
        named_sgl = {int(value) for value in re.findall(r"sgl[=_ -]*(\d+)(?!\d)", path.stem, re.I)}
        if len(named_sgl) > 1:
            raise ValueError(f"{path}: 文件名中包含多个 SGL 值")
        file_sgl = next(iter(named_sgl), sgl)
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                match = PROFILE.search(line)
                if not match:
                    continue
                values = dict(re.findall(r"(\w+)=([^\s]+)", match.group(1)))
                if values.get("result") == "error":
                    continue
                try:
                    sequence = int(values["sequence"])
                    if first_sequence is not None and sequence < first_sequence:
                        continue
                    if last_sequence is not None and sequence > last_sequence:
                        continue
                    if file_sgl is None:
                        raise ValueError("文件名需包含 sgl1/sgl6 等标记，或通过 --sgl 指定")
                    if sgl is not None and file_sgl != sgl:
                        raise ValueError("文件名的 SGL 与 --sgl 不一致")
                    rows, size = int(values["rows"]), int(values["bytes"])
                    if rows <= 0 or size <= 0 or size % rows:
                        raise ValueError("无法通过 bytes/rows 确定固定 token 大小")
                    identity = (str(path), size // rows, file_sgl, int(values["lane"]),
                                rows, int(values["active_threads"]))
                    for name, value in values.items():
                        if not name.endswith("_ns"):
                            continue
                        if selected_fields is not None and name not in selected_fields:
                            continue
                        if selected_fields is None and name in ("observed_ns", "done_ns"):
                            continue
                        if values.get(name.removesuffix("_ns") + "_valid") == "0":
                            continue
                        duration = int(value)
                        if duration < 0:
                            raise ValueError(f"{name} 为负数")
                        groups[(*identity, name.removesuffix("_ns") + "_us")].append(duration / 1000)
                    request_count += 1
                except (KeyError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not groups:
        raise ValueError("没有可统计的时间字段，请检查日志、sequence 范围和 --fields")
    return groups, request_count


def summarize(groups):
    result = []
    for identity, values in sorted(groups.items()):
        mean = statistics.mean(values)
        variance = statistics.variance(values) if len(values) > 1 else None
        stddev = variance ** .5 if variance is not None else None
        row = dict(zip(COLUMNS[:7], identity))
        row.update(samples=len(values), mean_us=mean, p50_us=percentile(values, .5),
                   p95_us=percentile(values, .95), p99_us=percentile(values, .99),
                   min_us=min(values), max_us=max(values), stddev_us=stddev,
                   variance_us2=variance, cv_pct=100 * stddev / mean if stddev is not None and mean else None)
        result.append(row)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="日志文件、glob 或目录（递归查找 *.log）")
    parser.add_argument("-o", "--output", type=Path, default=Path("gather-perf-results/cpu_summary.csv"))
    parser.add_argument("--fields", nargs="+", help="仅统计这些耗时，例如 total copy prepare join_wait")
    parser.add_argument("--first-sequence", type=int, help="仅统计 sequence >= 此值的请求")
    parser.add_argument("--last-sequence", type=int, help="仅统计 sequence <= 此值的请求")
    parser.add_argument("--sgl", type=int, help="文件名没有 SGL 标记时指定，适用于本次所有输入文件")
    args = parser.parse_args(argv)
    if args.first_sequence is not None and args.last_sequence is not None and args.first_sequence > args.last_sequence:
        parser.error("first-sequence 不能大于 last-sequence")
    if args.sgl is not None and args.sgl < 1:
        parser.error("--sgl 必须大于 0")
    try:
        paths = input_files(args.inputs)
        output = args.output.expanduser().resolve()
        if output in paths:
            raise ValueError("输出 CSV 不能覆盖输入日志")
        groups, count = collect(paths, args.fields, args.first_sequence, args.last_sequence, args.sgl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for row in summarize(groups):
                writer.writerow({name: f"{value:.6f}" if isinstance(value, float) else value
                                 for name, value in row.items()})
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"处理 {len(paths)} 个文件、{count} 条请求汇总，输出 {len(groups)} 条字段统计：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

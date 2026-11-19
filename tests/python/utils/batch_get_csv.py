"""CSV reports for one BatchGet benchmark invocation."""

import csv
from datetime import datetime
from pathlib import Path


METRICS = ("mean_us", "p50_us", "p90_us", "p99_us", "mean_GBps", "p50_GBps", "p99_GBps", "mean_GiBps")
SUMMARY_FIELDS = (
    "backend", "transport", "execution", "get_timing", "graph_api", "device_id", "peer",
    "request_length", "table_capacity", "miss_count", "token_bytes", "tokens_per_block",
    "payload_bytes_per_call", "source_access_pattern", "output_access_pattern", "sort_input_indices", "seed",
    "samples", "warmup", "aiv_cores", "aicpu_workers", "jettys_per_core", "max_inflight_total",
    "npu_eids", "peer_ub_device", "profiling_enabled", "kernel_profile_clock_mhz",
    *(f"{scope}_{metric}" for scope in ("e2e", "kernel") for metric in METRICS),
)
PROFILE_FIELDS = ("backend", "execution", "device_id", "miss_count", "kind", "name", "metric", "value")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"BATCH_GET_CSV_WRITTEN path={path} rows={len(rows)}", flush=True)


class BatchGetCsv:
    def __init__(self, args):
        directory = args.output_dir
        if directory is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            directory = Path("reports/batch-get") / f"{stamp}-{args.backend}-d{args.device_id}"
        self.summary_path = directory / "summary.csv"
        self.profile_path = directory / "profile.csv" if args.kernel_profile else None
        self.summary_rows = []
        self.profile_rows = []
        self.config = {name: getattr(args, name) for name in (
            "backend", "transport", "device_id", "peer", "request_length", "table_capacity", "miss_count",
            "token_bytes", "tokens_per_block", "source_access_pattern", "output_access_pattern", "seed", "warmup",
            "aiv_cores", "aicpu_workers",
        )}
        self.config.update(
            payload_bytes_per_call=args.miss_count * args.token_bytes,
            sort_input_indices=int(args.sort_input_indices),
            jettys_per_core=args.jettys_per_core if args.backend == "aiv" else None,
            max_inflight_total=args.max_inflight_total if args.backend == "aiv" else None,
            npu_eids=(",".join(args.npu_eids) or "auto") if args.transport == "urma" else None,
            peer_ub_device=(args.peer_ub_device or "auto") if args.transport == "urma" else None,
            profiling_enabled=int(args.kernel_profile),
            kernel_profile_clock_mhz=args.kernel_profile_clock_mhz if args.kernel_profile and args.aiv_cores else None,
        )
        self.graph_api = args.graph_api

    def add_summary(self, execution, timing, samples, measurements):
        self.summary_rows.append(dict(
            self.config, execution=execution, get_timing=timing, samples=samples,
            graph_api=self.graph_api if execution == "graph" else None,
            **{f"{scope}_{key}": value for scope, metrics in measurements.items() for key, value in metrics.items()},
        ))

    def add_profile(self, execution, details):
        config = {name: self.config[name] for name in ("backend", "device_id", "miss_count")}
        self.profile_rows.extend(dict(config, execution=execution, kind=kind, name=name, metric=metric, value=value)
                                 for kind, name, metric, value in details)

    def write(self):
        write_csv(self.summary_path, SUMMARY_FIELDS, self.summary_rows)
        if self.profile_path is not None:
            write_csv(self.profile_path, PROFILE_FIELDS, self.profile_rows)

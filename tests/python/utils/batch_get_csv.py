"""CSV reports for one BatchGet benchmark invocation."""

import csv
from datetime import datetime
from pathlib import Path
from .get_options import gather_read_wave_settings


METRICS = ("mean_us", "p50_us", "p90_us", "p99_us", "mean_GBps", "p50_GBps", "p99_GBps", "mean_GiBps")
SUMMARY_FIELDS = (
    "backend", "transport", "execution",
    "request_length", "table_capacity", "miss_count", "token_bytes", "tokens_per_block",
    "payload_bytes_per_call", "aiv_cores", "aicpu_workers", "jettys_per_core", "max_inflight_total",
    "gather_ddr_bytes", "scatter_hbm_bytes", "gather_read_wave_wrs", "gather_read_wave_wrs_mode",
        "jettys_per_worker",
    *(f"{scope}_{metric}" for scope in ("e2e", "kernel") for metric in METRICS),
)
PROFILE_FIELDS = ("backend", "execution", "miss_count", "kind", "name", "metric", "value")
TIMELINE_FIELDS = ("execution", "sample", "worker", "start_us", "finish_us", "ready_wait_us", "last")


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
        self.timeline_rows = []
        self.breakdown_rows = []
        self.config = {name: getattr(args, name) for name in (
            "backend", "transport", "request_length", "table_capacity", "miss_count", "token_bytes", "tokens_per_block",
            "aiv_cores", "aicpu_workers",
        )}
        self.config.update(
            jettys_per_worker=args.jettys_per_worker if args.backend == "gather_read" else None,
            payload_bytes_per_call=args.miss_count * args.token_bytes,
            jettys_per_core=args.jettys_per_core if args.backend == "aiv" else None,
            max_inflight_total=args.max_inflight_total if args.backend == "aiv" else None,
            gather_ddr_bytes=getattr(args, "gather_ddr_bytes", 0),
            scatter_hbm_bytes=getattr(args, "scatter_hbm_bytes", 0),
            **gather_read_wave_settings(args),
        )

    def add_summary(self, execution, measurements):
        self.summary_rows.append(dict(
            self.config, execution=execution,
            **{f"{scope}_{key}": value for scope, metrics in measurements.items() for key, value in metrics.items()},
        ))

    def add_profile(self, execution, details):
        config = {name: self.config[name] for name in ("backend", "miss_count")}
        self.profile_rows.extend(dict(config, execution=execution, kind=kind, name=name, metric=metric, value=value)
                                 for kind, name, metric, value in details)

    def write(self):
        write_csv(self.summary_path, SUMMARY_FIELDS, self.summary_rows)
        if self.profile_path is not None:
            write_csv(self.profile_path, PROFILE_FIELDS, self.profile_rows)
        if self.timeline_rows:
            write_csv(self.summary_path.with_name("worker_timeline.csv"), TIMELINE_FIELDS, self.timeline_rows)
        if self.breakdown_rows:
            write_csv(self.summary_path.with_name("kernel_breakdown.csv"), self.breakdown_rows[0].keys(),
                      self.breakdown_rows)

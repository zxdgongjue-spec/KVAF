"""CSV reports for one BatchGet benchmark invocation."""

import csv
from datetime import datetime
from pathlib import Path


METRICS = ("mean_us", "p50_us", "p90_us", "p99_us", "mean_GBps", "p50_GBps", "p99_GBps", "mean_GiBps")
SUMMARY_FIELDS = (
    "backend", "transport", "execution",
    "request_length", "table_capacity", "miss_count", "token_bytes", "tokens_per_block",
    "payload_bytes_per_call", "aiv_cores", "aicpu_workers", "jettys_per_core", "max_inflight_total",
    "get_binding", "graph_api", "graph_submit", "gets_per_replay", "latency_scope",
    *(f"{scope}_{metric}" for scope in ("e2e", "kernel", "host_replay") for metric in METRICS),
)
PROFILE_FIELDS = ("backend", "execution", "miss_count", "kind", "name", "metric", "value")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"BATCH_GET_CSV_WRITTEN path={path} rows={len(rows)}", flush=True)


class BatchGetCsv:
    def __init__(self, args):
        self.get_binding, self.graph_api, self.graph_get_count = args.get_binding, args.graph_api, args.graph_get_count
        self.graph_submit = args.graph_submit
        directory = args.output_dir
        if directory is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            directory = Path("reports/batch-get") / f"{stamp}-{args.backend}-d{args.device_id}"
        self.summary_path = directory / "summary.csv"
        self.profile_path = directory / "profile.csv" if args.kernel_profile else None
        self.summary_rows = []
        self.profile_rows = []
        self.config = {name: getattr(args, name) for name in (
            "backend", "transport", "request_length", "table_capacity", "miss_count", "token_bytes", "tokens_per_block",
            "aiv_cores", "aicpu_workers",
        )}
        self.config.update(
            payload_bytes_per_call=args.miss_count * args.token_bytes,
            jettys_per_core=args.jettys_per_core if args.backend == "aiv" else None,
            max_inflight_total=args.max_inflight_total if args.backend == "aiv" else None,
        )

    def add_summary(self, execution, measurements):
        self.summary_rows.append(dict(
            self.config, execution=execution,
            get_binding=self.get_binding if execution == "eager" else "capture",
            graph_api=self.graph_api if execution == "graph" else None,
            graph_submit=self.graph_submit if execution == "graph" else None,
            gets_per_replay=self.graph_get_count if execution == "graph" else 1,
            latency_scope="amortized_per_get" if execution == "graph" and self.graph_get_count > 1 else "single_get",
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

"""CSV reports for one BatchGet benchmark invocation."""

import csv
from datetime import datetime
from pathlib import Path


METRICS = ("mean_us", "p50_us", "p90_us", "p99_us", "max_us", "mean_GBps", "p50_GBps", "p99_GBps", "mean_GiBps")
SUMMARY_FIELDS = (
    "backend", "transport", "execution",
    "request_length", "table_capacity", "miss_count", "token_bytes", "tokens_per_block",
    "payload_bytes_per_call", "aiv_cores", "aicpu_workers", "jettys_per_core", "max_inflight_total",
    "sge_per_wr", "sgl_pipeline", "sgl_pipeline_mode", "sgl_push_chunked", "sgl_push_head_rows", "sgl_push_cooperative",
    *(f"{scope}_{metric}" for scope in ("e2e", "kernel") for metric in METRICS),
)
PROFILE_FIELDS = ("backend", "execution", "miss_count", "kind", "name", "metric", "value")
SAMPLE_FIELDS = ("backend", "execution", "sample", "e2e_us", "kernel_us", "outside_kernel_us",
                 "e2e_timing_source", "kernel_timing_source")


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
        self.samples_path = directory / "samples.csv"
        self.sample_rows = []
        self.summary_fields = SUMMARY_FIELDS
        if args.sgl_payload_only:
            self.summary_fields += tuple(f"sgl_payload_{metric}" for metric in
                                         (*METRICS, "p95_us", "data_wrs", "ready_wrs", "waves", "tail_cqes"))
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
            sge_per_wr=args.sge_per_wr if args.backend in ("cpu_sgl_write", "aiv_cpu_sgl_write", "cpu_sgl_push", "aiv_cpu_sgl_push") else None,
            sgl_pipeline=int(getattr(args, "sgl_pipeline", False)),
            sgl_pipeline_mode="device_push_chunked" if getattr(args, "sgl_push_chunked", False) else "device_push_head_suffix" if args.backend in ("cpu_sgl_push", "aiv_cpu_sgl_push") else "chunked" if getattr(args, "sgl_pipeline", False) else "disabled",
            sgl_push_chunked=int(getattr(args, "sgl_push_chunked", False)),
            sgl_push_head_rows=getattr(args, "sgl_push_head_rows", 1024),
            sgl_push_cooperative=int(getattr(args, "sgl_push_cooperative", False)),
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

    def add_samples(self, execution, samples, kernel_samples, e2e_source, kernel_source):
        if len(samples) != len(kernel_samples):
            raise ValueError("BatchGet paired sample count mismatch")
        for sample, (e2e, kernel) in enumerate(zip(samples, kernel_samples)):
            self.sample_rows.append(dict(
                backend=self.config["backend"], execution=execution, sample=sample,
                e2e_us=f"{e2e * 1000:.6f}", kernel_us=f"{kernel * 1000:.6f}",
                outside_kernel_us=f"{(e2e - kernel) * 1000:.6f}",
                e2e_timing_source=e2e_source, kernel_timing_source=kernel_source,
            ))

    def write(self):
        write_csv(self.summary_path, self.summary_fields, self.summary_rows)
        if self.sample_rows:
            write_csv(self.samples_path, SAMPLE_FIELDS, self.sample_rows)
        if self.profile_path is not None:
            write_csv(self.profile_path, PROFILE_FIELDS, self.profile_rows)

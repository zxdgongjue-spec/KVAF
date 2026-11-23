"""CSV reports for one BatchGet benchmark invocation."""

import csv
import json
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
        self.graph_sample_metadata = {name: getattr(args, name, None) for name in (
            "device_id", "warmup", "iterations", "graph_replays", "aicpu_workers",
        )}
        directory = args.output_dir
        if directory is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            directory = Path("reports/batch-get") / f"{stamp}-{args.backend}-d{args.device_id}"
        self.summary_path = directory / "summary.csv"
        self.profile_path = directory / "profile.csv" if args.kernel_profile else None
        self.summary_rows = []
        self.profile_rows = []
        self.graph_samples = None
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

    def add_graph_samples(self, profiles, host_bounds, event_ms):
        if not self.profile_path:
            return
        if not len(profiles) == len(host_bounds) == len(event_ms):
            raise AssertionError("graph timing/profile samples do not match")
        rounds = []
        for ordinal, (profile, bounds, elapsed) in enumerate(zip(profiles, host_bounds, event_ms), 1):
            nodes = profile if profile and isinstance(profile[0], list) else [profile]
            if any(p.get("device_clock") != "aicpu_monotonic_raw_ns" for node in nodes for p in node):
                raise AssertionError("AICPU raw-clock profiling requires the rebuilt NPU binding and ExecutionV6 kernel")
            rounds.append(dict(round=ordinal, host_begin_ns=bounds[0], host_end_ns=bounds[1],
                event_elapsed_ns=round(elapsed * self.graph_get_count * 1e6), nodes=[dict(
                    node=node, workers=[dict(worker=worker, **{field: int(p[field]) for field in (
                        "device_entry_ns", "device_start_ns", "device_finish_ns", "device_return_ns")})
                        for worker, p in enumerate(workers)]) for node, workers in enumerate(nodes)]))
        self.graph_samples = dict(format="kvaf-aicpu-samples-v3", execution="graph",
            backend=self.config["backend"], graph_api=self.graph_api, graph_submit=self.graph_submit,
            device_clock="aicpu_monotonic_raw_ns", host_clock="python_perf_counter_ns",
            gets_per_replay=self.graph_get_count, **self.graph_sample_metadata,
            timed_prefix_replays=(self.graph_sample_metadata["warmup"] + 1
                                  if self.graph_submit == "native" else 0), rounds=rounds)

    def write(self):
        write_csv(self.summary_path, SUMMARY_FIELDS, self.summary_rows)
        if self.profile_path is not None:
            write_csv(self.profile_path, PROFILE_FIELDS, self.profile_rows)
            if self.graph_samples is not None:
                path = self.profile_path.with_name("aicpu_samples.json")
                path.write_text(json.dumps(self.graph_samples, separators=(",", ":")) + "\n", encoding="ascii")
                print(f"BATCH_GET_SAMPLES_WRITTEN path={path} rounds={len(self.graph_samples['rounds'])}", flush=True)
        if self.graph_samples is None:
            self.summary_path.with_name("aicpu_samples.json").unlink(missing_ok=True)

"""Select the benchmark API path without changing its workload or providers."""

import os
import time


def add_execution_mode(parser):
    parser.add_argument(
        "--execution-mode", choices=("legacy", "per-call"), default="legacy",
        help="legacy fixed provider (default), or per-call dispatcher and resource leases",
    )


class BenchmarkExecution:
    def __init__(self, kvaf, mode, *, transport="urma", executor="aiv",
                 algorithm="direct", put_capacity=128, get_capacity=65536):
        if mode not in ("legacy", "per-call"):
            raise ValueError("execution mode must be legacy or per-call")
        self.mode = mode
        self.runtime_kwargs = {}
        self.put_kwargs = {}
        self.get_kwargs = {}
        self.get_config = None
        self.get_capacity = get_capacity
        self.output_lease = 0
        self.binding_setup_us = self.binding_teardown_us = 0.0
        supported = transport == "urma" and (executor, algorithm) in (("aiv", "direct"), ("aicpu", "gather"))
        self.output_binding = ("legacy" if mode == "legacy" else "not-needed" if not supported
                               else "transient" if os.environ.get("KVAF_PERF_REUSE_GET_OUTPUT", "1") == "0"
                               else "reused")
        # Do not reference new binding symbols or pass new kwargs to main's API.
        if mode == "legacy":
            return
        put = kvaf.PutExecutionConfig(transport, executor)
        get = kvaf.GetExecutionConfig(transport, executor, algorithm)
        self.get_config = get
        self.runtime_kwargs = dict(
            per_call_execution=True, default_put=put, default_get=get,
            enabled_put_configs=[put], enabled_get_configs=[get],
            put_resources=[(put, kvaf.ExecutionResources(request_capacity=put_capacity))],
            get_resources=[(get, kvaf.ExecutionResources(request_capacity=get_capacity))],
        )
        self.put_kwargs = dict(execution=put)
        self.get_kwargs = dict(execution=get)

    def drain_get(self, runtime):
        # An external stream synchronize does not retire the dispatcher's lease.
        # Include this completion work in timed call-to-completion measurements.
        if self.mode == "per-call":
            runtime.drain_get()

    def bind_output(self, runtime, output, stream=None):
        if self.output_binding == "reused":
            if self.output_lease:
                raise RuntimeError("benchmark output is already bound")
            begin = time.perf_counter_ns()
            self.output_lease = runtime.begin_get_output(
                self.get_config, output, self.get_capacity, stream)
            self.binding_setup_us = (time.perf_counter_ns() - begin) / 1000
            self.get_kwargs["output_lease"] = self.output_lease
        self._print_binding("begin", self.binding_setup_us)

    def release_output(self, runtime):
        if self.output_lease:
            begin = time.perf_counter_ns()
            runtime.end_get_output(self.output_lease)
            self.binding_teardown_us = (time.perf_counter_ns() - begin) / 1000
            self.output_lease = 0
            self.get_kwargs.pop("output_lease")
        self._print_binding("end", self.binding_teardown_us)

    def _print_binding(self, operation, elapsed):
        print(f"PERF_OUTPUT_BINDING execution_mode={self.mode} output_binding={self.output_binding} "
              f"operation={operation} elapsed_us={elapsed:.3f}", flush=True)

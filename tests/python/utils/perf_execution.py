"""Configure benchmark providers, resource budgets and reusable buffer leases."""

import os
import time


def add_execution_metadata(parser):
    # Keep a stable label in reports, without exposing a second execution path.
    parser.set_defaults(execution_mode="unified")


def default_put_cores(transport, executor):
    return 16 if transport == "ubmem" else 1 if executor == "aicpu" else 2


class BenchmarkExecution:
    def __init__(self, kvaf, *, transport="urma", executor="aiv",
                 algorithm="direct", put_capacity=128, get_capacity=65536,
                 cores=0, jettys=2, get_kernel="", put_kernel="",
                 max_inflight_total=2048):
        if not 1 <= put_capacity <= 65536 or not 1 <= get_capacity <= 65536:
            raise ValueError("Put/Get request capacities must be in [1, 65536]")
        self.mode = "unified"
        self.put_capacity = put_capacity
        self.get_capacity = get_capacity
        self.input_lease = self.output_lease = 0
        self.binding_setup_us = self.binding_teardown_us = 0.0
        self.input_setup_us = self.input_teardown_us = 0.0
        reusable_input = transport == "urma" and executor in ("aiv", "aicpu")
        self.input_binding = ("not-needed" if not reusable_input
                              else "transient" if os.environ.get("KVAF_PERF_REUSE_PUT_INPUT", "1") == "0"
                              else "reused")
        supported = transport == "urma" and (executor, algorithm) in (("aiv", "direct"), ("aicpu", "gather"))
        self.output_binding = ("not-needed" if not supported
                               else "transient" if os.environ.get("KVAF_PERF_REUSE_GET_OUTPUT", "1") == "0"
                               else "reused")
        self.put_config = put = kvaf.PutExecutionConfig(transport, executor)
        self.get_config = get = kvaf.GetExecutionConfig(transport, executor, algorithm)
        put_budget = dict(cores=cores, request_capacity=put_capacity)
        get_budget = dict(cores=cores, request_capacity=get_capacity)
        if executor == "aiv":
            put_budget["kernel_path"] = put_kernel
            get_budget["kernel_path"] = get_kernel
            if transport == "urma":
                put_budget["jettys"] = min(jettys, 2)
                get_budget.update(jettys=jettys, max_inflight_total=max_inflight_total)
        self.runtime_kwargs = dict(
            transport_mode=transport, default_put=put, default_get=get,
            put_resources=[(put, kvaf.ExecutionResources(**put_budget))],
            get_resources=[(get, kvaf.ExecutionResources(**get_budget))],
        )
        self.put_kwargs = dict(execution=put)
        self.get_kwargs = dict(execution=get)

    def bind_input(self, runtime, source, stream=None):
        if self.input_binding == "reused":
            if self.input_lease:
                raise RuntimeError("benchmark input is already bound")
            begin = time.perf_counter_ns()
            self.input_lease = runtime.begin_put_input(
                self.put_config, source, self.put_capacity, stream)
            self.input_setup_us = (time.perf_counter_ns() - begin) / 1000
            self.put_kwargs["input_lease"] = self.input_lease
        self._print_input_binding("begin", self.input_setup_us)

    def release_input(self, runtime):
        if self.input_lease:
            begin = time.perf_counter_ns()
            runtime.end_put_input(self.input_lease)
            self.input_teardown_us = (time.perf_counter_ns() - begin) / 1000
            self.input_lease = 0
            self.put_kwargs.pop("input_lease")
        self._print_input_binding("end", self.input_teardown_us)

    def _print_input_binding(self, operation, elapsed):
        print(f"PERF_INPUT_BINDING execution_mode={self.mode} input_binding={self.input_binding} "
              f"operation={operation} elapsed_us={elapsed:.3f}", flush=True)

    def drain_get(self, runtime):
        runtime.drain_get()

    def complete_async_get(self, runtime):
        # Drain performs the device wait and retires framework references.
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

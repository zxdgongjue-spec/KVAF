"""Select the benchmark API path without changing its workload or providers."""


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

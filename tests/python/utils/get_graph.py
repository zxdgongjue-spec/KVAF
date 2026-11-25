"""Captured Get resources and input validation shared by graph benchmarks."""

from contextlib import contextmanager


class GraphInputs:
    def __init__(self, make_plan, miss_count, output, stream):
        self.make_plan = make_plan
        self.fixed = make_plan()
        self.miss_count = miss_count
        self.output = output
        self.stream = stream

    def prepare(self, count):
        for target, source in zip(self.fixed, self.make_plan()):
            target.copy_(source)
        self.miss_count.fill_(count)
        self.output.zero_()
        self.stream.synchronize()


class GetGraph:
    def __init__(self, graph, api, stream):
        self.graph, self.api, self.stream = graph, api, stream
        self.validated = False

    def replay(self):
        if self.api == "acl":
            self.graph.replay(self.stream)
        else:
            self.graph.replay()

    def replay_timed(self):
        # GraphInputs.prepare has already fenced the input producers.
        return self.graph.replay_timed()

    def synchronize(self):
        self.stream.synchronize()


@contextmanager
def captured_get(args, runtime, torch, kvaf, execution, output, stream, submit):
    """Use a real warmup forward to prepare the single-node benchmark graph."""
    execution.drain_get(runtime)
    execution.release_output(runtime)
    with captured_get_resources(args, runtime, torch, kvaf, execution, stream,
                                [submit]) as (graph, _):
        yield graph


@contextmanager
def captured_get_resources(args, runtime, torch, kvaf, execution, stream, submits):
    """Exercise automatic node collection without changing replay timing."""
    options = dict(execution.get_kwargs)
    options.pop("output_lease", None)
    options.pop("graph_lease", None)
    resources = runtime.graph_resources()
    graph = None
    capturing = False
    replay = None
    try:
        with torch.npu.stream(stream):
            with resources.prepare(stream):
                for submit in submits:
                    submit(options)
        graph = kvaf.AclGraph() if args.graph_api == "acl" else torch.npu.NPUGraph()
        replay = GetGraph(graph, args.graph_api, stream)
        replay.leases = resources.leases
        with resources.capture(stream):
            if args.graph_api == "acl":
                graph.capture_begin(stream)
                capturing = True
                for submit in submits:
                    submit(options)
                capturing = False
                graph.capture_end()
            else:
                with torch.npu.graph(graph, stream=stream):
                    for submit in submits:
                        submit(options)
        yield replay, resources
    finally:
        try:
            if capturing:
                graph.capture_end()
        finally:
            # Direct replay is intentionally owned by the caller. Never release
            # captured addresses before the stream and native graph are retired.
            stream.synchronize()
            if graph is not None:
                graph.reset()
            resources.close(discard_result=not (replay and replay.validated))


@contextmanager
def captured_get_nodes(args, runtime, torch, kvaf, execution, stream, nodes):
    """Keep explicit leases available for the per-node report regression."""
    options = dict(execution.get_kwargs)
    options.pop("output_lease", None)
    leases = []
    graph = None
    capturing = False
    replay = None

    def submit_nodes():
        for lease, (_, submit) in zip(leases, nodes):
            submit(dict(options, graph_lease=lease))

    try:
        for output, _ in nodes:
            leases.append(runtime.begin_get_graph(execution.get_config, output, args.request_length, stream))
        graph = kvaf.AclGraph() if args.graph_api == "acl" else torch.npu.NPUGraph()
        replay = GetGraph(graph, args.graph_api, stream)
        if args.graph_api == "acl":
            graph.capture_begin(stream)
            capturing = True
            submit_nodes()
            capturing = False  # AclGraph closes capture even when capture_end raises.
            graph.capture_end()
        else:
            with torch.npu.graph(graph, stream=stream):
                submit_nodes()
        yield replay, leases
    finally:
        try:
            if capturing:
                graph.capture_end()
        finally:
            stream.synchronize()
            if graph is not None:
                graph.reset()
            end_get_graph_leases(runtime, stream, leases, discard=not (replay and replay.validated))


def end_get_graph_leases(runtime, stream, leases, *, discard):
    """End all leases after graph destruction, retrying shared-domain discard."""
    remaining = list(leases)
    for _ in range(2 if discard else 1):
        failed = []
        for lease in remaining:
            try:
                runtime.end_get_graph(lease, stream, discard_result=discard)
            except Exception as error:
                failed.append((lease, error))
        if not failed:
            return
        remaining = [lease for lease, _ in failed]
    raise failed[0][1]


def validate_graph_inputs(graph, inputs, check, count, replays):
    """Exercise full, partial and zero prefixes before either timing mode."""
    counts = (count, count // 2, 0, min(1, count), count)
    for iteration in range(max(5, replays)):
        active = counts[iteration % len(counts)]
        inputs.prepare(active)
        graph.replay()
        graph.synchronize()
        check(inputs.fixed, active)

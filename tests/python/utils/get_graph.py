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

    def synchronize(self):
        self.stream.synchronize()


@contextmanager
def captured_get(args, runtime, torch, kvaf, execution, output, stream, submit):
    """Capture one Get, destroy the graph before releasing its resource lease."""
    execution.drain_get(runtime)
    execution.release_output(runtime)
    options = dict(execution.get_kwargs)
    options.pop("output_lease", None)
    lease = runtime.begin_get_graph(execution.get_config, output, args.request_length, stream)
    options["graph_lease"] = lease
    graph = None
    capturing = False
    replay = None
    try:
        graph = kvaf.AclGraph() if args.graph_api == "acl" else torch.npu.NPUGraph()
        replay = GetGraph(graph, args.graph_api, stream)
        if args.graph_api == "acl":
            graph.capture_begin(stream)
            capturing = True
            submit(options)
            capturing = False  # AclGraph closes capture even when capture_end raises.
            graph.capture_end()
        else:
            with torch.npu.graph(graph, stream=stream):
                submit(options)
        yield replay
    finally:
        try:
            if capturing:
                graph.capture_end()
        finally:
            stream.synchronize()
            if graph is not None:
                graph.reset()
            runtime.end_get_graph(lease, stream, discard_result=not (replay and replay.validated))


def validate_graph_inputs(graph, inputs, check, count, replays):
    """Exercise full, partial and zero prefixes before either timing mode."""
    counts = (count, count // 2, 0, min(1, count), count)
    for iteration in range(max(5, replays)):
        active = counts[iteration % len(counts)]
        inputs.prepare(active)
        graph.replay()
        graph.synchronize()
        check(inputs.fixed, active)

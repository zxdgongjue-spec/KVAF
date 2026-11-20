"""Collect and summarize BatchGet device tasks in shared replay windows."""

from argparse import SUPPRESS
from bisect import bisect_left
from decimal import Decimal
import json
import re


KERNEL_NAMES = {
    "ubmem": "KvafExecuteBatchGetUbmemKernelV2",
    "urma": "KvafExecuteBatchGetAivKernelV2",
}
PROFILER_EXPORT_TIMEOUT_SECONDS = 300


def add_profiler_args(parser, *, hidden=False, enabled_by_default=False):
    toggle = parser.add_mutually_exclusive_group()
    toggle.add_argument("--torch-npu-profiler", action="store_true",
                        help=SUPPRESS if hidden else "capture the kernel timeline (on by default outside queued mode)")
    toggle.add_argument("--no-torch-npu-profiler", dest="torch_npu_profiler", action="store_false",
                        help=SUPPRESS if hidden else "disable kernel timeline collection")
    parser.set_defaults(torch_npu_profiler=enabled_by_default)
    parser.add_argument("--profiler-warmup", type=int, default=1,
                        help=SUPPRESS if hidden else "replay rounds for profiler warmup (default: 1)")
    parser.add_argument("--profiler-active", type=int, default=3,
                        help=SUPPRESS if hidden else "replay rounds to profile (default: 3)")


def validate_profiler_args(args):
    if args.profiler_warmup < 0 or args.profiler_active < 1:
        return "--profiler-warmup must be nonnegative and --profiler-active must be positive"
    return None


class BatchGetProfiler:
    """Record a bounded number of synchronized replays and parse after the run."""

    def __init__(self, torch, directory, device, warmup, active):
        self.torch = torch
        self.directory = directory
        self.device = device
        self.warmup = warmup
        self.active = active

    def __enter__(self):
        import torch_npu.profiler as api
        from torch_npu.profiler.profiler import analyse

        self.analyse = analyse
        self.directory.mkdir()
        self.profiler = api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            schedule=api.schedule(wait=0, warmup=self.warmup, active=self.active, repeat=1),
            on_trace_ready=api.tensorboard_trace_handler(str(self.directory), analyse_flag=False),
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=api._ExperimentalConfig(
                profiler_level=api.ProfilerLevel.Level1, aic_metrics=api.AiCMetrics.AiCoreNone,
                export_type=api.ExportType.Text, data_simplification=False),
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, *exc):
        self.profiler.__exit__(*exc)

    def range(self, iteration, phase):
        return self.torch.profiler.record_function(f"KVAF/device{self.device}/round{iteration}/{phase}")

    def step(self):
        self.profiler.step()

    @property
    def total_steps(self):
        return self.warmup + self.active

    def export(self, replays):
        if replays >= self.total_steps:
            self.analyse(str(self.directory), max_process_number=1)


def event_interval(event):
    # Keep the original CANN-aligned device time base. Convert before subtracting.
    begin, duration = (int(Decimal(str(event[key])) * 1000) for key in ("ts", "dur"))
    if begin < 0 or duration <= 0:
        raise ValueError("profiler events require nonnegative timestamps and positive durations")
    return begin, begin + duration


def load_device_rounds(directory, device, kernel_name, calls_per_round, round_indices):
    traces = list(directory.rglob("trace_view.json"))
    if len(traces) != 1:
        raise ValueError(f"device {device}: expected one trace_view.json under {directory}, found {len(traces)}")
    with traces[0].open(encoding="utf-8") as source:
        document = json.load(source, parse_float=Decimal)
    if isinstance(document, dict) and document.get("baseTimeNanoseconds"):
        raise ValueError(f"device {device}: use original CANN trace timestamps, without rebasing")
    events = document if isinstance(document, list) else document["traceEvents"]
    marker = re.compile(rf"KVAF/device{device}/round([0-9]+)/(graph_replay|wait_completion)")
    markers, host_pids = {}, set()
    for event in events:
        match = marker.fullmatch(str(event.get("name", ""))) if event.get("ph") == "X" else None
        if match:
            iteration, phase = int(match[1]), match[2]
            phases = markers.setdefault(iteration, {})
            if phase in phases:
                raise ValueError(f"device {device}: duplicate {phase} marker in round {iteration}")
            phases[phase] = (*event_interval(event), event.get("pid"), event.get("tid"))
            host_pids.add(event.get("pid"))

    kernels = []
    for event in events:
        if event.get("ph") != "X" or kernel_name.lower() not in str(event.get("name", "")).lower():
            continue
        category = str(event.get("cat", "")).lower()
        task = str(event.get("args", {}).get("Task Type", "")).upper()
        if event.get("pid") in host_pids or category in {"cpu_op", "user_annotation", "python_function"}:
            continue
        if category not in {"kernel", "npu", "aiv"} and task not in {"AI_CORE", "AI_VECTOR_CORE", "AIV"}:
            continue
        kernels.append((*event_interval(event), event.get("pid"), event.get("tid")))
    kernels.sort(key=lambda kernel: kernel[0])
    starts = [kernel[0] for kernel in kernels]

    rounds = {}
    for iteration in round_indices:
        phases = markers.get(iteration, {})
        if set(phases) != {"graph_replay", "wait_completion"}:
            raise ValueError(f"device {device}: missing profiler markers for round {iteration} in {traces[0]}")
        submit, complete = phases["graph_replay"], phases["wait_completion"]
        if submit[2:] != complete[2:] or submit[1] > complete[0]:
            raise ValueError(f"device {device}: invalid profiler markers for round {iteration}")
        begin, end = submit[0], complete[1]
        tasks = kernels[bisect_left(starts, begin):bisect_left(starts, end)]
        if len(tasks) != calls_per_round or any(task[1] > end for task in tasks):
            raise ValueError(f"device {device}: round {iteration} needs {calls_per_round} complete "
                             f"{kernel_name} device tasks, found {len(tasks)} candidates in {traces[0]}")
        if len({task[2:] for task in tasks}) != 1 or any(left[1] > right[0] for left, right in zip(tasks, tasks[1:])):
            raise ValueError(f"device {device}: round {iteration} has overlapping tasks or multiple device streams")
        rounds[iteration] = [(task[0], task[1]) for task in tasks]
    return rounds


def overlap_bandwidth(timelines, round_indices, payload_bytes_per_get):
    selected = dict.fromkeys(timelines, 0)
    duration_ns = dict.fromkeys(timelines, 0)
    for iteration in round_indices:
        tasks_by_device = {device: rounds[iteration] for device, rounds in timelines.items()}
        common_begin = max(tasks[0][0] for tasks in tasks_by_device.values())
        common_end = min(tasks[-1][1] for tasks in tasks_by_device.values())
        for device, tasks in tasks_by_device.items():
            for begin, end in tasks:
                if common_begin <= begin and end <= common_end:
                    selected[device] += 1
                    duration_ns[device] += end - begin

    return {
        device: dict(overlap_selected_gets=selected[device], overlap_selected_kernel_ns=duration,
                     overlap_mean_GBps=selected[device] * payload_bytes_per_get / duration if duration else None)
        for device, duration in duration_ns.items()
    }

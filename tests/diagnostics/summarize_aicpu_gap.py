#!/usr/bin/env python3
"""Extract bounded, timing-only AICPU gap diagnostics using the standard library."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import gzip
import json
import math
from pathlib import Path
import re
import statistics
import sys


API_NAMES = (
    "aclrtLaunchKernelWithHostArgs", "aclrtLaunchCpuKernel", "aclrtLaunchKernel",
    "aclmdlRIExecuteAsync", "aclmdlExecuteAsync", "rtCpuKernelLaunch", "rtKernelLaunch",
    "rtModelExecute", "aclrtSynchronizeStreamWithTimeout", "aclrtSynchronizeStream",
    "aclrtSynchronizeEvent", "rtEventSynchronize", "aclrtRecordEvent", "rtEventRecord",
    "rtStreamSynchronizeWithTimeout", "rtStreamSynchronize", "aclrtMemcpyAsync",
    "aclrtMemcpy", "rtMemcpyAsync", "rtMemcpy",
)
GET_PATTERN = r"KvafAicpu(?:GatherRead|CpuPushScatter)Execution"
CORE_PROFILE = {
    "gap": {"e2e", "kernel", "e2e_minus_kernel", "python_boundary", "before_launch",
            "runtime_window", "runtime_outside_kernel", "after_completion", "intercall",
            "graph_event_outside_kernel", "graph_host_outside_event", "entry_outside_body", "e2e_outside_entry"},
    "device_boundary": {"entry_span", "entry_to_body", "body_to_return"},
    "host_binding": {"tensor_views", "stream", "set_device", "total", "gil_reacquire"},
    "host_stream": {"select", "handle"},
    "host_core": {"lifecycle", "table", "direction", "previous_drain", "validate", "dispatch", "finish", "other"},
    "host_dispatch": {"lock", "resolve", "output_check", "previous_drain", "context", "describe",
                      "acquire", "bind", "warmup", "provider", "context_release", "other"},
    "host_provider": {"prepare", "launch_api", "status_enqueue", "stream_sync", "status_check", "cleanup"},
    "host_binding_drain": {"set_device", "total", "gil_reacquire"},
    "host_graph": {"stream_resolve", "timing_prepare", "start_event_record", "replay_submit",
                   "end_event_record", "event_sync", "elapsed_time", "gil_reacquire", "native_total",
                   "binding_other", "replay_and_wait"},
    "stage": {"read", "ready_wait", "cq_wait", "scatter"},
}
CORE_PROFILE["host_graph_per_get"] = CORE_PROFILE["host_graph"]


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def number(value):
    try:
        value = float(str(value).replace(",", ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def pick(values, *keys):
    return next((values[k] for k in keys if k in values and values[k] not in (None, "")), None)


def identifier(value):
    if value is None or str(value).strip().lower() in ("", "n/a", "none", "-1"):
        return None
    try:
        return str(int(str(value), 0))
    except ValueError:
        return str(value)


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)

    return dict(n=len(ordered), mean=round(statistics.mean(ordered), 3), p50=percentile(.5),
                p99=percentile(.99), min=round(ordered[0], 3), max=round(ordered[-1], 3))


def api_name(name):
    return next((api for api in API_NAMES if re.search(re.escape(api) + r"(?:V\d+)?(?:\W|$)", name)), None)


def api_kind(api):
    key = norm(api)
    if "recordevent" in key or "eventrecord" in key:
        return "event_api"
    if "launch" in key or "execute" in key:
        return "launch"
    return "sync" if "synchronize" in key else "copy_api"


def interval_union(events):
    end, total = -math.inf, 0.0
    for event in sorted(events, key=lambda e: e.start):
        total += max(0.0, event.end - max(end, event.start))
        end = max(end, event.end)
    return total


def task_label(name, kind="aicpu"):
    key = norm(name)
    if "kvafaicpugatherread" in key:
        return "kvaf_gather"
    if "kvafaicpucpupushscatter" in key:
        return "kvaf_scatter"
    if "kvafaicpu" in key:
        return "kvaf_other"
    return kind


@dataclass
class Event:
    pid: str
    tid: str
    start: float
    duration: float
    kind: str
    label: str
    ids: dict
    direction: str = "unknown"
    target: bool = False

    @property
    def end(self):
        return self.start + self.duration

    @property
    def track(self):
        return self.pid, self.tid


def stream_evidence(first, event, stream_tracks):
    if first.pid != event.pid or (first.ids["device"] and event.ids["device"] and
                                   first.ids["device"] != event.ids["device"]):
        return None
    # Physical and logical Stream IDs are different namespaces. A known
    # physical identity also allows different logical/display tracks.
    for key in ("physical_stream", "stream"):
        if first.ids.get(key) is not None and event.ids.get(key) is not None:
            return key if first.ids[key] == event.ids[key] else None
    return "stream_track_metadata" if first.track == event.track and first.track in stream_tracks else None


def event_from_raw(raw, process_names, pattern):
    start, duration = number(raw.get("ts")), number(raw.get("dur"))
    if start is None or duration is None or duration < 0:
        return None
    args = {norm(k): v for k, v in (raw.get("args") or {}).items()}
    name = str(raw.get("name", ""))
    kernel = str(pick(args, "kernelname", "opname", "operatorname") or name)
    pid, tid = str(raw.get("pid", "")), str(raw.get("tid", ""))
    task_type = norm(pick(args, "tasktype", "optype") or "")
    role = process_names.get(pid, "")
    device_track = ("ascendhardware" in role or "npuhardware" in role or
                    role.startswith("device") or role in ("npu", "aicpu"))
    api = api_name(name)
    if api and not device_track:
        kind, label = api_kind(api), api
    elif "aicpu" in task_type or ("aicpu" in norm(name) and "cann" not in role):
        kind, label = "aicpu", task_label(kernel)
    elif any(key in task_type for key in ("eventrecord", "recordevent")):
        kind, label = "device_event", "device_event"
    elif ("memcpy" in norm(name) or "memcpy" in task_type) and (device_track or task_type):
        kind, label = "device_copy", "device_copy"
    else:
        kind, label = "other", "other"
    direction = norm(pick(args, "direction", "copykind", "memcpytype", "copytype") or name)
    direction = ("d2h" if any(s in direction for s in ("d2h", "devicetohost", "device2host")) else
                 "h2d" if any(s in direction for s in ("h2d", "hosttodevice", "host2device")) else "unknown")
    ids = {key: identifier(pick(args, *aliases)) for key, aliases in {
        "stream": ("streamid", "stream"), "physical_stream": ("physicstreamid", "physicalstreamid"),
        "task": ("taskid",), "model": ("modelid",),
        "device": ("deviceid", "device"), "connection": ("connectionid",),
        "correlation": ("correlationid", "correlation"), "event": ("eventid", "eventhandle", "event")}.items()}
    for key in ("connection", "correlation", "model"):
        if ids[key] in ("0", "18446744073709551615"):
            ids[key] = None
    return Event(pid, tid, start, duration, kind, label, ids, direction,
                 kind == "aicpu" and bool(pattern.search(kernel) or pattern.search(name)))


class Trace:
    def __init__(self, root, pattern):
        self.root = root
        self.events, self.flows = [], defaultdict(list)
        self.graph_replays, self.sample_rounds, self.selected_task_events = [], {}, []
        self.selected_csv_tasks = {}
        self.warnings, self.files = Counter(), 0
        self.stream_tracks = set()
        processes, raw_events, seen = {}, [], {}
        files = sorted(set(root.rglob("msprof_*.json")) | set(root.rglob("msprof_*.json.gz")))
        for source, path in enumerate(files):
            try:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8-sig") as stream:
                    document = json.load(stream)
                rows = document.get("traceEvents", []) if isinstance(document, dict) else document
                if not isinstance(rows, list):
                    self.warnings["TRACE_NOT_EVENT_ARRAY"] += 1
                    continue
                self.files += 1
                stack = defaultdict(list)
                for raw in rows:
                    if not isinstance(raw, dict):
                        continue
                    phase = raw.get("ph")
                    if phase == "M" and raw.get("name") == "process_name":
                        processes[str(raw.get("pid", ""))] = norm((raw.get("args") or {}).get("name", ""))
                    if phase == "M" and raw.get("name") == "thread_name" and re.fullmatch(
                            r"Stream\s+\d+", str((raw.get("args") or {}).get("name", "")).strip(), re.I):
                        self.stream_tracks.add((str(raw.get("pid", "")), str(raw.get("tid", ""))))
                    if phase in ("s", "t", "f"):
                        value = raw.get("id2", raw.get("id"))
                        if value is not None and number(raw.get("ts")) is not None:
                            local = isinstance(value, dict) and "local" in value
                            key = (str(raw.get("cat", "")), str(raw.get("name", "")), str(raw.get("scope", "")),
                                   json.dumps(value, sort_keys=True), str(raw.get("pid")) if local else "")
                            self.flows[key].append((phase, str(raw.get("pid", "")),
                                                    str(raw.get("tid", "")), float(raw["ts"])))
                        continue
                    track = str(raw.get("pid", "")), str(raw.get("tid", ""))
                    if phase == "B":
                        stack[track].append(raw)
                        continue
                    if phase == "E":
                        if not stack[track]:
                            self.warnings["UNMATCHED_END_EVENT"] += 1
                            continue
                        begin = stack[track].pop()
                        if number(raw.get("ts")) is None or number(begin.get("ts")) is None:
                            continue
                        raw = dict(begin, ph="X", dur=float(raw["ts"]) - float(begin["ts"]))
                        phase = "X"
                    if phase != "X":
                        continue
                    fingerprint = json.dumps(raw, sort_keys=True, separators=(",", ":"))
                    if fingerprint in seen and seen[fingerprint] != source:
                        self.warnings["DUPLICATE_SHARD_EVENT_REMOVED"] += 1
                        continue
                    seen[fingerprint] = source
                    raw_events.append(raw)
                self.warnings["UNMATCHED_BEGIN_EVENT"] += sum(map(len, stack.values()))
            except (OSError, ValueError, TypeError) as error:
                self.warnings["TRACE_READ_" + type(error).__name__] += 1
        for raw in raw_events:
            event = event_from_raw(raw, processes, pattern)
            if event:
                self.events.append(event)
        self.events.sort(key=lambda e: (e.start, -e.duration))
        self.tracks = defaultdict(list)
        for index, event in enumerate(self.events):
            self.tracks[event.track].append(index)
        self.starts = {track: [self.events[i].start for i in indexes] for track, indexes in self.tracks.items()}
        self.max_ends = {}
        for track, indexes in self.tracks.items():
            ends, maximum = [], -math.inf
            for i in indexes:
                maximum = max(maximum, self.events[i].end)
                ends.append(maximum)
            self.max_ends[track] = ends

    def at(self, track, timestamp, device=False):
        indexes = self.tracks.get(track, [])
        position = bisect_right(self.starts.get(track, []), timestamp + .001) - 1
        candidates = []
        while position >= 0 and self.max_ends[track][position] >= timestamp - .001:
            index = indexes[position]
            event = self.events[index]
            if event.start <= timestamp + .001 and event.end >= timestamp - .001:
                preferred = event.kind in (("aicpu", "device_copy", "device_event") if device else ("launch", "copy_api", "event_api"))
                candidates.append((not preferred, event.duration, index))
            position -= 1
        return min(candidates)[2] if candidates else None

    def outer_api(self, index, kind):
        event = self.events[index]
        candidates = [i for i in self.tracks[event.track] if self.events[i].kind == kind
                      and self.events[i].start <= event.start + .001 and self.events[i].end >= event.end - .001]
        if not candidates:
            return None
        return max(candidates, key=lambda i: (self.events[i].label.startswith("acl"), self.events[i].duration))

    def links(self):
        self.capture_sources = defaultdict(set)
        edges = defaultdict(set)
        for points in self.flows.values():
            previous = None
            for phase, pid, tid, timestamp in sorted(set(points), key=lambda p: (p[3], {"s": 0, "t": 1, "f": 2}[p[0]])):
                index = self.at((pid, tid), timestamp, phase == "f")
                if phase == "s":
                    previous = index
                elif index is not None and previous is not None:
                    edges[index].add(previous)
                    previous = index if phase == "t" else None
        result, api_ids = {}, defaultdict(list)
        model_starts = defaultdict(list)
        for event in self.events:
            if event.kind == "launch" and "execute" in norm(event.label):
                model_starts[event.track].append(event.start)
        def capture_only(host_index, event_index):
            host = self.events[host_index]
            event = self.events[event_index]
            if event.target and "launch" in norm(host.label):
                self.capture_sources[event_index].add(host_index)
            # A capture-time kernel ID does not identify a later model replay.
            starts = model_starts[host.track]
            position = bisect_right(starts, event.start)
            stale = (event.kind == "aicpu" and "launch" in norm(host.label) and
                     position > 0 and starts[position - 1] >= host.end)
            if stale:
                self.warnings["CAPTURE_LAUNCH_NOT_REPLAY_LINK"] += 1
            return stale
        device_ids = {e.ids["device"] for e in self.events
                      if e.kind in ("aicpu", "device_copy") and e.ids["device"] is not None}
        for index, event in enumerate(self.events):
            if event.kind not in ("launch", "copy_api", "event_api"):
                continue
            for key in ("connection", "correlation"):
                if event.ids[key] is not None:
                    api_ids[key, event.ids[key]].append(index)
            if event.ids["stream"] is not None and event.ids["task"] is not None:
                api_ids["stream_task", event.ids["stream"], event.ids["task"]].append(index)
        for index, event in enumerate(self.events):
            if event.kind not in ("aicpu", "device_copy", "device_event"):
                continue
            kind = {"aicpu": "launch", "device_copy": "copy_api", "device_event": "event_api"}[event.kind]
            pending, visited, hosts = list(edges[index]), set(), set()
            while pending:
                source = pending.pop()
                if source in visited:
                    continue
                visited.add(source)
                outer = self.outer_api(source, kind)
                if outer is not None:
                    hosts.add(outer)
                pending.extend(edges[source])
            if len(hosts) == 1:
                host_index = hosts.pop()
                if not capture_only(host_index, index):
                    result[index] = (host_index, "flow")
                    continue
            if len(hosts) > 1:
                self.warnings["AMBIGUOUS_FLOW_LINK"] += 1
                continue
            possible = set()
            for key in ("connection", "correlation"):
                if event.ids[key] is not None:
                    possible.update(api_ids[key, event.ids[key]])
            if event.kind == "device_event":
                unique = {}
                for key in ("connection", "correlation"):
                    if event.ids[key] is None:
                        continue
                    roots = {self.outer_api(i, "event_api") for i in api_ids[key, event.ids[key]]
                             if self.events[i].kind == "event_api"}
                    roots.discard(None)
                    if len(roots) == 1:
                        unique[next(iter(roots))] = key
                if len(unique) == 1:
                    host_index, method = next(iter(unique.items()))
                    host = self.events[host_index]
                    if ((len(device_ids) <= 1 or host.ids["device"] is not None) and
                            not (host.ids["device"] and event.ids["device"] and host.ids["device"] != event.ids["device"])):
                        # Identity is independent of Host/device clock alignment.
                        # Preserve the pair; flag clock order instead of losing it.
                        result[index] = (host_index, method + "_unique")
                        if event.start < host.start - 1:
                            self.warnings["UNIQUE_EVENT_ID_CLOCK_ORDER_WARNING"] += 1
                        continue
                elif len(unique) > 1:
                    self.warnings["CONFLICTING_UNIQUE_EVENT_IDS"] += 1
                    continue
            if event.ids["stream"] is not None and event.ids["task"] is not None:
                possible.update(api_ids["stream_task", event.ids["stream"], event.ids["task"]])
            candidates = []
            for host_index in possible:
                host = self.events[host_index]
                if host.kind != kind or host.start > event.start + 1:
                    continue
                if len(device_ids) > 1 and host.ids["device"] is None:
                    continue
                if host.ids["device"] is not None and event.ids["device"] is not None and host.ids["device"] != event.ids["device"]:
                    continue
                matching = [key for key in ("connection", "correlation")
                            if event.ids[key] is not None and event.ids[key] == host.ids[key]]
                if event.ids["task"] is not None and event.ids["stream"] is not None and all(
                        event.ids[key] == host.ids[key] for key in ("task", "stream")):
                    matching.append("stream_task")
                if matching:
                    if capture_only(host_index, index):
                        continue
                    candidates.append((host.start, host_index, matching[0]))
            if candidates:
                latest = max(row[0] for row in candidates)
                nearest = [row for row in candidates if row[0] == latest]
                if len(nearest) == 1:
                    _, host_index, method = nearest[0]
                    outer = self.outer_api(host_index, kind)
                    result[index] = (outer if outer is not None else host_index, method)
                else:
                    self.warnings["AMBIGUOUS_ID_LINK"] += 1
        return result

    def report(self, last, samples, explicit_pattern, execution="eager", gets_per_replay=1, timing_layout=None):
        if execution == "graph":
            return self.graph_report(last, samples, gets_per_replay, timing_layout)
        return self.eager_report(last, samples, explicit_pattern)

    def eager_report(self, last, samples, explicit_pattern):
        links = self.links()
        targets = [i for i, event in enumerate(self.events) if event.target]
        selection = "kernel_name"
        if not targets and not explicit_pattern:
            targets = [i for i, event in enumerate(self.events) if event.kind == "aicpu" and event.label != "kvaf_other"]
            selection = "all_aicpu_fallback_unverified"
        groups = defaultdict(list)
        for index in targets:
            if index in links:
                groups[links[index][0]].append(index)
        launches = sorted(groups, key=lambda i: self.events[i].start)
        selected = launches[-last:] if last else launches
        tasks = [self.events[i] for launch in selected for i in groups[launch]]
        if not groups:
            tasks = [self.events[i] for i in targets]
            if last:
                self.warnings["LAST_NOT_APPLIED_WITHOUT_LAUNCH_LINKS"] += 1
        chains = []
        for launch_index in selected:
            launch = self.events[launch_index]
            device_tasks = [self.events[i] for i in groups[launch_index]]
            next_launch = min((e.start for e in self.events if e.track == launch.track and e.kind == "launch"
                               and e.start > launch.start and e.start >= launch.end - .001), default=math.inf)
            syncs = [e for e in self.events if e.track == launch.track and e.kind == "sync"
                     and launch.end <= e.start < next_launch]
            if not syncs:
                continue
            sync = min(syncs, key=lambda e: (e.start, -e.duration))
            outer_syncs = [e for e in syncs if e.start <= sync.start and e.end >= sync.end and e.label.startswith("acl")]
            sync = max(outer_syncs, key=lambda e: e.duration) if outer_syncs else sync
            copy_indexes = [i for i, e in enumerate(self.events) if e.track == launch.track and e.kind == "copy_api"
                            and launch.end <= e.start and e.end <= sync.start]
            copy_indexes = sorted({self.outer_api(i, "copy_api") for i in copy_indexes})
            copies = [self.events[i] for i, (host, _) in links.items() if host in copy_indexes
                      and self.events[i].kind == "device_copy" and self.events[i].direction != "h2d"]
            task_start, task_end = min(e.start for e in device_tasks), max(e.end for e in device_tasks)
            relative = lambda e: [round(e.start - launch.start, 3), round(e.end - launch.start, 3)]
            metrics = dict(launch_api=launch.duration, launch_return_to_sync_begin=sync.start - launch.end,
                           sync_call=sync.duration, launch_to_sync_return=sync.end - launch.start,
                           task_envelope=task_end - task_start,
                           task_union=interval_union(device_tasks),
                           inter_task_gaps=task_end - task_start - interval_union(device_tasks),
                           aligned_task_end_to_sync_return=sync.end - task_end,
                           aligned_launch_to_task_begin=task_start - launch.start)
            chain = dict(launch_ordinal=launches.index(launch_index) + 1, workers_or_task_events=len(device_tasks),
                         launch_api=launch.label, sync_api=sync.label,
                         link_methods=sorted({links[i][1] for i in groups[launch_index]}),
                         host_launch_us=relative(launch), device_task_us=[round(task_start - launch.start, 3),
                         round(task_end - launch.start, 3)], host_sync_us=relative(sync),
                         host_copy_enqueue_us=[relative(self.events[i]) for i in copy_indexes[:4]])
            if "execute" in norm(launch.label):
                previous_launch = max((e.end for e in self.events if e.track == launch.track and
                                       e.kind == "launch" and e.end <= launch.start), default=-math.inf)
                records = [i for i, e in enumerate(self.events) if e.track == launch.track and e.kind == "event_api"]
                before = [i for i in records if previous_launch <= self.events[i].start and self.events[i].end <= launch.start]
                after = [i for i in records if launch.end <= self.events[i].start < sync.start]
                if before and after:
                    record_hosts = [max(before, key=lambda i: self.events[i].start), min(after, key=lambda i: self.events[i].start)]
                    records = [[self.events[i] for i, (host, _) in links.items()
                                if host == record and self.events[i].kind == "device_event"] for record in record_hosts]
                    if all(len(events) == 1 for events in records):
                        first, final = records[0][0], records[1][0]
                        chain["device_event_bracket_us"] = [relative(first), relative(final)]
                        metrics.update(device_event_start_to_task=task_start - first.end,
                                       device_task_to_end_event=final.start - task_end)
                        chain["event_bracket_order_ok"] = first.end <= task_start + 1 and task_end <= final.start + 1
            if copies:
                copy_begin, copy_end = min(e.start for e in copies), max(e.end for e in copies)
                metrics.update(device_copy_envelope=copy_end - copy_begin,
                               device_task_end_to_copy_begin=copy_begin - task_end,
                               aligned_copy_end_to_sync_return=sync.end - copy_end)
                chain["device_copy_us"] = [round(copy_begin - launch.start, 3), round(copy_end - launch.start, 3)]
                chain["copy_direction"] = "d2h" if all(e.direction == "d2h" for e in copies) else "unknown"
            chain["cross_clock_order_ok"] = (task_start >= launch.start - 1 and task_end <= sync.end + 1 and
                                              (not copies or (copy_begin >= task_end - 1 and copy_end <= sync.end + 1)))
            chain["metrics_us"] = {key: round(value, 3) for key, value in metrics.items()}
            chains.append(chain)
        result = self.chain_report(targets, groups, selected, chains, tasks, selection, samples)
        result["execution"] = "eager"
        return result

    def graph_report(self, last, samples, gets_per_replay, timing_layout=None):
        links = self.links()
        targets = [i for i, event in enumerate(self.events) if event.target]
        roots = {self.outer_api(i, e.kind) for i, e in enumerate(self.events)
                 if e.kind in ("launch", "event_api", "sync")}
        roots.discard(None)
        by_process = defaultdict(list)
        records = defaultdict(list)
        models = []
        outer_end = {}
        for i in sorted(roots, key=lambda i: (self.events[i].start, -self.events[i].duration)):
            event = self.events[i]
            # Ignore nested Runtime calls, including sync inside another API kind.
            if event.end <= outer_end.get(event.track, -math.inf):
                continue
            outer_end[event.track] = event.end
            by_process[event.pid].append(i)
            if event.kind == "launch" and "execute" in norm(event.label):
                models.append(i)
        for device, (host, _) in links.items():
            if self.events[device].kind == "device_event":
                records[host].append(device)
        rejects, groups, candidates = Counter(), {}, {}
        match_counts = {key: Counter() for key in ("same_stream", "same_device_inner", "same_device_overlap")}
        match_examples = []
        internal_stream_matches = 0
        timed_candidates = 0
        targets_by_device = defaultdict(list)
        for i in targets:
            targets_by_device[self.events[i].pid].append(i)
        target_starts = {pid: [self.events[i].start for i in indexes] for pid, indexes in targets_by_device.items()}
        occupied = set()
        for model in models:
            launch = self.events[model]
            # The benchmark records Events immediately around one replay. Allow
            # serialized Torch Host-queue threads, but no intervening submission.
            hosts = by_process[launch.pid]
            position = hosts.index(model)
            if position == 0 or position + 2 >= len(hosts):
                rejects["missing_host_bracket"] += 1
                continue
            start_host, end_host, sync_host = hosts[position - 1], hosts[position + 1], hosts[position + 2]
            start, end, sync = (self.events[i] for i in (start_host, end_host, sync_host))
            if (start.kind != "event_api" or end.kind != "event_api" or sync.kind != "sync" or
                    "event" not in norm(sync.label) or
                    not start.end <= launch.start <= launch.end <= end.start <= end.end <= sync.start):
                rejects["not_timed_record_model_record_event_sync"] += 1
                continue
            timed_candidates += 1
            if end.ids.get("event") and sync.ids.get("event") and end.ids["event"] != sync.ids["event"]:
                rejects["end_event_identity_mismatch"] += 1
                continue
            if len(records[start_host]) != 1 or len(records[end_host]) != 1:
                rejects["missing_or_ambiguous_device_event_links"] += 1
                continue
            first, final = self.events[records[start_host][0]], self.events[records[end_host][0]]
            if first.end > final.start:
                rejects["device_event_stream_or_order_mismatch"] += 1
                continue
            record_stream = stream_evidence(first, final, self.stream_tracks)
            if not record_stream:
                rejects["unknown_or_inconsistent_device_stream"] += 1
                continue
            def same_stream(event):
                return bool(stream_evidence(first, event, self.stream_tracks))
            lane = self.tracks[first.track]
            begin = bisect_right(self.starts[first.track], first.end - .001)
            stop = bisect_right(self.starts[first.track], final.start + .001)
            window = lane[begin:stop]
            position = begin - 1
            crosses_start = False
            while position >= 0 and self.max_ends[first.track][position] > first.end + .001:
                event = self.events[lane[position]]
                crosses_start |= event.kind in ("aicpu", "device_copy") and event.end > first.end + .001
                position -= 1
            if crosses_start:
                rejects["device_work_crosses_start_event"] += 1
                continue
            device_indexes = targets_by_device[first.pid]
            low = bisect_right(target_starts.get(first.pid, []), first.end - .001)
            high = bisect_right(target_starts.get(first.pid, []), final.start + .001)
            inside = [i for i in device_indexes[low:high] if self.events[i].end <= final.start + .001 and
                      not (first.ids["device"] and self.events[i].ids["device"] and
                           first.ids["device"] != self.events[i].ids["device"])]
            task_ids = [i for i in inside if same_stream(self.events[i])]
            overlaps = [i for i in device_indexes if self.events[i].start < final.end and self.events[i].end > first.start]
            for key, count in (("same_stream", len(task_ids)), ("same_device_inner", len(inside)),
                               ("same_device_overlap", len(overlaps))):
                match_counts[key][str(count)] += 1
            def captured_origin(i):
                return any(self.events[h].pid == launch.pid and self.events[h].end <= start.start
                           for h in self.capture_sources[i])
            def model_origin(i):
                return (links.get(i, (None,))[0] == model or
                        bool(launch.ids.get("model") and launch.ids["model"] == self.events[i].ids.get("model")))
            def stream_relation(i):
                event = self.events[i]
                def equal_id(key):
                    return None if first.ids.get(key) is None or event.ids.get(key) is None else first.ids[key] == event.ids[key]
                return dict(same_trace_track=first.track == event.track,
                            physical_stream_equal=equal_id("physical_stream"), logical_stream_equal=equal_id("stream"),
                            captured_origin_same_host_process=captured_origin(i), model_origin=model_origin(i))
            method = "event_bracket_same_stream"
            if len(task_ids) != gets_per_replay and len(inside) == gets_per_replay:
                # Graph nodes may use internal streams. Under this benchmark's
                # serial replay contract, require capture/model provenance as
                # well as the unique expected task set inside the Event window.
                if all(model_origin(i) or captured_origin(i) for i in inside):
                    task_ids = inside
                    method = "event_bracket_captured_get_internal_stream"
            if len(task_ids) != gets_per_replay:
                rejects["unexpected_get_task_count"] += 1
                if len(match_examples) < samples:
                    relative = lambda e: [round(e.start - launch.start, 3), round(e.end - launch.start, 3)]
                    match_examples.append(dict(
                        model_ordinal=models.index(model) + 1, expected_gets=gets_per_replay,
                        device_event_bracket_us=[relative(first), relative(final)],
                        candidates=[dict(stream_relation(i), interval_us=relative(self.events[i]),
                            inside_inner_window=i in inside) for i in overlaps[:8]]))
                continue
            device_tasks = [self.events[i] for i in task_ids]
            if occupied.intersection(task_ids):
                rejects["task_in_multiple_replays"] += 1
                continue
            if any(self.events[i].kind in ("aicpu", "device_copy") and not self.events[i].target
                   and same_stream(self.events[i]) for i in window):
                rejects["unrelated_device_work_in_bracket"] += 1
                continue
            if any(self.events[i].kind in ("aicpu", "device_copy") and self.events[i].end > final.start + .001
                   for i in window):
                rejects["device_work_crosses_end_event"] += 1
                continue
            occupied.update(task_ids)
            groups[model] = task_ids
            internal_stream_matches += method == "event_bracket_captured_get_internal_stream"
            task_begin, task_end = min(e.start for e in device_tasks), max(e.end for e in device_tasks)
            union = interval_union(device_tasks)
            metrics = dict(
                launch_api=launch.duration, sync_call=sync.duration,
                launch_return_to_sync_begin=sync.start - launch.end,
                launch_to_sync_return=sync.end - launch.start,
                host_record_begin_to_sync_return=sync.end - start.start,
                host_start_event_record=start.duration, host_end_event_record=end.duration,
                task_envelope=task_end - task_begin, task_union=union,
                task_sum=sum(e.duration for e in device_tasks),
                task_overlap=sum(e.duration for e in device_tasks) - union,
                inter_task_gaps=task_end - task_begin - union,
                device_event_record_inner_window=final.start - first.end,
                device_event_record_outer_window=final.end - first.start,
                device_event_start_to_task=task_begin - first.end,
                device_task_to_end_event=final.start - task_end,
                device_end_event_record=final.duration,
                aligned_start_record_begin_to_device_event_end=first.end - start.start,
                aligned_end_event_end_to_sync_return=sync.end - final.end,
                aligned_launch_to_task_begin=task_begin - launch.start,
                aligned_task_end_to_sync_return=sync.end - task_end)
            metrics.update({key + "_per_get": metrics[key] / gets_per_replay for key in (
                "task_union", "task_sum", "inter_task_gaps", "device_event_record_inner_window",
                "device_event_record_outer_window", "device_event_start_to_task", "device_task_to_end_event")})
            relative = lambda e: [round(e.start - launch.start, 3), round(e.end - launch.start, 3)]
            candidates[model] = dict(
                launch_api=launch.label, sync_api=sync.label, workers_or_task_events=len(task_ids),
                link_methods=[method], stream_identity=record_stream,
                task_stream_relations=[stream_relation(i) for i in task_ids],
                event_link_methods=sorted({links[records[h][0]][1] for h in (start_host, end_host)}),
                host_order="thread" if start.track == launch.track == end.track == sync.track else "process_serial",
                host_launch_us=relative(launch), host_sync_us=relative(sync),
                host_event_records_us=[relative(start), relative(end)],
                device_event_bracket_us=[relative(first), relative(final)],
                device_task_us=[round(task_begin - launch.start, 3), round(task_end - launch.start, 3)],
                device_task_intervals_us=[relative(e) for e in device_tasks],
                cross_clock_order_ok=(first.start >= start.start - 1 and launch.start <= task_begin + 1 and
                                      task_end <= sync.end + 1 and final.end <= sync.end + 1),
                metrics_us={key: round(value, 3) for key, value in metrics.items()})
        self.graph_replays = [dict(model=model, tasks=[self.events[i] for i in task_ids],
            host_start_us=self.events[model].start + candidates[model]["host_event_records_us"][0][0],
            host_end_us=self.events[model].start + candidates[model]["host_sync_us"][1],
            event_low_us=candidates[model]["metrics_us"]["device_event_record_inner_window"],
            event_high_us=candidates[model]["metrics_us"]["device_event_record_outer_window"])
            for model, task_ids in groups.items()]
        formal = list(groups)
        selection_status = "no_sample_metadata"
        if timing_layout is not None:
            prefix, rounds = timing_layout["timed_prefix_replays"], len(timing_layout["rounds"])
            if len(formal) == prefix + rounds and gets_per_replay == timing_layout["gets_per_replay"]:
                formal = formal[prefix:]
                self.sample_rounds = {model: ordinal for ordinal, model in enumerate(formal, 1)}
                selection_status = "formal_rounds_by_serial_replay_contract"
            else:
                selection_status = "sample_trace_replay_count_mismatch"
        selected = formal[-last:] if last else formal
        if not selected:
            self.warnings["NO_VALIDATED_GRAPH_REPLAY_BRACKETS"] += 1
        if last and len(selected) < last:
            self.warnings["GRAPH_REPLAYS_FEWER_THAN_REQUESTED"] += last - len(selected)
        chains = [dict(candidates[i], launch_ordinal=models.index(i) + 1) for i in selected]
        tasks = [self.events[i] for model in selected for i in groups[model]]
        result = self.chain_report(targets, groups, selected, chains, tasks, "kernel_name", samples)
        result.update(execution="graph", association="validated_device_event_containment",
                      association_contract="serialized_batch_get_benchmark_replay",
                      model_execute_calls=len(models), timed_replay_brackets=len(groups),
                      timed_replay_candidates=timed_candidates, requested_replays=last,
                      selected_replays=len(selected), expected_gets_per_replay=gets_per_replay,
                      replay_rejections=dict(rejects), linked_device_event_records=sum(map(len, records.values())),
                      internal_stream_replays=internal_stream_matches,
                      sample_selection=selection_status,
                      formal_timed_replays=len(self.sample_rounds),
                      excluded_timing_warmup_replays=(timing_layout["timed_prefix_replays"] if self.sample_rounds else 0),
                      graph_match_diagnostics={key: dict(value) for key, value in match_counts.items()},
                      graph_match_examples=match_examples,
                      metrics_scope="per_replay; fields ending _per_get are divided by Get count")
        return result

    def chain_report(self, targets, groups, selected, chains, tasks, selection, samples):
        # Replayed CSV Task IDs can repeat. Filter only unambiguous identities.
        occurrences = Counter(e.ids["task"] for e in self.events if e.kind == "aicpu" and e.ids["task"] is not None)
        self.selected_task_events = tasks
        self.selected_csv_tasks = {e.ids["task"]: e.ids for e in tasks if occurrences[e.ids["task"]] == 1}
        if len(self.selected_csv_tasks) < len(tasks):
            self.warnings["SELECTED_CSV_IDS_MISSING_OR_REUSED"] += len(tasks) - len(self.selected_csv_tasks)
        ordered = sorted(chains, key=lambda c: c["metrics_us"]["launch_to_sync_return"])
        positions = sorted({round(i * (len(ordered) - 1) / max(1, samples - 1)) for i in range(samples)}) if ordered else []
        apis = defaultdict(list)
        for event in self.events:
            if event.kind in ("launch", "sync", "copy_api", "event_api"):
                apis[event.label].append(event.duration)
        coverage = {}
        for kind in ("launch", "copy_api", "event_api", "aicpu", "device_copy", "device_event"):
            events = [e for e in self.events if e.kind == kind]
            coverage[kind] = dict(events=len(events), **{
                key: sum(e.ids[key] is not None for e in events)
                for key in ("stream", "physical_stream", "model", "task", "device", "connection", "correlation")})
        return dict(json_files=self.files, complete_events=len(self.events), target_selection=selection,
                    flow_groups=len(self.flows), link_field_coverage=coverage,
                    target_events=len(targets), linked_target_events=sum(map(len, groups.values())),
                    linked_launches=len(groups), selected_launches=len(selected), complete_chains=len(chains),
                    chains_with_device_copy=sum("device_copy_us" in c for c in chains),
                    cross_clock_order_warnings=sum(not c["cross_clock_order_ok"] for c in chains),
                    target_task_duration_us=stats([e.duration for e in tasks]),
                    chain_stats_us={key: stats([c["metrics_us"][key] for c in chains if key in c["metrics_us"]
                                               and (not key.startswith("aligned_") or c["cross_clock_order_ok"])])
                                    for key in sorted({k for c in chains for k in c["metrics_us"]})},
                    representative_chains=[ordered[i] for i in positions],
                    all_trace_api_us={key: stats(value) for key, value in sorted(apis.items())},
                    warnings={k: v for k, v in sorted(self.warnings.items()) if v})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def benchmark_report(root):
    if root is None:
        return {}
    result = defaultdict(dict)
    for path in sorted(root.rglob("summary.csv")):
        for row in read_csv(path):
            if row.get("backend") not in ("gather_read", "cpu_push_scatter") or row.get("execution") not in ("eager", "graph"):
                continue
            key = row["backend"] + "/" + row["execution"]
            if "summary" in result[key]:
                raise ValueError("INPUT_BENCHMARK_MULTIPLE_RUNS")
            result[key]["summary"] = {k: round(number(v), 3) for k, v in row.items() if number(v) is not None
                and (k in ("miss_count", "token_bytes", "aicpu_workers", "gets_per_replay") or
                     k in ("e2e_mean_us", "kernel_mean_us", "e2e_p99_us", "kernel_p99_us", "e2e_mean_GBps",
                           "host_replay_mean_us", "host_replay_p99_us", "host_replay_mean_GBps"))}
            result[key]["configuration"] = {k: row[k] for k in ("get_binding", "graph_api", "graph_submit", "latency_scope")
                if row.get(k) in ("ordinary", "prepared", "capture", "torch", "acl", "python", "native",
                                 "single_get", "amortized_per_get")}
    for path in sorted(root.rglob("profile.csv")):
        for row in read_csv(path):
            if row.get("backend") not in ("gather_read", "cpu_push_scatter") or row.get("execution") not in ("eager", "graph"):
                continue
            kind, name = row.get("kind"), row.get("name")
            if name not in CORE_PROFILE.get(kind, set()) or row.get("metric") not in ("mean_us", "p50_us", "p99_us"):
                continue
            value = number(row.get("value"))
            if value is not None:
                key = row["backend"] + "/" + row["execution"]
                result[key].setdefault("profile_us", {}).setdefault(kind + "." + name, {})[row["metric"]] = round(value, 3)
    return dict(result)


def report_execution(benchmark, requested):
    modes = {key.rsplit("/", 1)[1] for key in benchmark}
    if requested == "auto":
        if len(modes) > 1:
            raise ValueError("INPUT_MIXED_EXECUTION: specify --execution eager or graph")
        requested = next(iter(modes), "eager")
    if modes and requested not in modes:
        raise ValueError("INPUT_EXECUTION_MISMATCH")
    counts = {item.get("summary", {}).get("gets_per_replay", 1)
              for key, item in benchmark.items() if key.endswith("/" + requested)}
    if len(counts) > 1:
        raise ValueError("INPUT_MIXED_GET_COUNTS")
    count = next(iter(counts), 1)
    if count != int(count) or not 1 <= count <= 16:
        raise ValueError("INPUT_INVALID_GET_COUNT")
    return requested, int(count)


def graph_scope_comparison(benchmark, timeline):
    if timeline.get("execution") != "graph":
        return
    selected = timeline["selected_replays"]
    total = (timeline["formal_timed_replays"] if timeline.get("sample_selection") ==
             "formal_rounds_by_serial_replay_contract" else timeline.get("timed_replay_candidates", selected))
    complete = (selected > 0 and selected == total and
                (not timeline.get("requested_replays") or selected >= timeline["requested_replays"]))
    timeline["scope_comparison_selection_complete"] = complete
    if not complete:
        timeline["scope_comparison_note"] = "Skipped task-minus-body/entry mean differences: selected trace replays do not cover the complete timed selection."
        return
    trace_stats = timeline["chain_stats_us"]
    task = trace_stats.get("task_sum_per_get", {}).get("mean")
    inner = trace_stats.get("device_event_record_inner_window_per_get", {}).get("mean")
    outer = trace_stats.get("device_event_record_outer_window_per_get", {}).get("mean")
    if task is None:
        return
    comparisons = {}
    for key, item in benchmark.items():
        if not key.endswith("/graph"):
            continue
        profile = item.get("profile_us", {})
        body = item.get("summary", {}).get("kernel_mean_us")
        entry = profile.get("device_boundary.entry_span", {}).get("mean_us")
        event = item.get("summary", {}).get("e2e_mean_us")
        values = {"trace_task_mean_per_get": task,
                  "trace_event_record_window_low_per_get": inner,
                  "trace_event_record_window_high_per_get": outer}
        if body is not None:
            values["trace_task_minus_benchmark_body"] = round(task - body, 3)
        if entry is not None:
            values["trace_task_minus_benchmark_entry"] = round(task - entry, 3)
        if event is not None:
            values["benchmark_event_mean"] = event
        comparisons[key] = values
    timeline["scope_mean_comparison_us"] = comparisons
    timeline["scope_comparison_note"] = "Selected trace means versus benchmark means; not paired per-call body/task timers. Record-task windows approximate hardware Event timestamps."


def require(condition):
    if not condition:
        raise ValueError


def load_graph_samples(root):
    if root is None:
        return None
    paths = list(root.rglob("aicpu_samples.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError("INPUT_MULTIPLE_SAMPLE_FILES")
    data = json.loads(paths[0].read_text(encoding="utf-8-sig"))
    integer = lambda v: isinstance(v, int) and not isinstance(v, bool)
    try:
        require(data["format"] in ("kvaf-aicpu-samples-v1", "kvaf-aicpu-samples-v2", "kvaf-aicpu-samples-v3") and data["execution"] == "graph")
        counters = data["format"] == "kvaf-aicpu-samples-v2"
        if counters:
            require(data["boundary_clock"] == "aarch64_cntvct")
        require(data["device_clock"] == ("aicpu_monotonic_raw_ns" if data["format"] == "kvaf-aicpu-samples-v3" else "aicpu_steady_ns"))
        require(data["host_clock"] == "python_perf_counter_ns")
        require(data["backend"] in ("gather_read", "cpu_push_scatter"))
        require(data["graph_submit"] in ("native", "python") and data["graph_api"] in ("acl", "torch"))
        require(integer(data["device_id"]) and data["device_id"] >= 0)
        require(integer(data["iterations"]) and data["iterations"] == len(data["rounds"]) > 0)
        require(integer(data["warmup"]) and data["warmup"] >= 0)
        require(integer(data["gets_per_replay"]) and 1 <= data["gets_per_replay"] <= 16)
        require(integer(data["aicpu_workers"]) and 1 <= data["aicpu_workers"] <= 8)
        require(data["timed_prefix_replays"] == (data["warmup"] + 1 if data["graph_submit"] == "native" else 0))
        previous_host, previous_device = 0, 0
        for ordinal, sample in enumerate(data["rounds"], 1):
            require(sample["round"] == ordinal)
            require(all(integer(sample[k]) for k in ("host_begin_ns", "host_end_ns", "event_elapsed_ns")))
            require(previous_host <= sample["host_begin_ns"] < sample["host_end_ns"])
            require(sample["event_elapsed_ns"] > 0)
            previous_host = sample["host_end_ns"]
            require(len(sample["nodes"]) == data["gets_per_replay"])
            for node_index, node in enumerate(sample["nodes"]):
                require(node["node"] == node_index and len(node["workers"]) == data["aicpu_workers"])
                for worker_index, worker in enumerate(node["workers"]):
                    require(worker["worker"] == worker_index)
                    values = [worker[k] for k in ("device_entry_ns", "device_start_ns", "device_finish_ns", "device_return_ns")]
                    require(all(integer(v) for v in values) and 0 < values[0] <= values[1] <= values[2] <= values[3])
                    if counters:
                        ticks = [worker[k] for k in ("device_entry_syscnt", "device_return_syscnt", "device_syscnt_frequency")]
                        require(all(integer(v) for v in ticks))
                        require(ticks == [0, 0, 0] or (0 < ticks[0] <= ticks[1] and ticks[2] > 0))
                require(previous_device <= min(w["device_entry_ns"] for w in node["workers"]))
                previous_device = max(w["device_return_ns"] for w in node["workers"])
    except (ValueError, KeyError, TypeError):
        raise ValueError("INPUT_INVALID_GRAPH_SAMPLES") from None
    return data


def calibrated_clock(path, device_id, source_clock="aicpu_steady_ns"):
    """An independent calibration supplies two bracketing clock anchors."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    try:
        require(data["format"] == "kvaf-aicpu-clock-map-v1" and data["device_id"] == device_id)
        require(data["source_clock"] == source_clock and data["target_clock"] == "msprof_trace_us")
        require(data["provenance"] in ("msprof_clock_sync", "independent_timestamp_calibration"))
        anchors = data["anchors"]
        require(len(anchors) == 2)
        require(all(isinstance(a["device_ns"], int) and not isinstance(a["device_ns"], bool) for a in anchors))
        require(all(isinstance(a[key], (float, int)) and not isinstance(a[key], bool) and number(a[key]) is not None
                    for a in anchors for key in ("trace_us", "uncertainty_us")))
        require(all(a["uncertainty_us"] >= 0 for a in anchors))
        first, last = anchors
        span = last["device_ns"] - first["device_ns"]
        require(span > 0 and last["trace_us"] > first["trace_us"])
    except (ValueError, KeyError, TypeError):
        raise ValueError("INPUT_INVALID_CLOCK_MAP") from None

    def convert(value):
        if not first["device_ns"] <= value <= last["device_ns"]:
            raise ValueError("INPUT_CLOCK_MAP_DOES_NOT_COVER_SAMPLES")
        fraction = (value - first["device_ns"]) / span
        return (first["trace_us"] + fraction * (last["trace_us"] - first["trace_us"]),
                (1 - fraction) * first["uncertainty_us"] + fraction * last["uncertainty_us"])
    return convert


def msprof_device_clock(root, tasks, frequency, diagnostics=None, raw_clock=False):
    """Use independent capture clock anchors, never fitted task boundaries.

    Mirrors msprof InfoConfReader.get_delta_time/trans_into_local_time:
    https://github.com/Ascend/msprof/blob/master/analysis/common_func/info_conf_reader.py
    """
    if root.name == "mindstudio_profiler_output":
        root = root.parent

    def is_metadata(path):
        return path.is_file() and path.suffix != ".done"

    directories = sorted({p.parent for p in root.rglob("dev_start.log*") if is_metadata(p)})
    devices = {p.ids.get("device") for p in tasks} - {None}
    if devices:
        directories = [p for p in directories if p.name.removeprefix("device_") in devices]
    if not directories:
        raise ValueError("sync_metadata_missing")
    if len(directories) != 1:
        raise ValueError("sync_metadata_ambiguous_device")
    directory = directories[0]

    def metadata(pattern):
        paths = [p for p in directory.glob(pattern) if is_metadata(p)]
        if len(paths) != 1:
            if diagnostics is not None:
                diagnostics.update(required_file_pattern=pattern,
                    matches_in_counter_directory=len(paths),
                    matches_in_capture=sum(is_metadata(p) for p in root.rglob(pattern)))
            raise ValueError("sync_metadata_missing_or_ambiguous")
        return paths[0].read_text(encoding="utf-8-sig")

    def numeric_fields(text):
        fields = re.findall(r"^\s*([a-zA-Z_]+)\s*[:=]\s*([-+\d.eE]+)\s*$", text, re.M)
        if len({key.lower() for key, _ in fields}) != len(fields):
            raise ValueError("sync_ambiguous_metadata_fields")
        return {key.lower(): Decimal(value) for key, value in fields}

    try:
        dev = numeric_fields(metadata("dev_start.log*"))
        host = numeric_fields(metadata("host_start.log*"))
        info = json.loads(metadata("info.json*"), parse_float=Decimal)
        start = json.loads(metadata("start_info*"), parse_float=Decimal)
        if raw_clock:
            device_frequency = Decimal(1000000000)
            counter_origin = dev["clock_monotonic_raw"]
        else:
            device_frequency = Decimal(str(info["DeviceInfo"][0]["hwts_frequency"])) * 1000000
            if device_frequency <= 0 or abs(device_frequency - frequency) > 1:
                if diagnostics is not None:
                    diagnostics.update(sampled_counter_frequency_hz=frequency,
                        metadata_hwts_frequency_hz=float(device_frequency), requires_raw_clock_samples=True)
                raise ValueError("sync_counter_frequency_mismatch")
            counter_origin = dev["cntvct"]
        host_ns = host["clock_monotonic_raw"]
        if host.get("cntvct_diff", 0):
            host_frequency = Decimal(str(info["CPU"][0]["Frequency"])) * 1000000
            if host_frequency <= 0:
                raise ValueError("sync_invalid_host_frequency")
            host_ns += host["cntvct_diff"] * 1000000000 / host_frequency
        local_offset_us = Decimal(str(start["collectionTimeBegin"])) - Decimal(str(start["clockMonotonicRaw"])) / 1000
        trace_origin_us = host_ns / 1000 + local_offset_us
        if not all(v.is_finite() for v in (counter_origin, trace_origin_us, device_frequency)) or counter_origin <= 0:
            raise ValueError("sync_invalid_metadata")
    except (KeyError, IndexError, TypeError, InvalidOperation, ZeroDivisionError):
        raise ValueError("sync_invalid_metadata") from None

    def convert(counter):
        return trace_origin_us + (Decimal(counter) - counter_origin) * 1000000 / device_frequency, 0.0
    return convert


def entry_boundary_report(trace, timeline, data, clock_map=None, sample_limit=5):
    if data is None:
        return dict(status="samples_missing", note="Rerun the benchmark with --kernel-profile to export aicpu_samples.json.")
    if timeline.get("sample_selection") != "formal_rounds_by_serial_replay_contract":
        return dict(status="sample_trace_replay_count_mismatch", measured_rounds=len(data["rounds"]),
                    expected_brackets=data["timed_prefix_replays"] + len(data["rounds"]),
                    observed_brackets=len(trace.graph_replays))
    pairs, errors, host_lower, host_upper = [], [], [], []
    for replay in trace.graph_replays:
        ordinal = trace.sample_rounds.get(replay["model"])
        if ordinal is None:
            continue
        sample = data["rounds"][ordinal - 1]
        elapsed = sample["event_elapsed_ns"] / 1000
        error = max(replay["event_low_us"] - elapsed, elapsed - replay["event_high_us"], 0)
        errors.append(error)
        host_lower.append(replay["host_end_us"] - sample["host_end_ns"] / 1000 - 1.0)
        host_upper.append(replay["host_start_us"] - sample["host_begin_ns"] / 1000 + 1.0)
        for node, task in zip(sample["nodes"], sorted(replay["tasks"], key=lambda e: e.start)):
            workers = node["workers"]
            entry = min(w["device_entry_ns"] for w in workers)
            returned = max(w["device_return_ns"] for w in workers)
            body = (max(w["device_finish_ns"] for w in workers) - min(w["device_start_ns"] for w in workers)) / 1000
            pairs.append(dict(round=ordinal, node=node["node"], entry=entry, returned=returned,
                task=task, workers=workers, entry_span=(returned - entry) / 1000, body_span=body,
                worker_start_skew=(max(w["device_entry_ns"] for w in workers) - entry) / 1000,
                worker_wall_mean=statistics.mean((w["device_return_ns"] - w["device_entry_ns"]) / 1000 for w in workers),
                worker_wall_max=max((w["device_return_ns"] - w["device_entry_ns"]) / 1000 for w in workers)))
    # Record-task windows are approximate; reject a changed run/order rather than aligning it by duration.
    if any(error > 2.0 for error in errors):
        return dict(status="sample_trace_event_window_mismatch", window_error_us=stats(errors))
    if max(host_lower) > min(host_upper):
        return dict(status="sample_trace_host_cadence_mismatch",
                    note="No constant unit-rate Host offset contains the corresponding API calls; check run identity, missing replays or clock drift.")
    selected_first = len(data["rounds"]) - timeline["selected_replays"] + 1
    selected = [p for p in pairs if p["round"] >= selected_first]
    if not selected:
        return dict(status="no_selected_sample_pairs")
    for p in pairs:
        p["task_span"] = p["task"].end - p["task"].start
        p["task_outside_entry"] = p["task_span"] - p["entry_span"]
    fields = ("task_span", "entry_span", "body_span", "task_outside_entry", "worker_start_skew", "worker_wall_mean", "worker_wall_max")
    result = dict(status="paired", association="formal_serial_replay_ordinal_and_event_window_check",
        selected_rounds=timeline["selected_replays"], selected_nodes=len(selected), workers=data["aicpu_workers"],
        event_window_error_us=stats(errors),
        host_pairing_check="common_unit_rate_offset_contains_each_replay_API_envelope",
        stats_us={key: stats([p[key] for p in selected]) for key in fields},
        per_node_stats_us={str(node): {key: stats([p[key] for p in selected if p["node"] == node])
                                     for key in ("body_span", "task_outside_entry")}
                           for node in range(data["gets_per_replay"])})
    representatives = sorted(selected, key=lambda p: p["task_outside_entry"])
    indexes = sorted({round(i * (len(representatives) - 1) / max(1, sample_limit - 1)) for i in range(sample_limit)})
    result["representative_nodes"] = [dict(round=representatives[i]["round"], node=representatives[i]["node"],
        **{key + "_us": round(representatives[i][key], 3) for key in fields}) for i in indexes]
    convert, counter_mode = None, False
    raw_clock = data["device_clock"] == "aicpu_monotonic_raw_ns"
    if clock_map is not None:
        convert = calibrated_clock(clock_map, data["device_id"], data["device_clock"])
    elif raw_clock or data["format"] == "kvaf-aicpu-samples-v2":
        frequency = 1000000000
        if not raw_clock:
            frequencies = {w["device_syscnt_frequency"] for p in pairs for w in p["workers"]}
            if len(frequencies) != 1 or not next(iter(frequencies)):
                result["clock"] = dict(status="hardware_counter_unavailable_or_inconsistent", split_available=False)
                return result
            frequency = next(iter(frequencies))
        diagnostics = {}
        try:
            convert = msprof_device_clock(trace.root, [p["task"] for p in pairs], frequency, diagnostics, raw_clock)
        except ValueError as error:
            code = str(error)
            result["clock"] = dict(status=code if code.startswith("sync_") else "sync_invalid_metadata", split_available=False)
            if diagnostics:
                result["clock"]["metadata_diagnostics"] = diagnostics
            return result
        counter_mode = not raw_clock
    if convert is not None:
        before, after, uncertainty, mapped_outside = [], [], [], []
        counter_span_delta = []
        try:
            for p in selected:
                entry_value = min(w["device_entry_syscnt"] for w in p["workers"]) if counter_mode else p["entry"]
                return_value = max(w["device_return_syscnt"] for w in p["workers"]) if counter_mode else p["returned"]
                entry, u1 = convert(entry_value)
                returned, u2 = convert(return_value)
                entry, returned = Decimal(str(entry)), Decimal(str(returned))
                before.append(float(entry - Decimal(str(p["task"].start))))
                after.append(float(Decimal(str(p["task"].end)) - returned))
                uncertainty.append(max(u1, u2) + 1.0)
                mapped_outside.append(p["task_span"] - float(returned - entry))
                counter_span_delta.append(float(returned - entry) - p["entry_span"])
        except ValueError:
            result["clock"] = dict(status="calibration_does_not_cover_samples", split_available=False)
            return result
        if any(a < -u or b < -u for a, b, u in zip(before, after, uncertainty)):
            result["clock"] = dict(status="calibration_inconsistent_with_task_containment", split_available=False)
            return result
        if counter_mode and any(abs(delta) > max(2.0, p["entry_span"] * .001)
                                for delta, p in zip(counter_span_delta, selected)):
            result["clock"] = dict(status="counter_and_steady_durations_disagree", split_available=False)
            return result
        result["clock"] = dict(status=("msprof_monotonic_raw_aligned" if raw_clock and clock_map is None else
                                      "msprof_counter_aligned" if counter_mode else "independently_calibrated"),
                               split_available=True, boundary_uncertainty_us=max(uncertainty))
        if raw_clock and clock_map is None:
            result["clock"]["source"] = "capture_device_monotonic_raw_to_host_monotonic_raw"
        if counter_mode:
            result["clock"]["source"] = "capture_dev_start_host_start_info_start_info"
            result["stats_us"]["counter_minus_steady_entry"] = stats(counter_span_delta)
        result["stats_us"].update(task_begin_to_first_entry=stats(before), last_return_to_task_end=stats(after),
            calibrated_entry_span=stats([p["task_span"] - outside for p, outside in zip(selected, mapped_outside)]),
            calibrated_task_outside_entry=stats(mapped_outside),
            boundary_reconciliation_error=stats([a + b - c for a, b, c in zip(before, after, mapped_outside)]))
        before_mean, after_mean = statistics.mean(before), statistics.mean(after)
        margin = max(uncertainty) * 2
        result["dominant_boundary"] = ("before_entry" if before_mean - after_mean > margin else
                                       "after_return" if after_mean - before_mean > margin else "similar_within_uncertainty")
        for item in result["representative_nodes"]:
            index = next(i for i, p in enumerate(selected) if (p["round"], p["node"]) == (item["round"], item["node"]))
            item.update(task_begin_to_first_entry_us=round(before[index], 3), last_return_to_task_end_us=round(after[index], 3),
                        calibrated_entry_span_us=round(selected[index]["task_span"] - mapped_outside[index], 3))
    else:
        # Containment bounds a common offset; it cannot identify a clock offset or a unique split.
        origin = pairs[0]["entry"]
        lower = max(p["task"].start - (p["entry"] - origin) / 1000 - 1.0 for p in pairs)
        upper = min(p["task"].end - (p["returned"] - origin) / 1000 + 1.0 for p in pairs)
        if lower > upper:
            result["clock"] = dict(status="no_consistent_unit_rate_offset", split_available=False)
            return result
        pre = [((p["entry"] - origin) / 1000 - p["task"].start) for p in selected]
        post = [(p["task"].end - (p["returned"] - origin) / 1000) for p in selected]
        result["clock"] = dict(status="bounds_only_no_independent_calibration", split_available=False,
            assumption="unit_rate_constant_offset_for_this_device_run", containment_tolerance_us=1.0,
            constraint_nodes=len(pairs), offset_range_width_us=round(upper - lower, 3),
            task_begin_to_first_entry_mean_bounds_us=[round(statistics.mean(pre) + o, 3) for o in (lower, upper)],
            last_return_to_task_end_mean_bounds_us=[round(statistics.mean(post) - o, 3) for o in (upper, lower)],
            note="Bounds share one unknown offset; no midpoint is a measured split. Independent calibration is required for point estimates.")
        lookup = {(p["round"], p["node"]): p for p in selected}
        for item in result["representative_nodes"]:
            p = lookup[item["round"], item["node"]]
            item["task_begin_to_first_entry_bounds_us"] = [round(
                (p["entry"] - origin) / 1000 + o - p["task"].start, 3) for o in (lower, upper)]
            item["last_return_to_task_end_bounds_us"] = [round(
                p["task"].end - (p["returned"] - origin) / 1000 - o, 3) for o in (upper, lower)]
    return result


def column_time(row, stems):
    for key, value in row.items():
        key = norm(key)
        for stem in stems:
            if key in (stem, stem + "us", stem + "ns", stem + "ms", stem + "s"):
                value = number(value)
                if value is not None:
                    scale = .001 if key.endswith("ns") else 1000 if key.endswith("ms") else 1e6 if key == stem + "s" else 1
                    return value * scale
    return None


def csv_report(root, selected_tasks=None, selected_events=None):
    durations, api_totals, files, warnings = defaultdict(list), defaultdict(lambda: [0, 0]), Counter(), Counter()
    selected, matched, selected_rows = defaultdict(list), set(), 0
    selected_tasks = selected_tasks or {}
    selected_events = selected_events or []
    timestamp_matches = 0
    prefixes = ("task_time_", "op_summary_", "api_statistic_", "acl_statistic_", "aicpu_")
    for path in sorted(root.rglob("*.csv")):
        source = next((p[:-1] for p in prefixes if path.name.startswith(p)), None)
        if source is None:
            continue
        try:
            files[source] += 1
            for raw in read_csv(path):
                row = {norm(k): v for k, v in raw.items() if k is not None}
                name = str(pick(row, "name", "opname", "nodename", "node", "apiname") or "")
                task_type = norm(pick(row, "tasktype", "optype", "type") or "")
                api = api_name(name)
                if source in ("api_statistic", "acl_statistic"):
                    count = number(pick(row, "count", "calls", "callcount", "frequency"))
                    total = column_time(row, ("totaltime", "total"))
                    mean = column_time(row, ("avgtime", "averagetime", "meantime", "mean", "avg"))
                    if api and count is not None and count > 0 and (total is not None or mean is not None):
                        api_totals[api][0] += count
                        api_totals[api][1] += total if total is not None else mean * count
                    continue
                if source == "aicpu":
                    task = identifier(pick(row, "taskid"))
                    identity = selected_tasks.get(task)
                    match = identity is not None and all(
                        not identity[key] or identifier(pick(row, key + "id")) in (None, identity[key])
                        for key in ("stream", "device"))
                    match_key = task
                    if match and selected_events:
                        indexes = [i for i, event in enumerate(selected_events) if event.ids["task"] == task]
                        if len(indexes) == 1:
                            match_key = ("event", indexes[0])
                    timestamp = column_time(row, ("timestamp", "starttime", "taskstarttime"))
                    if selected_events and timestamp is not None:
                        candidates = [i for i, event in enumerate(selected_events) if task is not None and event.ids["task"] == task
                            and all(not event.ids[key] or identifier(pick(row, key + "id")) in (None, event.ids[key])
                                    for key in ("stream", "device"))
                            and event.start - 1 <= timestamp <= event.end + 1]
                        match = len(candidates) == 1
                        if match:
                            match_key = ("event", candidates[0])
                            timestamp_matches += 1
                    if match:
                        matched.add(match_key)
                        selected_rows += 1
                    for field in ("dispatchtime", "totaltime", "tasktime", "computetime", "memcpytime"):
                        value = column_time(row, (field,))
                        if value is not None:
                            durations[source + "." + field].append(value)
                            if match:
                                selected[field].append(value)
                    continue
                if "aicpu" not in task_type and "aicpu" not in norm(name) and "memcpy" not in task_type and "memcpy" not in norm(name):
                    continue
                label = task_label(name) if "aicpu" in task_type or "aicpu" in norm(name) else "device_copy"
                value = column_time(row, ("taskduration", "duration", "tasktime"))
                if value is not None:
                    durations[source + "." + label].append(value)
        except (OSError, ValueError, TypeError) as error:
            warnings["CSV_READ_" + type(error).__name__] += 1
    return dict(files=dict(files), all_rows_duration_us={key: stats(value) for key, value in sorted(durations.items())},
                selected_aicpu_rows=dict(requested_unique_tasks=len(selected_events) if selected_events else len(selected_tasks),
                    matched_tasks=len(matched), rows=selected_rows, timestamp_matched_rows=timestamp_matches,
                    match_method="IDs_and_unique_exported_timestamp_containment_or_unique_ID",
                    row_duration_us={key: stats(value) for key, value in sorted(selected.items())}),
                aggregate_api_us={key: dict(calls=int(value[0]), total=round(value[1], 3), mean=round(value[1] / value[0], 3))
                                  for key, value in sorted(api_totals.items())}, warnings=dict(warnings))


def bounded_report(report, limit):
    report["omitted_sections"] = []
    while True:
        text = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False)
        if len(text) + 2 <= limit:
            return text + "\n"
        trace = report.get("timeline", {})
        if trace.get("representative_chains"):
            trace["representative_chains"].pop()
            section = "one_representative_chain"
        elif trace.get("graph_match_examples"):
            trace["graph_match_examples"].pop()
            section = "one_graph_match_example"
        elif report.get("entry_boundary", {}).get("representative_nodes"):
            report["entry_boundary"]["representative_nodes"].pop()
            section = "one_entry_boundary_example"
        elif "per_node_stats_us" in report.get("entry_boundary", {}):
            report["entry_boundary"].pop("per_node_stats_us")
            section = "entry_boundary_per_node_stats"
        elif "all_trace_api_us" in trace:
            trace.pop("all_trace_api_us")
            section = "all_trace_api_us"
        elif "csv_supplement" in report:
            report.pop("csv_supplement")
            section = "csv_supplement"
        elif "benchmark" in report:
            report.pop("benchmark")
            section = "benchmark"
        elif "chain_stats_us" in trace:
            trace.pop("chain_stats_us")
            section = "chain_stats_us"
        elif "link_field_coverage" in trace:
            trace.pop("link_field_coverage")
            section = "link_field_coverage"
        elif "scope_mean_comparison_us" in trace:
            trace.pop("scope_mean_comparison_us")
            trace.pop("scope_comparison_note", None)
            section = "scope_mean_comparison_us"
        elif len(report.get("notes", [])) > 3:
            report["notes"].pop()
            section = "notes_detail"
        else:
            raise ValueError("REPORT_LIMIT_TOO_SMALL")
        if section not in report["omitted_sections"]:
            report["omitted_sections"].append(section)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msprof-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--clock-map", type=Path,
                        help="independent two-anchor aicpu_steady_ns to msprof_trace_us calibration; never fit task boundaries")
    parser.add_argument("--execution", choices=("auto", "eager", "graph"), default="auto",
                        help="infer from benchmark CSV; graph selects validated replay Event brackets")
    parser.add_argument("--last", type=int, default=0, help="last N linked eager launches or validated timed graph replays; 0 keeps all")
    parser.add_argument("--samples", type=int, default=5, help="representative chains (1..10)")
    parser.add_argument("--kernel-pattern", help="optional device kernel-name regular expression")
    parser.add_argument("--max-chars", type=int, default=60000, help="ASCII report limit, including newline (4096..65535)")
    parser.add_argument("--output", type=Path, help="also save the same report locally")
    args = parser.parse_args(argv)
    if args.last < 0 or not 1 <= args.samples <= 10 or not 4096 <= args.max_chars <= 65535:
        parser.error("require --last >= 0, --samples 1..10 and --max-chars 4096..65535")
    try:
        root = args.msprof_dir
        if not root.is_dir() or (args.benchmark_dir and not args.benchmark_dir.is_dir()):
            raise ValueError("INPUT_DIRECTORY_MISSING")
        captures = [p for p in root.rglob("PROF_*") if p.is_dir() and
                    not any(parent.name.startswith("PROF_") for parent in p.relative_to(root).parents if str(parent) != ".")]
        if len(captures) > 1:
            raise ValueError("MULTIPLE_CAPTURES: pass one exact PROF directory")
        root = captures[0] if captures else root
        pattern = re.compile(args.kernel_pattern or GET_PATTERN, re.IGNORECASE)
        trace = Trace(root, pattern)
        benchmark = benchmark_report(args.benchmark_dir)
        execution, gets = report_execution(benchmark, args.execution)
        samples_data = load_graph_samples(args.benchmark_dir) if execution == "graph" else None
        if samples_data:
            item = benchmark.get(samples_data["backend"] + "/graph", {})
            if (not item or gets != samples_data["gets_per_replay"] or any(
                    item.get("configuration", {}).get(k) not in (None, samples_data[k])
                    for k in ("graph_api", "graph_submit"))):
                raise ValueError("INPUT_SAMPLE_BENCHMARK_CONFIG_MISMATCH")
        timeline = trace.report(args.last, args.samples, args.kernel_pattern is not None, execution, gets, samples_data)
        graph_scope_comparison(benchmark, timeline)
        supplement = csv_report(root, trace.selected_csv_tasks, trace.selected_task_events)
        if supplement["selected_aicpu_rows"]["matched_tasks"] == len(trace.selected_task_events):
            timeline.get("warnings", {}).pop("SELECTED_CSV_IDS_MISSING_OR_REUSED", None)
        report = dict(format="kvaf-aicpu-gap-core-v1", units="us", last_linked_launches=args.last,
                      notes=["Timing fields only; no paths, payloads, addresses, EIDs or original args are emitted.",
                             "Trace timestamps use Chrome trace microseconds; displayTimeUnit does not change them.",
                             "Representative intervals are relative to each Host launch start; IDs are not exported.",
                             "aligned_* gaps use msprof alignment; order-violating chains are excluded from aligned statistics but retained in samples.",
                             "Direct Host/device links use flows/IDs. Graph grouping uses Event containment plus same-stream evidence or unique model/captured-kernel provenance under serialized replay; methods are labeled and no nearest-task matching is used.",
                             "Graph requires record/model/record/Event-sync Host order, explicitly linked device Events, and the expected Get count. Internal-stream inference requires the serial BatchGet benchmark contract; it never substitutes eager warmup chains.",
                             "Device Event task begin/end windows approximate hardware Event timestamp bounds; they are not exact torch Event timestamps.",
                             "Body and entry spans stay separate. Multi-Get Event/kernel metrics are per Get; Host graph durations are per replay unless marked per_get.",
                             "All-trace API and CSV statistics include setup/warmup/readback; selected chains use --last.",
                             "Aggregate benchmark means and trace task means are separate scopes; entry_boundary uses per-round samples when available.",
                             "Version-3 device timestamps use CLOCK_MONOTONIC_RAW and the capture's device/Host monotonic-raw anchors. CNTFRQ and HWTS frequency are not assumed equal. No task boundary is used as a clock anchor.",
                             "Legacy counter samples require a matching clock domain or an independent --clock-map. Boundary uncertainty covers trace quantization, not an independently measured hardware synchronization error."],
                      benchmark=benchmark,
                      timeline=timeline,
                      csv_supplement=supplement)
        if execution == "graph":
            report["entry_boundary"] = entry_boundary_report(trace, timeline, samples_data, args.clock_map, args.samples)
        if not trace.files:
            report["notes"].append("No readable timeline JSON: CSVs alone cannot localize dispatch/completion gaps.")
        text = bounded_report(report, args.max_chars)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="ascii", newline="\n") as stream:
                stream.write(text)
        sys.stdout.write(text)
        return 0
    except (OSError, ValueError, TypeError, re.error) as error:
        # Do not print exception details that may contain controlled source data or paths.
        code = str(error) if isinstance(error, ValueError) and str(error).startswith(("INPUT_", "MULTIPLE_", "REPORT_")) else type(error).__name__
        sys.stderr.write("AICPU_GAP_EXTRACT_ERROR " + code + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

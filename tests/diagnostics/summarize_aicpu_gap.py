#!/usr/bin/env python3
"""Extract bounded, timing-only AICPU gap diagnostics using the standard library."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
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
    "host_graph": {"start_event_record", "replay_submit", "end_event_record", "event_sync", "replay_and_wait"},
    "host_graph_per_get": {"start_event_record", "replay_submit", "end_event_record", "event_sync", "replay_and_wait"},
    "stage": {"read", "ready_wait", "cq_wait", "scatter"},
}


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
        "stream": ("streamid", "stream"), "task": ("taskid",),
        "device": ("deviceid", "device"), "connection": ("connectionid",),
        "correlation": ("correlationid", "correlation")}.items()}
    for key in ("connection", "correlation"):
        if ids[key] in ("0", "18446744073709551615"):
            ids[key] = None
    return Event(pid, tid, start, duration, kind, label, ids, direction,
                 kind == "aicpu" and bool(pattern.search(kernel) or pattern.search(name)))


class Trace:
    def __init__(self, root, pattern):
        self.events, self.flows = [], defaultdict(list)
        self.warnings, self.files = Counter(), 0
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
        def capture_only(host_index, event):
            host = self.events[host_index]
            # A capture-time kernel ID does not identify a later model replay.
            stale = event.kind == "aicpu" and "launch" in norm(host.label) and any(
                e.track == host.track and e.kind == "launch" and "execute" in norm(e.label)
                and host.end <= e.start <= event.start for e in self.events)
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
                if not capture_only(host_index, event):
                    result[index] = (host_index, "flow")
                    continue
            if len(hosts) > 1:
                self.warnings["AMBIGUOUS_FLOW_LINK"] += 1
                continue
            possible = set()
            for key in ("connection", "correlation"):
                if event.ids[key] is not None:
                    possible.update(api_ids[key, event.ids[key]])
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
                    if capture_only(host_index, event):
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

    def report(self, last, samples, explicit_pattern):
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
        # CSV Task IDs can be reused by graph replays. Filter only unambiguous IDs.
        occurrences = Counter(e.ids["task"] for e in self.events if e.kind == "aicpu" and e.ids["task"] is not None)
        self.selected_csv_tasks = {e.ids["task"]: e.ids for e in tasks if occurrences[e.ids["task"]] == 1}
        if len(self.selected_csv_tasks) < len(tasks):
            self.warnings["SELECTED_CSV_IDS_MISSING_OR_REUSED"] += len(tasks) - len(self.selected_csv_tasks)
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
                for key in ("stream", "task", "device", "connection", "correlation")})
        return dict(json_files=self.files, complete_events=len(self.events), target_selection=selection,
                    flow_groups=len(self.flows), link_field_coverage=coverage,
                    target_events=len(targets), linked_target_events=sum(map(len, groups.values())),
                    linked_launches=len(launches), selected_launches=len(selected), complete_chains=len(chains),
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
                     k in ("e2e_mean_us", "kernel_mean_us", "e2e_p99_us", "kernel_p99_us", "e2e_mean_GBps"))}
            result[key]["configuration"] = {k: row[k] for k in ("get_binding", "graph_api", "latency_scope")
                if row.get(k) in ("ordinary", "prepared", "capture", "torch", "acl", "single_get", "amortized_per_get")}
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


def csv_report(root, selected_tasks=None):
    durations, api_totals, files, warnings = defaultdict(list), defaultdict(lambda: [0, 0]), Counter(), Counter()
    selected, matched, selected_rows = defaultdict(list), set(), 0
    selected_tasks = selected_tasks or {}
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
                    if match:
                        matched.add(task)
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
                selected_aicpu_rows=dict(requested_unique_tasks=len(selected_tasks), matched_tasks=len(matched),
                    rows=selected_rows, row_duration_us={key: stats(value) for key, value in sorted(selected.items())}),
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
        else:
            raise ValueError("REPORT_LIMIT_TOO_SMALL")
        if section not in report["omitted_sections"]:
            report["omitted_sections"].append(section)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msprof-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--last", type=int, default=0, help="select the last N linked kernel launches; 0 keeps all")
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
        report = dict(format="kvaf-aicpu-gap-core-v1", units="us", last_linked_launches=args.last,
                      notes=["Timing fields only; no paths, payloads, addresses, EIDs or original args are emitted.",
                             "Trace timestamps use Chrome trace microseconds; displayTimeUnit does not change them.",
                             "Representative intervals are relative to each Host launch start; IDs are not exported.",
                             "aligned_* gaps use msprof alignment; order-violating chains are excluded from aligned statistics but retained in samples.",
                             "Only explicit flows or matching IDs link Host/device tasks; no nearest-task guessing.",
                             "Graph chains use model execute plus Event/stream sync. Event brackets require linked record tasks; absence is not zero overhead.",
                             "Body and entry spans stay separate. Multi-Get Event/kernel metrics are per Get; Host graph durations are per replay unless marked per_get.",
                             "All-trace API and CSV statistics include setup/warmup/readback; selected chains use --last.",
                             "Benchmark kernel body and CANN task duration are separate scopes, not paired per-call samples."],
                      benchmark=benchmark_report(args.benchmark_dir),
                      timeline=trace.report(args.last, args.samples, args.kernel_pattern is not None),
                      csv_supplement=csv_report(root, trace.selected_csv_tasks))
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

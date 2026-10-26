#!/usr/bin/env python3
"""Summarize KVAF DDR initialization logs without importing an NPU runtime.

Example:
  python3 analyze_ddr_init.py --cpu-log cpu.log --npu-log npu.log \
      --output report.json --csv phases.csv

Stage timers are nested. Client RPC timers include server work and queueing; do
not add client and server timings. Linux vmstat counters are system-wide signals,
not proof that this allocation or THP caused a delay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

PREFIXES = {"KVAF_INIT_PROFILE ": "profile", "DDR_INIT_PROBE ": "probe"}
ERROR_SIGNAL = re.compile(r"\b[A-Z][A-Z0-9_]*_(?:ERROR|FAILED)\b")
USAGE_FIELDS = (
    "user_cpu_ms", "system_cpu_ms", "minor_faults", "major_faults",
    "voluntary_context_switches", "involuntary_context_switches", "usage_scope",
)
NOTES = [
    "Stage timers are nested; do not sum initialize, RPC and server phases.",
    "initialize is successful only when its done record and a ready record are both present.",
    "CPU/NPU matching is inferred from a unique session/base/bytes/candidate tuple; it does not verify server host identity.",
    "Global vmstat deltas include other processes and are correlations, not proof of THP causation.",
    "Snapshot RSS and AnonHugePages describe the entire CPU service, not an individual region.",
    "Snapshot collection/logging contribute to outer timers; compare leaf phases and the recorded collection cost across modes.",
    "p95 is reported only with at least 20 timed samples; smaller groups show min/median/max.",
    "A begin without a matching end is pending in this log excerpt, not evidence of failure.",
    "Absolute timestamps help locate records; elapsed values use local steady clocks, so do not subtract timestamps across hosts.",
]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _page_mode(value: Any) -> Any:
    return value if value not in (None, "", "na") else None


def _identity_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return value
    return value


def read_logs(logs: list[tuple[str, Path]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Keep source locations and raw records; ignore unrelated runtime chatter."""
    records, warnings, error_signals = [], [], []
    for label, path in logs:
        source = str(path.resolve())
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                found = [(line.find(prefix), prefix, kind) for prefix, kind in PREFIXES.items()
                         if prefix in line]
                if not found:
                    if ERROR_SIGNAL.search(line):
                        error_signals.append({"source": source, "line": line_number,
                                              "text": line.rstrip(), "input_role": label})
                    continue
                offset, prefix, kind = min(found)
                payload = line[offset + len(prefix):].strip()
                try:
                    record = json.loads(payload)
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                except (ValueError, json.JSONDecodeError) as exc:
                    warnings.append({"source": source, "line": line_number,
                                     "message": f"invalid {kind} JSON: {exc}"})
                    continue
                records.append({"kind": kind, "source": source, "line": line_number,
                                "input_role": label, "record": record})
    return records, warnings, error_signals


def probe_runs(records: list[dict]) -> list[dict]:
    """A config starts a run, keeping PID reuse and concatenated logs separate."""
    runs, current = [], {}
    for item in records:
        if item["kind"] != "probe":
            continue
        record = item["record"]
        key = (item["source"], record.get("pid"))
        run = current.get(key)
        new_initialize = (record.get("event") == "begin" and
                          record.get("operation") == "initialize" and run is not None and
                          run.get("initialize_event") is not None)
        if run is None or record.get("event") == "config" or new_initialize:
            if run is not None:
                run["last_line"] = item["line"] - 1
            run = {"id": f"run-{len(runs) + 1}", "source": item["source"],
                   "pid": record.get("pid"), "device": record.get("device"),
                   "first_line": item["line"], "last_line": None, "config": {},
                   "operations": [], "ready": False, "initialize_event": None,
                   "initialize_elapsed_s": None, "status": "incomplete"}
            runs.append(run)
            current[key] = run
        if record.get("event") == "config":
            run["config"] = record.copy()
        if record.get("operation"):
            run["operations"].append(record.copy())
        if record.get("operation") == "initialize" and record.get("event") in ("done", "failed"):
            run["initialize_event"] = record["event"]
            run["initialize_elapsed_s"] = record.get("elapsed_s")
        if record.get("event") == "ready":
            run["ready"] = True
        run["status"] = ("failed" if run["initialize_event"] == "failed" else
                         "success" if run["initialize_event"] == "done" and run["ready"] else
                         "done_without_ready" if run["initialize_event"] == "done" else "incomplete")
    return runs


def profile_attempts(records: list[dict]) -> list[dict]:
    attempts, current, epochs = [], {}, {}
    for item in records:
        if item["kind"] == "probe" and item["record"].get("event") == "config":
            process = (item["source"], item["record"].get("pid"))
            epochs[process] = epochs.get(process, 0) + 1
        if item["kind"] != "profile":
            continue
        record = item["record"]
        process = (item["source"], record.get("pid"))
        key = (item["source"], record.get("role"), record.get("pid"),
               record.get("attempt"), epochs.get(process, 0))
        attempt = current.get(key)
        # Attempt numbers restart with a process/service. Split repeated top-level
        # begins instead of silently merging a concatenated/reused identity.
        boundary = (record.get("event") == "snapshot" and record.get("phase") == "before_allocate" and
                    attempt is not None and any(r.get("event") == "end" and r.get("phase") == "total_allocate"
                                                for r in attempt["records"]))
        boundary = boundary or (record.get("event") == "begin" and
                                record.get("phase") in ("total_allocate", "candidate_initialize", "candidate_query") and
                                attempt is not None and any(r.get("event") == "begin" and
                                r.get("phase") == record.get("phase") for r in attempt["records"]))
        if attempt is None or boundary:
            attempt = {"id": f"attempt-{len(attempts) + 1}", "source": item["source"],
                       "role": record.get("role"), "pid": record.get("pid"),
                       "attempt": record.get("attempt"), "first_line": item["line"],
                       "process_epoch": epochs.get(process, 0),
                       "last_line": item["line"], "bytes": record.get("bytes"),
                       "candidate": record.get("candidate"), "page_mode": _page_mode(record.get("page_mode")),
                       "records": [], "phases": [], "snapshots": [], "regions": [],
                       "incomplete_phases": [], "_pending": {}}
            current[key] = attempt
            attempts.append(attempt)
        attempt["last_line"] = item["line"]
        for field in ("bytes", "candidate", "page_mode"):
            if (record.get(field) is not None and record.get(field) != "" and
                    (field != "page_mode" or _page_mode(record[field]) is not None)):
                attempt[field] = record[field]
        attempt["records"].append(record.copy())
        if record.get("event") == "begin":
            pending = {**record, "source": item["source"], "line": item["line"]}
            attempt["_pending"].setdefault(record.get("phase"), []).append(pending)
        elif record.get("event") == "end":
            attempt["phases"].append(record.copy())
            pending = attempt["_pending"].get(record.get("phase"), [])
            if pending:
                pending.pop()
        elif record.get("event") == "snapshot":
            attempt["snapshots"].append(record.copy())
        elif record.get("event") == "region":
            attempt["regions"].append(record.copy())
    for attempt in attempts:
        pending = attempt.pop("_pending")
        attempt["incomplete_phases"] = sorted(
            (begin for stack in pending.values() for begin in stack),
            key=lambda begin: begin["line"])
    return attempts


def region_identity(region: dict) -> tuple | None:
    fields = ("session", "base", "bytes", "candidate")
    if any(field not in region for field in fields):
        return None
    return tuple(_identity_value(region[field]) for field in fields)


def correlate(attempts: list[dict]) -> list[dict]:
    cpu_regions: dict[tuple, list[dict]] = {}
    for attempt in attempts:
        if attempt["role"] != "cpu":
            continue
        for region in attempt["regions"]:
            identity = region_identity(region)
            if identity is not None:
                matches = cpu_regions.setdefault(identity, [])
                if not any(item["id"] == attempt["id"] for item in matches):
                    matches.append(attempt)
    correlations = []
    for attempt in attempts:
        if attempt["role"] not in ("npu", "runtime"):
            continue
        for region in attempt["regions"]:
            identity = region_identity(region)
            matches = cpu_regions.get(identity, []) if identity is not None else []
            result = {"npu_attempt": attempt["id"], "region": region.copy(),
                      "cpu_attempts": [item["id"] for item in matches],
                      "status": "matched" if len(matches) == 1 else
                                "ambiguous" if len(matches) > 1 else
                                "missing_identity" if identity is None else "unmatched",
                      "confidence": "inferred_unique" if len(matches) == 1 else "none"}
            correlations.append(result)
            if len(matches) == 1:
                attempt["page_mode"] = matches[0].get("page_mode") or attempt["page_mode"]
    # RuntimeCore emits nested stages under the same NPU process/attempt, but
    # does not own the region record. Apply a uniquely inferred mode there too.
    contexts: dict[tuple, set] = {}
    for attempt in attempts:
        if attempt["role"] == "npu" and _page_mode(attempt.get("page_mode")):
            key = (attempt["source"], attempt["pid"], attempt["attempt"], attempt["process_epoch"])
            contexts.setdefault(key, set()).add(attempt["page_mode"])
    for attempt in attempts:
        if attempt["role"] in ("npu", "runtime") and not _page_mode(attempt.get("page_mode")):
            modes = contexts.get((attempt["source"], attempt["pid"], attempt["attempt"],
                                  attempt["process_epoch"]), set())
            if len(modes) == 1:
                attempt["page_mode"] = next(iter(modes))
    return correlations


def _deltas(before: dict, after: dict, monotonic: bool = False) -> tuple[dict, list[str]]:
    differences, reset = {}, []
    for key in sorted(before.keys() & after.keys()):
        if _number(before[key]) and _number(after[key]):
            difference = after[key] - before[key]
            if monotonic and difference < 0:
                reset.append(key)
            else:
                differences[key] = difference
    return differences, reset


def snapshot_deltas(attempt: dict) -> list[dict]:
    snapshots = attempt["snapshots"]
    if len(snapshots) < 2:
        return []
    pairs = list(zip(snapshots, snapshots[1:]))
    # Also expose the complete allocation interval when there are intermediate
    # snapshots; retain adjacent intervals to localize counter changes.
    before = next((s for s in snapshots if s.get("phase") == "before_allocate"), None)
    after = next((s for s in reversed(snapshots) if s.get("phase") == "after_register"), None)
    if before is not None and after is not None and (before, after) not in pairs:
        pairs.append((before, after))
    result = []
    for before, after in pairs:
        counters, resets = _deltas(before.get("global_vmstat", {}), after.get("global_vmstat", {}), True)
        numa, _ = _deltas(before.get("numa_memfree_kib", {}), after.get("numa_memfree_kib", {}))
        process, _ = _deltas(before.get("process_status", {}), after.get("process_status", {}))
        memory, _ = _deltas(before.get("global_meminfo_kib", {}), after.get("global_meminfo_kib", {}))
        row = {"from": before.get("phase"), "to": after.get("phase"),
               "global_vmstat_delta": counters, "counter_decreases": resets,
               "numa_memfree_delta_kib": numa, "process_status_delta_kib": process,
               "global_meminfo_delta_kib": memory}
        field = "process_anon_huge_pages_kib"
        if _number(before.get(field)) and _number(after.get(field)):
            row["process_anon_huge_pages_delta_kib"] = after[field] - before[field]
        result.append(row)
    return result


def phase_rows(attempts: list[dict]) -> list[dict]:
    rows = []
    for attempt in attempts:
        for phase in attempt["phases"]:
            row = {field: attempt.get(field) for field in
                   ("source", "role", "pid", "attempt", "bytes", "candidate", "page_mode")}
            for field in ("bytes", "candidate"):
                if field in phase:
                    row[field] = phase[field]
            row["page_mode"] = _page_mode(phase.get("page_mode")) or attempt.get("page_mode")
            row["attempt_id"] = attempt["id"]
            row.update({field: phase.get(field) for field in
                        ("phase", "elapsed_ms", "outcome", *USAGE_FIELDS)})
            row["outcome"] = row["outcome"] or "unknown"
            rows.append(row)
    return rows


def summarize_phases(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["bytes"], row["page_mode"], row["role"], row["phase"])
        groups.setdefault(key, []).append(row)
    result = []
    for (size, mode, role, phase), samples in groups.items():
        values = sorted(row["elapsed_ms"] for row in samples if _number(row["elapsed_ms"]))
        row = {"bytes": size, "page_mode": mode, "role": role, "phase": phase,
               "count": len(values), "failed_count": sum(s["outcome"] == "failed" for s in samples),
               "unknown_count": sum(s["outcome"] == "unknown" for s in samples)}
        if values:
            row.update({"min_ms": values[0], "median_ms": statistics.median(values),
                        "p95_ms": values[math.ceil(len(values) * .95) - 1] if len(values) >= 20 else None,
                        "max_ms": values[-1]})
        result.append(row)
    return sorted(result, key=lambda r: (str(r["page_mode"]), r["bytes"] or 0,
                                         str(r["role"]), str(r["phase"])))


def analyze(records: list[dict], warnings: list[dict] | None = None,
            error_signals: list[dict] | None = None) -> dict:
    runs = probe_runs(records)
    attempts = profile_attempts(records)
    correlations = correlate(attempts)
    for attempt in attempts:
        attempt["snapshot_deltas"] = snapshot_deltas(attempt)
        measured = [s for s in attempt["snapshots"] if _number(s.get("collection_elapsed_ms"))]
        allocation = [s["collection_elapsed_ms"] for s in measured if s.get("phase") != "after_release"]
        release = [s["collection_elapsed_ms"] for s in measured if s.get("phase") == "after_release"]
        attempt["snapshot_collection_ms"] = {
            "allocation": sum(allocation) if allocation else None,
            "release": sum(release) if release else None,
        }
    for run in runs:
        related = [a for a in attempts if a["source"] == run["source"] and
                   a["pid"] == run["pid"] and a["role"] in ("npu", "runtime") and
                   a["first_line"] >= run["first_line"] and
                   (run["last_line"] is None or a["first_line"] <= run["last_line"])]
        run["attempt_ids"] = [a["id"] for a in related]
        # A retry is evidenced by separate candidate attempts, not by discovering
        # several EIDs, nor by multiple nested phases within one attempt.
        candidate_attempts = [a for a in related if any(p.get("phase") == "candidate_initialize"
                                                       for p in a["records"]) or a["regions"]]
        run["candidate_attempt_count"] = len(candidate_attempts) if related else None
        run["additional_candidate_attempts"] = max(0, len(candidate_attempts) - 1) if related else None
        run["retry_count"] = sum(
            any(p.get("phase") == "candidate_initialize" and p.get("outcome") == "failed"
                for p in attempt["phases"])
            for attempt in candidate_attempts[:-1]) if related else None
    rows = phase_rows(attempts)
    return {"schema_version": 1, "notes": NOTES, "runs": runs, "attempts": attempts,
            "correlations": correlations, "phase_rows": rows,
            "phase_summaries": summarize_phases(rows), "warnings": warnings or [],
            "error_signals": error_signals or [], "raw_records": records}


def _display(value: Any, precision: int = 3) -> str:
    if value is None or value == "":
        return "?"
    return f"{value:.{precision}f}" if isinstance(value, float) else str(value)


def print_report(report: dict) -> None:
    print("Initialization totals (seconds; success requires initialize done + ready):")
    if not report["runs"]:
        print("  No DDR_INIT_PROBE records; only stage profiles are available.")
    for run in report["runs"]:
        config = run["config"]
        size = config.get("allocated_bytes", config.get("requested_bytes"))
        gib = size / 1024**3 if _number(size) else None
        print(f"  {Path(run['source']).name} pid={run['pid']} device={run['device']} "
              f"GiB={_display(gib)} initialize={_display(run['initialize_elapsed_s'], 6)} "
              f"status={run['status']} candidates={_display(run['candidate_attempt_count'])} "
              f"retries={_display(run['retry_count'])}")
    print("\nPer-attempt phases (milliseconds; nested, not additive):")
    for attempt in report["attempts"]:
        stages = ", ".join(f"{p.get('phase')}={_display(p.get('elapsed_ms'))}"
                           f"({p.get('outcome', 'unknown')})" for p in attempt["phases"])
        print(f"  {attempt['id']} {Path(attempt['source']).name} role={attempt['role']} "
              f"pid={attempt['pid']} attempt={attempt['attempt']} "
              f"mode={_display(attempt['page_mode'])}: {stages or 'no completed phases'}")
        if attempt["incomplete_phases"]:
            pending = ", ".join(f"{p.get('phase')} (begin line {p['line']})"
                                for p in attempt["incomplete_phases"])
            print(f"    Pending in this log excerpt: {pending}")
    print("\nStage summaries (milliseconds; all timed outcomes, failed samples counted separately):")
    print("  mode GiB role phase n failed min median p95 max")
    for row in report["phase_summaries"]:
        gib = row["bytes"] / 1024**3 if _number(row["bytes"]) else None
        print("  " + " ".join(_display(v) for v in
              (row["page_mode"], gib, row["role"], row["phase"], row["count"],
               row["failed_count"], row.get("min_ms"), row.get("median_ms"),
               row.get("p95_ms"), row.get("max_ms"))))
    print("\nAllocation intervals (system-wide vmstat deltas; not per-process attribution):")
    found = False
    for attempt in report["attempts"]:
        intervals = attempt["snapshot_deltas"]
        allocation = [d for d in intervals if d["from"] == "before_allocate" and
                      d["to"] == "after_register"]
        leaf_windows = [d for d in intervals if (d["from"], d["to"]) in
                        (("before_memset", "after_memset"),
                         ("before_register", "after_register"))]
        for delta in (allocation + leaf_windows) or intervals:
            nonzero = {k: v for k, v in delta["global_vmstat_delta"].items() if v}
            print(f"  {attempt['id']} pid={attempt['pid']} attempt={attempt['attempt']} "
                  f"{delta['from']}->{delta['to']} counters={json.dumps(nonzero, sort_keys=True)}")
            if delta["global_meminfo_delta_kib"]:
                print("    Global memory change (KiB): " +
                      json.dumps(delta["global_meminfo_delta_kib"], sort_keys=True))
            if delta["counter_decreases"]:
                print(f"    Counter decreases omitted: {', '.join(delta['counter_decreases'])}")
            found = True
        overhead = attempt["snapshot_collection_ms"]
        if overhead["allocation"] is not None or overhead["release"] is not None:
            print(f"  {attempt['id']} snapshot collection overhead: "
                  f"allocation={_display(overhead['allocation'])} ms "
                  f"release={_display(overhead['release'])} ms")
    if not found:
        print("  No paired snapshots available.")
    ambiguous = sum(c["status"] == "ambiguous" for c in report["correlations"])
    unmatched = sum(c["status"] in ("unmatched", "missing_identity") for c in report["correlations"])
    print(f"\nCorrelation: {ambiguous} ambiguous, {unmatched} unmatched; "
          f"parse warnings={len(report['warnings'])}, raw error signals={len(report['error_signals'])}.")
    for note in report["notes"]:
        print("  Note: " + note)
    for warning in report["warnings"]:
        print(f"  Warning: {warning['source']}:{warning['line']}: {warning['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-log", type=Path, action="append", default=[], help="CPU service log; repeatable")
    parser.add_argument("--npu-log", type=Path, action="append", default=[], help="NPU probe log; repeatable")
    parser.add_argument("--output", type=Path, help="Write complete JSON report, including raw records")
    parser.add_argument("--csv", type=Path, help="Write one row per completed phase")
    args = parser.parse_args(argv)
    logs = [("cpu", p) for p in args.cpu_log] + [("npu", p) for p in args.npu_log]
    if not logs:
        parser.error("provide at least one --cpu-log or --npu-log")
    # Avoid counting one file twice if accidentally passed under both flags.
    unique, seen = [], set()
    for label, path in logs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append((label, path))
            seen.add(resolved)
    try:
        records, warnings, errors = read_logs(unique)
    except OSError as exc:
        parser.error(str(exc))
    report = analyze(records, warnings, errors)
    print_report(report)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.csv:
        fields = ["source", "role", "pid", "attempt", "attempt_id", "bytes", "candidate", "page_mode",
                  "phase", "elapsed_ms", "outcome", *USAGE_FIELDS]
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report["phase_rows"])
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())

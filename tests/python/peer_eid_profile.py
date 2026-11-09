"""Per-peer traffic across cores, with overlap evidence only within a core."""

import csv
from pathlib import Path

from npu_eids import peer_eid_path


def summarize(samples, peer_eids, cores, jettys, token_bytes):
    single_peer = len(peer_eids) == 1
    rows = []
    for sample_index, profiles in enumerate(samples):
        if len(profiles) != cores:
            raise ValueError("peer profile core count mismatch")
        sample_paths = set()
        for core, profile in enumerate(profiles):
            lanes = profile["jettys"][:jettys]
            if len(lanes) != jettys:
                raise ValueError("peer profile Jetty count mismatch")
            groups = [[] for _ in peer_eids]
            for slot, lane in enumerate(lanes):
                path = int(lane["eid_path"])
                if path != peer_eid_path(core, slot, cores, jettys, len(peer_eids)):
                    raise ValueError("peer profile route does not match requested EID order")
                if single_peer and int(lane["requests"]) == 0:
                    if any(int(lane[key]) != 0 for key in (
                            "completion_count", "first_doorbell_cycle", "last_doorbell_cycle",
                            "first_completion_cycle", "last_completion_cycle")):
                        raise ValueError("idle peer Jetty has completion or timeline data")
                    continue
                if int(lane["requests"]) <= 0 or int(lane["completion_count"]) <= 0:
                    raise ValueError("a configured peer Jetty had no completed traffic")
                first, last, done_first, done_last = (int(lane[key]) for key in (
                    "first_doorbell_cycle", "last_doorbell_cycle",
                    "first_completion_cycle", "last_completion_cycle"))
                if not (0 < first <= last <= done_last and first <= done_first <= done_last):
                    raise ValueError("invalid peer Jetty timeline")
                groups[path].append(lane)
                sample_paths.add(path)
            per_core = []
            for path, group in enumerate(groups):
                if not group and not single_peer:
                    continue  # This peer is assigned to another core.
                requests = sum(int(lane["requests"]) for lane in group)
                row = dict(sample=sample_index, core=core, peer_path=path,
                           peer_eid=peer_eids[path], requests=requests,
                           bytes=requests * token_bytes)
                for key, reduce in (("first_doorbell_cycle", min),
                                    ("last_doorbell_cycle", max),
                                    ("first_completion_cycle", min),
                                    ("last_completion_cycle", max)):
                    row[key] = reduce(int(lane[key]) for lane in group) if group else 0
                per_core.append(row)
            # First-wave overlap is stricter than overlapping whole-run envelopes:
            # All peers on this core must be posted before ANY peer's completion.
            # Never compare timestamps belonging to different cores.
            overlap = None if len(per_core) < 2 else (
                min(r["first_completion_cycle"] for r in per_core) -
                max(r["first_doorbell_cycle"] for r in per_core))
            for row in per_core:
                row["first_wave_observed_overlap_cycles"] = "" if overlap is None else max(0, overlap)
            rows.extend(per_core)
        if not single_peer and sample_paths != set(range(len(peer_eids))):
            raise ValueError("a requested peer EID has no traffic in this sample")
    if not rows:
        raise ValueError("no peer profile samples")
    return rows


def report(samples, peer_eids, cores, jettys, token_bytes, csv_path):
    rows = summarize(samples, peer_eids, cores, jettys, token_bytes)
    if csv_path:
        with Path(csv_path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for path, eid in enumerate(peer_eids):
        selected = [row for row in rows if row["peer_path"] == path]
        print(f"PEER_EID_TRAFFIC path={path} eid={eid} "
              f"requests={sum(r['requests'] for r in selected)} "
              f"bytes={sum(r['bytes'] for r in selected)}")
    if len(peer_eids) == 1:
        print(f"PEER_EID_BASELINE samples={len(samples)} cores={cores} "
              "execution=single_kernel concurrency_check=not_applicable")
        return
    overlaps = [row["first_wave_observed_overlap_cycles"] for row in rows
                if row["first_wave_observed_overlap_cycles"] != ""]
    minimum = min(overlaps) if overlaps else None
    if jettys < len(peer_eids):
        print(f"PEER_EID_MULTI_CORE_TRAFFIC samples={len(samples)} cores={cores} "
              f"min_within_core_observed_overlap_cycles={minimum if minimum is not None else 'not_applicable'} "
              "execution=single_kernel cross_core_overlap=not_measured")
    else:
        print(f"PEER_EID_CONCURRENCY samples={len(samples)} cores={cores} "
              f"min_first_wave_observed_overlap_cycles={minimum} "
              "execution=single_kernel evidence=doorbell_to_observed_cqe")
    if minimum is not None and minimum <= 0:
        raise ValueError("concurrent first-wave submission was not observed in every sample/core")

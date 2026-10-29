"""Observed concurrent READ windows; cycle comparisons stay within one AIV core."""

import csv
from pathlib import Path


def summarize(samples, peer_eids, cores, jettys, token_bytes):
    rows = []
    for sample_index, profiles in enumerate(samples):
        if len(profiles) != cores:
            raise ValueError("peer profile core count mismatch")
        for core, profile in enumerate(profiles):
            lanes = profile["jettys"][:jettys]
            if len(lanes) != jettys:
                raise ValueError("peer profile Jetty count mismatch")
            groups = [[] for _ in peer_eids]
            for slot, lane in enumerate(lanes):
                path = int(lane["eid_path"])
                if path != slot % len(peer_eids):
                    raise ValueError("peer profile route does not match requested EID order")
                if int(lane["requests"]) <= 0 or int(lane["completion_count"]) <= 0:
                    raise ValueError("a configured peer Jetty had no completed traffic")
                first, last, done_first, done_last = (int(lane[key]) for key in (
                    "first_doorbell_cycle", "last_doorbell_cycle",
                    "first_completion_cycle", "last_completion_cycle"))
                if not (0 < first <= last <= done_last and first <= done_first <= done_last):
                    raise ValueError("invalid peer Jetty timeline")
                groups[path].append(lane)
            per_core = []
            for path, group in enumerate(groups):
                if not group:
                    raise ValueError("a requested peer EID has no Jetty")
                requests = sum(int(lane["requests"]) for lane in group)
                row = dict(sample=sample_index, core=core, peer_path=path,
                           peer_eid=peer_eids[path], requests=requests,
                           bytes=requests * token_bytes)
                for key, reduce in (("first_doorbell_cycle", min),
                                    ("last_doorbell_cycle", max),
                                    ("first_completion_cycle", min),
                                    ("last_completion_cycle", max)):
                    row[key] = reduce(int(lane[key]) for lane in group)
                per_core.append(row)
            # First-wave overlap is stricter than overlapping whole-run envelopes:
            # every peer was submitted before ANY peer's first observed completion.
            overlap = min(r["first_completion_cycle"] for r in per_core) - max(
                r["first_doorbell_cycle"] for r in per_core)
            for row in per_core:
                row["first_wave_observed_overlap_cycles"] = max(0, overlap)
            rows.extend(per_core)
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
    minimum = min(row["first_wave_observed_overlap_cycles"] for row in rows)
    print(f"PEER_EID_CONCURRENCY samples={len(samples)} cores={cores} "
          f"min_first_wave_observed_overlap_cycles={minimum} "
          "execution=single_kernel evidence=doorbell_to_observed_cqe")
    if minimum <= 0 and len(peer_eids) > 1:
        raise ValueError("concurrent first-wave submission was not observed in every sample/core")

"""Paired Host/device duration accounting; never mix absolute clock domains."""

import statistics

from utils.statistics import percentile


def eager_host_sample(host, e2e_us, kernel_us):
    """Four disjoint terms sum to E2E minus kernel; API subspans are inclusive."""
    begin, submitted = host["begin_ns"], host["end_ns"]
    end = host["drain_end_ns"] or submitted
    launch, launched = host["provider_launch_begin_ns"], host["provider_launch_end_ns"]
    sync, synced = host["provider_sync_begin_ns"], host["provider_sync_end_ns"]
    if not 0 < begin <= launch <= launched <= sync <= synced <= end:
        raise AssertionError("missing or unordered eager Host timestamps; rebuild the matching binding")
    if host["drain_end_ns"] and not submitted <= host["drain_begin_ns"] <= sync <= end:
        raise AssertionError("Host submit/drain samples do not match")
    groups = {}
    binding = ("tensor_views", "request_validation", "stream", "refs", "binding_lock",
               "set_device", "core_call", "gil_reacquire", "owner_update", "total")
    core = ("lifecycle", "table", "direction", "previous_drain", "validate", "dispatch", "finish", "total")
    dispatch = ("lock", "resolve", "output_check", "previous_drain", "context", "describe", "acquire",
                "bind", "warmup", "provider", "context_release", "total")
    for kind, fields, prefix in (("binding", binding, ""), ("core", core, "core_"),
                                 ("dispatch", dispatch, "dispatch_")):
        values = {field: host[prefix + field + "_ns"] / 1000 for field in fields}
        values["other"] = values["total"] - sum(values[field] for field in fields[:-1])
        groups[kind] = values
    groups["provider"] = {name: host["provider_" + name + "_ns"] / 1000 for name in (
        "prepare", "submit", "completion", "cleanup", "lock", "capture_check", "status_enqueue",
        "stream_sync", "status_check", "prepare_capture", "previous_completion", "input_build",
        "profile_buffers", "registration")}
    groups["provider"]["launch_api"] = (launched - launch) / 1000
    groups["stream"] = {name: host["stream_" + name + "_ns"] / 1000 for name in ("select", "handle")}
    provider = groups["provider"]
    for total, parts in (("prepare", ("prepare_capture", "previous_completion", "input_build", "profile_buffers", "registration")),
                         ("submit", ("launch_api",)),
                         ("completion", ("capture_check", "status_enqueue", "stream_sync", "status_check"))):
        provider[total + "_other"] = provider[total] - sum(provider[name] for name in parts)
    for kind, prefix, fields in (
        ("binding_drain", "drain_", ("binding_lock", "set_device", "core_call", "gil_reacquire", "owner_update", "total")),
        ("core_drain", "core_drain_", ("lock", "operator", "release", "total")),
        ("dispatch_drain", "dispatch_drain_", ("lock", "provider", "release", "total")),
    ):
        groups[kind] = {name: host[prefix + name + "_ns"] / 1000 for name in fields}
        groups[kind]["other"] = groups[kind]["total"] - sum(groups[kind][name] for name in fields[:-1])
    groups["gap"] = dict(
        e2e=e2e_us, kernel=kernel_us, e2e_minus_kernel=e2e_us - kernel_us,
        python_boundary=e2e_us - (end - begin) / 1000,
        before_launch=(launch - begin) / 1000,
        runtime_window=(synced - launch) / 1000,
        runtime_outside_kernel=(synced - launch) / 1000 - kernel_us,
        after_completion=(end - synced) / 1000,
        intercall=(host["drain_begin_ns"] - submitted) / 1000 if host["drain_end_ns"] else 0,
    )
    return {(kind if kind == "gap" else "host_" + kind, name): value
            for kind, values in groups.items() for name, value in values.items()}


def graph_host_sample(host, e2e_us, kernel_us, gets=1):
    sample = {("host_graph", key): value / 1000 for key, value in host.items()}
    # Host durations above are per replay; Event/kernel durations are per Get.
    sample.update({("host_graph_per_get", key): value / 1000 / gets for key, value in host.items()})
    sample.update({("gap", "e2e"): e2e_us, ("gap", "kernel"): kernel_us,
                   ("gap", "graph_event_outside_kernel"): e2e_us - kernel_us,
                   ("gap", "graph_host_outside_event"): host["replay_and_wait"] / 1000 / gets - e2e_us})
    return sample


def summarize_host_samples(samples):
    if not samples or any(sample.keys() != samples[0].keys() for sample in samples):
        raise AssertionError("Host profile sample fields do not match")
    details = []
    for kind, name in samples[0]:
        values = [sample[kind, name] for sample in samples]
        for metric, value in (("mean_us", statistics.mean(values)),
                              ("p50_us", percentile(values, .5)), ("p99_us", percentile(values, .99))):
            details.append((kind, name, metric, f"{value:.6f}"))
    return details

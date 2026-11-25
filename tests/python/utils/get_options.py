"""Workload and backend options shared by single- and multi-device Get runs."""

import argparse
import os
import sys

from .npu_eids import parse_npu_eids


BACKENDS = ("aiv", "ubmem", "gather_read", "cpu_push_scatter", "cpu_sgl_write", "aiv_cpu_sgl_write")
DEFAULT_PARALLELISM = {"aiv": None, "ubmem": 16, "gather_read": 5, "cpu_push_scatter": 5, "cpu_sgl_write": 5, "aiv_cpu_sgl_write": 4}
MAX_JETTYS_PER_CORE = 64
MAX_INT32 = (1 << 31) - 1
MAX_UINT64 = (1 << 64) - 1


def environment_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def environment_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def uses_aicpu(args):
    return args.backend in ("gather_read", "cpu_push_scatter", "cpu_sgl_write")


def parallelism_field(backend):
    return "aicpu_workers" if backend in ("gather_read", "cpu_push_scatter", "cpu_sgl_write") else "aiv_cores"


def add_workload_arguments(parser, *, defaults=None, request_aliases=()):
    defaults = defaults or {}

    def default(name, fallback):
        if name in defaults:
            return defaults[name]
        value = os.environ.get("KVAF_BATCH_GET_PERF_" + name.upper())
        return int(value) if value is not None else fallback

    group = parser.add_argument_group("Get workload (all backends)")
    group.add_argument("--request-length", *request_aliases, type=int,
                       default=default("request_length", 32768),
                       help="capacity of indices, offsets and output_indices")
    group.add_argument("--table-capacity", type=int, default=default("table_capacity", 16384 * 78),
                       help="number of valid source blocks in the AccessTable")
    group.add_argument("--tokens-per-block", type=int, default=default("tokens_per_block", 128))
    group.add_argument("--token-bytes", type=int, default=default("token_bytes", 656),
                       help="bytes transferred per active request")
    group.add_argument("--miss-count", type=int, default=default("miss_count", None),
                       help="active request-prefix length; defaults to request-length")
    group.add_argument("--source-access-pattern", choices=("random", "contiguous", "shuffled-contiguous"),
                       default=os.environ.get("KVAF_BATCH_GET_PERF_SOURCE_ACCESS_PATTERN", "random"),
                       help="remote reads: random over the full table, contiguous, or the same contiguous set shuffled")
    group.add_argument("--output-access-pattern", choices=("random", "contiguous"),
                       default=os.environ.get("KVAF_BATCH_GET_PERF_OUTPUT_ACCESS_PATTERN", "contiguous"))
    order = group.add_mutually_exclusive_group()
    order.add_argument("--sort-input-indices", dest="sort_input_indices", action="store_true",
                       help="sort active requests by source address, preserving the output mapping")
    order.add_argument("--no-sort-input-indices", dest="sort_input_indices", action="store_false")
    group.set_defaults(sort_input_indices=environment_bool("KVAF_BATCH_GET_PERF_SORT_INPUT_INDICES"))
    group.add_argument("--warmup", type=int, default=default("warmup", 3))
    group.add_argument("--seed", type=int, default=default("seed", 20260903))
    group.add_argument("--populate-chunk-entries", type=int, default=default("populate_chunk_entries", 128),
                       help="setup-only Put chunk size; does not change the Get workload")


def add_backend_arguments(parser, *, sweep=False):
    group = parser.add_argument_group("Get backend and execution resources")
    group.add_argument("--backend", choices=BACKENDS, help="Get implementation")
    group.add_argument("--transport", choices=("urma", "ubmem"),
                       help="compatibility selector: urma selects aiv, ubmem selects ubmem")
    count_options = dict(type=int, default=None)
    if sweep:
        count_options["nargs"] = "+"
    group.add_argument("--aiv-cores", **count_options,
                       help="AIV cores per device (1..48; defaults: local URMA 2, remote URMA 4, ubmem 16)")
    group.add_argument("--aicpu-workers", **count_options,
                       help="AICPU workers per device; gather_read/cpu_push_scatter/cpu_sgl_write only (1..8)")
    # Preserve old commands while keeping distinct hardware concepts in help.
    group.add_argument("--workers", dest="compat_workers", **count_options, help=argparse.SUPPRESS)
    group.add_argument("--jettys-per-core", type=int, default=None,
                       help="Jettys per AIV core; aiv only (1..64; defaults: local 2, remote 1)")
    group.add_argument("--max-inflight-total", type=int, default=None,
                       help="unconfirmed READ budget; aiv only (default: 2048; 0 disables the limit)")
    group.add_argument("--sge-per-wr", type=int,
                       default=environment_int("KVAF_BATCH_GET_PERF_SGE_PER_WR", 0),
                       help="maximum source SGEs per WRITE WR; cpu_sgl_write/aiv_cpu_sgl_write (0=auto, 1..16)")
    group.add_argument("--sgl-pipeline", action="store_true",
                       help="pipeline 1024-row chunks through construction, descriptor READ and concurrent WRITEs; CPU SGL Write backends only")
    group.add_argument("--npu-eids", type=parse_npu_eids,
                       default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
                       help="comma-separated NPU EIDs; AIV stripes across them, AICPU uses the first")
    group.add_argument("--peer-ub-device", default="", metavar="DEVICE[,DEVICE...]",
                       help="CPU UB device or ordered remote Direct Get FE list (URMA). "
                            "Automatic remote FE count: KVAF_DEFAULT_PEER_FE_COUNT (default 2).")
    return group


def resolve_backend(args, parser, *, default_backend="aiv", sweep=False):
    args.backend_explicit = args.backend is not None
    args.backend = args.backend or ({"urma": "aiv", "ubmem": "ubmem"}.get(args.transport) or default_backend)
    expected = "ubmem" if args.backend == "ubmem" else "urma"
    if args.transport is not None and args.transport != expected:
        parser.error("--backend conflicts with --transport")
    args.transport = expected
    field = parallelism_field(args.backend)
    other = "aiv_cores" if uses_aicpu(args) else "aicpu_workers"
    legacy = getattr(args, other)
    if legacy is not None:
        if uses_aicpu(args) and not sweep and args.aicpu_workers is None:
            print("Deprecated: use --aicpu-workers for AICPU; --aiv-cores is a legacy spelling here", file=sys.stderr)
            args.aicpu_workers, args.aiv_cores = legacy, None
        else:
            parser.error(f"--{other.replace('_', '-')} does not apply to --backend {args.backend}")
    selected, compat = getattr(args, field), args.compat_workers
    if selected is not None and compat is not None and selected != compat:
        parser.error("conflicting parallelism options")
    counts = selected if selected is not None else compat
    if counts is None:
        default_count = DEFAULT_PARALLELISM[args.backend]
        if sweep:
            env_name = "KVAF_BATCH_GET_AICPU_WORKERS" if uses_aicpu(args) else "KVAF_BATCH_GET_AIV_CORES"
            value = os.environ.get(env_name)
            counts = [default_count] if value is None else [int(v) for v in value.split()]
        else:
            env_name = "KVAF_BATCH_GET_PERF_AICPU_WORKERS" if uses_aicpu(args) else "KVAF_BATCH_GET_PERF_AIV_CORES"
            counts = environment_int(env_name, environment_int("KVAF_BATCH_GET_PERF_AIV_CORES", default_count))
    setattr(args, field, counts)
    args.parallelism = counts
    del args.compat_workers
    # Existing commands often passed these fixed values for every backend.
    # Keep them accepted, but reject attempts to tune an unsupported resource.
    if args.backend != "aiv":
        if args.jettys_per_core not in (None, 2):
            parser.error("--jettys-per-core is configurable only for --backend aiv")
        if args.max_inflight_total not in (None, 2048):
            parser.error("--max-inflight-total is configurable only for --backend aiv")
        args.jettys_per_core, args.max_inflight_total = 2, 2048
    else:
        if args.jettys_per_core is None:
            args.jettys_per_core = environment_int("KVAF_BATCH_GET_PERF_JETTYS_PER_CORE", None)
        if args.max_inflight_total is None:
            args.max_inflight_total = environment_int("KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL", 2048)
    if args.miss_count is None:
        args.miss_count = args.request_length


def validate_workload(args):
    if args.sgl_pipeline and args.backend not in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
        return "--sgl-pipeline requires a CPU SGL Write backend"
    if not 0 <= args.sge_per_wr <= 16:
        return "sge-per-wr must be in [0, 16] (0 selects the device limit)"
    if args.sge_per_wr and args.backend not in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
        return "sge-per-wr is configurable only for --backend cpu_sgl_write"
    if not 1 <= args.request_length <= 65536:
        return "request-length must be in [1, 65536]"
    if not 1 <= args.table_capacity <= MAX_INT32:
        return "table-capacity must be in [1, INT32_MAX]"
    if not 1 <= args.tokens_per_block <= MAX_INT32 or not 1 <= args.token_bytes <= MAX_INT32:
        return "tokens-per-block and token-bytes must be in [1, INT32_MAX]"
    if (args.tokens_per_block - 1) * args.token_bytes > MAX_INT32:
        return "the largest per-block token offset must fit int32"
    if args.table_capacity * args.tokens_per_block * args.token_bytes > MAX_UINT64:
        return "configured table size must fit uint64"
    if not 0 <= args.miss_count <= args.request_length:
        return "miss-count must be between zero and request-length"
    maximum = 8 if uses_aicpu(args) or args.backend == "aiv_cpu_sgl_write" else 48
    if args.parallelism is not None and not 1 <= args.parallelism <= maximum:
        return f"{parallelism_field(args.backend).replace('_', '-')} must be in [1, {maximum}]"
    if (uses_aicpu(args) or args.backend == "aiv_cpu_sgl_write") and max(args.token_bytes * 65536, args.token_bytes * args.tokens_per_block) > 0xFFFFFFFF:
        return "AICPU token workspace and block bytes must fit uint32"
    if args.transport == "ubmem" and args.peer_ub_device:
        return "peer-ub-device applies only to URMA"
    if args.jettys_per_core is not None and not 1 <= args.jettys_per_core <= MAX_JETTYS_PER_CORE:
        return "jettys-per-core must be in [1, 64]"
    if args.backend == "aiv":
        if (args.parallelism is not None and args.jettys_per_core is not None
                and len(args.npu_eids) > args.parallelism * args.jettys_per_core):
            return "npu-eids count must not exceed aiv-cores * jettys-per-core"
        if not 0 <= args.max_inflight_total <= 0xFFFFFFFF:
            return "max-inflight-total must fit uint32"
        if args.parallelism is not None and args.max_inflight_total and args.max_inflight_total < 2 * args.parallelism:
            return "max-inflight-total must be zero or at least twice aiv-cores"
    if args.backend in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
        if args.output_access_pattern != "contiguous":
            return "cpu_sgl_write requires --output-access-pattern contiguous"
        if args.sort_input_indices:
            return "cpu_sgl_write requires --no-sort-input-indices (output_indices must equal request positions)"
    if args.warmup < 0:
        return "warmup must be non-negative"
    if args.populate_chunk_entries <= 0 or min(args.populate_chunk_entries, args.table_capacity) > 65536:
        return "populate-chunk-entries must be positive and each Put batch must fit 65536 entries"
    return None


def workload_command(args):
    """Serialize exactly the workload/resources checked by the parent controller."""
    field = parallelism_field(args.backend)
    fields = ("backend", "request_length", "miss_count", "table_capacity", "token_bytes", "tokens_per_block",
              "populate_chunk_entries", "warmup", "seed", "source_access_pattern", "output_access_pattern")
    command = []
    for name in fields:
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    if args.parallelism is not None:
        command.extend(("--" + field.replace("_", "-"), str(args.parallelism)))
    command.append("--sort-input-indices" if args.sort_input_indices else "--no-sort-input-indices")
    if args.backend == "aiv":
        if args.jettys_per_core is not None:
            command += ["--jettys-per-core", str(args.jettys_per_core)]
        command += ["--max-inflight-total", str(args.max_inflight_total)]
    if args.backend in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
        command += ["--sge-per-wr", str(args.sge_per_wr)]
        if getattr(args, "sgl_pipeline", False):
            command += ["--sgl-pipeline"]
    if args.npu_eids:
        command += ["--npu-eids", ",".join(args.npu_eids)]
    if args.peer_ub_device:
        command += ["--peer-ub-device", args.peer_ub_device]
    return command


def configure_backend_help(parser, argv=None):
    """Show applicable backend options while retaining compatibility parsing."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--backend", choices=BACKENDS)
    selector.add_argument("--transport", choices=("urma", "ubmem"))
    selected, _ = selector.parse_known_args(argv)
    backend = selected.backend or {"urma": "aiv", "ubmem": "ubmem"}.get(selected.transport)
    if backend is None:
        return
    aicpu = backend in ("gather_read", "cpu_push_scatter", "cpu_sgl_write")
    hidden = set()
    hidden.add("aiv_cores" if aicpu else "aicpu_workers")
    if backend != "aiv":
        hidden.update(("jettys_per_core", "max_inflight_total"))
    if backend not in ("cpu_sgl_write", "aiv_cpu_sgl_write"):
        hidden.update(("sge_per_wr", "sgl_payload_only", "sgl_pipeline"))
    if backend == "ubmem":
        hidden.update(("npu_eids", "peer_ub_device"))
    if aicpu:
        hidden.add("kernel_profile_clock_mhz")
    for action in parser._actions:
        if action.dest in hidden:
            action.help = argparse.SUPPRESS

#!/usr/bin/env bash
# Sweep BatchGet with one/two streams and transports. Core counts are per stream.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_remote_cpu_batch_get_dual_stream_perf.sh /absolute/path/to/build-npu [--stream-count 1|2] [--transport-count 1|2] [batch-get-perf-options...]

Stream count defaults to KVAF_NPU_BATCH_GET_STREAM_COUNT or 2; --stream-count
takes precedence. Transport count defaults to KVAF_NPU_BATCH_GET_TRANSPORT_COUNT
or 1; two transports require two streams. Kernel profiling is enabled. Sweep values override the
corresponding forwarded options. Other options (e.g. --iterations,
--request-length, --npu-eids, --sort-input-indices) go to the existing benchmark.

Space-separated sweep lists (environment variables):
  KVAF_BATCH_GET_DUAL_STREAM_TOKEN_BYTES       default: 128 256 512 656 1024 2048 4096
  KVAF_BATCH_GET_DUAL_STREAM_MISS_COUNTS       default: 32 64 128 256 512 1024 2048 4096 8192
  KVAF_BATCH_GET_DUAL_STREAM_AIV_CORES         default: 1 2 4 8 16 24 (PER STREAM)
  KVAF_BATCH_GET_DUAL_STREAM_JETTYS_PER_CORE   default: 1 2 4 8 16 32 64

Other environment variables:
  KVAF_BATCH_GET_KERNEL_CLOCK_MHZ             default: 1000 (set for target device)
  KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR          default: a new directory under TMPDIR
  KVAF_BATCH_GET_DUAL_STREAM_CSV              default: LOG_DIR/batch_get_{single,dual}_stream_summary.csv
                                             two transports add _2_transports before .csv
  KVAF_CPU_SERVICE_ADDR, DEVICE_ID, KVAF_NPU_EIDS, KVAF_REMOTE_CPU_PERF_TIMEOUT
                                             same as run_remote_cpu_batch_get_perf.sh

Total cores = stream_count * AIV_CORES (maximum 48).
Total Jettys = stream_count * AIV_CORES * JETTYS_PER_CORE.
The existing DUAL_STREAM environment variable names apply to both modes.
MISS_COUNTS describes the whole batch shared by the selected streams. Bandwidth uses
miss_count * token_bytes and the earliest-start-to-latest-finish kernel wall time.
EOF
}

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
KVAF_BUILD_DIR="$1"
shift
stream_count="${KVAF_NPU_BATCH_GET_STREAM_COUNT-2}"
transport_count="${KVAF_NPU_BATCH_GET_TRANSPORT_COUNT-1}"
perf_options=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stream-count)
      [[ $# -ge 2 ]] || { echo "ERROR: --stream-count requires 1 or 2" >&2; exit 2; }
      stream_count="$2"
      shift 2
      ;;
    --stream-count=*) stream_count="${1#*=}"; shift ;;
    --transport-count)
      [[ $# -ge 2 ]] || { echo "ERROR: --transport-count requires 1 or 2" >&2; exit 2; }
      transport_count="$2"
      shift 2
      ;;
    --transport-count=*) transport_count="${1#*=}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) perf_options+=("$1"); shift ;;
  esac
done
[[ "$stream_count" == 1 || "$stream_count" == 2 ]] || {
  echo "ERROR: stream-count must be 1 or 2: $stream_count" >&2
  exit 2
}
[[ "$transport_count" == 1 || "$transport_count" == 2 ]] || {
  echo "ERROR: transport-count must be 1 or 2: $transport_count" >&2
  exit 2
}
if (( transport_count > stream_count )); then
  echo "ERROR: two transports require --stream-count 2" >&2
  exit 2
fi
# Restore forwarded arguments, including the no-options case on Bash 3.
if [[ ${#perf_options[@]} -gt 0 ]]; then set -- "${perf_options[@]}"; fi

KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_BATCH_GET_PERF_RUNNER="$KVAF_ROOT/tests/run_remote_cpu_batch_get_perf.sh"
[[ -x "$KVAF_BATCH_GET_PERF_RUNNER" ]] || {
  echo "ERROR: performance runner is not executable: $KVAF_BATCH_GET_PERF_RUNNER" >&2
  exit 1
}

read -r -a token_sizes <<< "${KVAF_BATCH_GET_DUAL_STREAM_TOKEN_BYTES-128 256 512 656 1024 2048 4096}"
read -r -a miss_counts <<< "${KVAF_BATCH_GET_DUAL_STREAM_MISS_COUNTS-32 64 128 256 512 1024 2048 4096 8192}"
read -r -a core_counts <<< "${KVAF_BATCH_GET_DUAL_STREAM_AIV_CORES-1 2 4 8 16 24}"
read -r -a jetty_counts <<< "${KVAF_BATCH_GET_DUAL_STREAM_JETTYS_PER_CORE-1 2 4 8 16 32 64}"
: "${KVAF_BATCH_GET_KERNEL_CLOCK_MHZ:=1000}"

validate_list() {
  local name="$1" minimum="$2" maximum="$3" value
  shift 3
  [[ $# -gt 0 && -n "$1" ]] || { echo "ERROR: $name must not be empty" >&2; exit 2; }
  for value in "$@"; do
    if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ || ${#value} -gt 10 ]] ||
      (( value < minimum || value > maximum )); then
      echo "ERROR: $name values must be integers in [$minimum, $maximum]: $value" >&2
      exit 2
    fi
  done
}
# Bash 3 treats empty arrays as unset under nounset; pass an empty value to the
# validator so a malformed list still receives a useful error.
validate_list KVAF_BATCH_GET_DUAL_STREAM_TOKEN_BYTES 1 2147483647 "${token_sizes[@]-}"
validate_list KVAF_BATCH_GET_DUAL_STREAM_MISS_COUNTS 0 2147483647 "${miss_counts[@]-}"
validate_list KVAF_BATCH_GET_DUAL_STREAM_AIV_CORES 1 "$((48 / stream_count))" "${core_counts[@]-}"
validate_list KVAF_BATCH_GET_DUAL_STREAM_JETTYS_PER_CORE 1 64 "${jetty_counts[@]-}"
validate_list KVAF_BATCH_GET_KERNEL_CLOCK_MHZ 1 2147483647 "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"

: "${KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR:=}"
if [[ -z "$KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR" ]]; then
  KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kvaf_batch_get_${stream_count}_stream.XXXXXX")"
else
  mkdir -p "$KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR"
fi
stream_label=dual
if [[ "$stream_count" == 1 ]]; then stream_label=single; fi
transport_suffix=""
if [[ "$transport_count" == 2 ]]; then transport_suffix="_2_transports"; fi
: "${KVAF_BATCH_GET_DUAL_STREAM_CSV:=$KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR/batch_get_${stream_label}_stream_summary${transport_suffix}.csv}"
mkdir -p "$(dirname "$KVAF_BATCH_GET_DUAL_STREAM_CSV")"
: > "$KVAF_BATCH_GET_DUAL_STREAM_CSV"
combinations=$((${#token_sizes[@]} * ${#miss_counts[@]} * ${#core_counts[@]} * ${#jetty_counts[@]}))
echo "BATCH_GET_DUAL_STREAM_LOG_DIR path=$KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR"
echo "BATCH_GET_DUAL_STREAM_CSV path=$KVAF_BATCH_GET_DUAL_STREAM_CSV"
echo "BATCH_GET_DUAL_STREAM_PLAN stream_count=$stream_count transport_count=$transport_count core_count_scope=per_stream combinations=$combinations clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"

# Append after every successful configuration so earlier results survive a later
# failure. Parse cycle fields rather than rounded millisecond profile values.
append_result() {
  python3 - "$1" "$KVAF_BATCH_GET_DUAL_STREAM_CSV" "$2" "$3" "$4" "$5" \
    "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" "$stream_count" "$transport_count" <<'PY'
import csv
import math
import os
import sys

log_path, csv_path = sys.argv[1:3]
token_bytes, miss_count, cores, jettys, clock, streams, transports = map(int, sys.argv[3:])
total_cores = streams * cores
records = {}
resources = {kind: [] for kind in ("BATCH_GET_TRANSPORT_READY", "BATCH_GET_TRANSPORT_CONTEXT", "BATCH_GET_TRANSPORT_WORKER")}
with open(log_path, encoding="utf-8", errors="replace") as source:
    for line in source:
        parts = line.split()
        if parts[:1] == ["[KVAF_AIV]"]:
            parts = parts[1:]
        if not parts:
            continue
        fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
        key = fields.get("operation") if parts[0] == "PERF_RESULT" else parts[0]
        records[key] = fields
        if key == "BATCH_GET_TRANSPORT_READY" and fields.get("transport_id") == "0":
            # Runtime may discard a candidate EID after an initialization probe
            # and retry. Validate only the final transport group used by the run.
            resources = {kind: [] for kind in resources}
        if key in resources:
            resources[key].append(fields)

def require(record, **expected):
    actual = records.get(record, {})
    for key, value in expected.items():
        if actual.get(key) != str(value):
            raise ValueError(f"{record}: expected {key}={value}, got {actual.get(key)!r}")

def number(record, field):
    value = float(records[record][field])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {record} {field}={value}")
    return value

try:
    require("BATCH_GET_STREAMS_READY", streams=streams, transport_count=transports,
            cores_per_stream=cores, total_cores=total_cores,
            jettys_per_core=jettys, total_jettys=total_cores * jettys)
    require("BATCH_GET_PERF_CONFIG", stream_count=streams, aiv_cores=cores,
            transport_count=transports,
            total_aiv_cores=total_cores, jettys_per_core=jettys,
            miss_count=miss_count, token_bytes=token_bytes)
    require("BATCH_GET_PERF_CORRECTNESS_PASS", miss_count=miss_count)
    require("KERNEL_PROFILE_CONTRACT", stream_count=streams, cores_per_stream=cores,
            transport_count=transports,
            total_aiv_cores=total_cores, jettys_per_core=jettys, clock_mhz=clock)
    paths = int(records["BATCH_GET_PERF_CONFIG"]["eid_paths"])
    ready = resources["BATCH_GET_TRANSPORT_READY"]
    contexts = resources["BATCH_GET_TRANSPORT_CONTEXT"]
    workers = resources["BATCH_GET_TRANSPORT_WORKER"]
    expected_contexts = {(t, p) for t in range(transports) for p in range(paths)}
    context_map = {(int(r["transport_id"]), int(r["eid_path"])): int(r["context"], 16) for r in contexts}
    if (paths <= 0 or len(ready) != transports or
            {int(r["transport_id"]) for r in ready} != set(range(transports)) or
            len(contexts) != len(expected_contexts) or set(context_map) != expected_contexts or
            len(set(context_map.values())) != len(expected_contexts) or 0 in context_map.values()):
        raise ValueError("missing or shared BatchGet transport contexts")
    for r in ready:
        if (int(r["workers"]) != total_cores * jettys // transports or
                int(r["cores_per_stream"]) != cores or int(r["eid_paths"]) != paths or
                int(r["first_stream"]) != (0 if transports == 1 else int(r["transport_id"])) or
                int(r["stream_count"]) != streams // transports):
            raise ValueError("unexpected worker allocation per transport")
    expected_workers = {(s, c, j) for s in range(streams) for c in range(cores) for j in range(jettys)}
    if (len(workers) != len(expected_workers) or
            {(int(r["stream"]), int(r["stream_core"]), int(r["jetty_slot"])) for r in workers} != expected_workers):
        raise ValueError("missing or duplicate BatchGet workers")
    for r in workers:
        transport_id = int(r["transport_id"])
        if (transport_id != (0 if transports == 1 else int(r["stream"])) or
                int(r["context"], 16) != context_map[(transport_id, int(r["eid_path"]))]):
            raise ValueError("worker bound to the wrong transport/context")
    for field in ("sq_va", "cq_va"):
        addresses = {int(r[field], 16) for r in workers}
        if len(addresses) != len(expected_workers) or 0 in addresses:
            raise ValueError(f"shared or missing worker {field}")
    # TP handles are logged for diagnosis; distinct context-scoped handles do
    # not prove independent physical TPs. Never infer TP isolation from them.
    row = dict(token_bytes=token_bytes, miss_count=miss_count, stream_count=streams,
               transport_count=transports, context_count=len(context_map),
               cores_per_stream=cores, total_aiv_cores=total_cores,
               jettys_per_core=jettys, total_jettys=total_cores * jettys,
               payload_bytes=miss_count * token_bytes, clock_mhz=clock,
               samples=int(records["batch_get_kernel_wall"]["samples"]))
    if row["samples"] <= 0:
        raise ValueError("no kernel profile samples")
    for stage in ("wall", "total", "post", "completion_wait"):
        operation = f"batch_get_kernel_{stage}"
        require(operation, samples=row["samples"])
        for stat in ("mean", "p50", "p90", "p99"):
            row[f"kernel_{stage}_{stat}_us"] = f"{number(operation, stat + '_cycles') / clock:.3f}"
    for stat in ("mean", "p50"):
        cycles = number("batch_get_kernel_wall", stat + "_cycles")
        if cycles <= 0:
            raise ValueError("kernel wall time must be positive")
        # One batch is split across streams; payload is independent of stream count.
        row[f"kernel_wall_{stat}_GBps"] = f"{row['payload_bytes'] * clock / cycles / 1000:.3f}"
    require("batch_get", samples=row["samples"], miss_count=miss_count)
    for stat in ("mean", "p50", "p90", "p99"):
        row[f"host_{stat}_us"] = f"{number('batch_get', stat + '_ms') * 1000:.3f}"
except (KeyError, ValueError) as error:
    sys.exit(f"ERROR: incomplete or inconsistent {streams}-stream results in {log_path}: {error}")

with open(csv_path, "a", newline="", encoding="utf-8") as destination:
    writer = csv.DictWriter(destination, fieldnames=list(row))
    if os.path.getsize(csv_path) == 0:
        writer.writeheader()
    writer.writerow(row)
print("BATCH_GET_DUAL_STREAM_RESULT " + " ".join(f"{key}={value}" for key, value in row.items()))
PY
}

completed=0
for token_bytes in "${token_sizes[@]}"; do
  for miss_count in "${miss_counts[@]}"; do
    for cores in "${core_counts[@]}"; do
      for jettys in "${jetty_counts[@]}"; do
        log_file="$KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR/stream_count_${stream_count}_transport_count_${transport_count}_token_bytes_${token_bytes}_miss_count_${miss_count}_cores_per_stream_${cores}_jettys_per_core_${jettys}.log"
        echo "BATCH_GET_DUAL_STREAM_BEGIN run=$((completed + 1))/$combinations stream_count=$stream_count transport_count=$transport_count token_bytes=$token_bytes miss_count=$miss_count cores_per_stream=$cores total_aiv_cores=$((stream_count * cores)) jettys_per_core=$jettys total_jettys=$((stream_count * cores * jettys))"
        if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" "$@" \
          --stream-count "$stream_count" --token-bytes "$token_bytes" --miss-count "$miss_count" \
          --transport-count "$transport_count" \
          --aiv-cores "$cores" --jettys-per-core "$jettys" \
          --kernel-profile --kernel-profile-clock-mhz "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
          2>&1 | tee "$log_file"; then
          append_result "$log_file" "$token_bytes" "$miss_count" "$cores" "$jettys"
        else
          status=$?
          echo "BATCH_GET_DUAL_STREAM_FAILED status=$status completed=$completed log=$log_file csv=$KVAF_BATCH_GET_DUAL_STREAM_CSV" >&2
          exit "$status"
        fi
        completed=$((completed + 1))
      done
    done
  done
done

python3 - "$KVAF_BATCH_GET_DUAL_STREAM_CSV" "$stream_count" "$transport_count" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as source:
    rows = list(csv.DictReader(source))
print(f"\nBATCH_GET_DUAL_STREAM_SUMMARY combinations={len(rows)} stream_count={sys.argv[2]} transport_count={sys.argv[3]}")
print("timing=earliest_kernel_start_to_latest_kernel_finish units=us bandwidth=decimal_GB/s")
columns = ("token_bytes", "miss_count", "stream_count", "transport_count", "cores_per_stream", "total_aiv_cores",
           "jettys_per_core", "total_jettys", "kernel_wall_mean_us", "kernel_wall_p50_us",
           "kernel_wall_p90_us", "kernel_wall_p99_us", "kernel_wall_p50_GBps", "host_p50_us")
print(" ".join(columns))
for row in rows:
    print(" ".join(row[column] for column in columns))
print(f"BATCH_GET_DUAL_STREAM_CSV_WRITTEN path={sys.argv[1]} rows={len(rows)}")
PY

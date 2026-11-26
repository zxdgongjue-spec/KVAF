#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_multi_client_bw.sh --demo PATH --mapping FILE --operation read|write [options]

Required mapping format (one client/NPU/RDMA NIC/server endpoint per line):
  label  acl_device_id  client_rdma_device  gid_index  server_host:port
  Blank lines and lines beginning with # are ignored. Fields are whitespace separated.

Options:
  --rdma-port N    Local RDMA port (default: 1)
  --bytes N        Bytes per WR (default: 1048576)
  --depth N        Outstanding WRs per QP (default: 16)
  --qp-count N     QPs per client/NPU/RDMA NIC (default: 1, maximum: 32)
  --warmup N       Warmup WRs per client (default: 10)
  --iterations N   Measured WRs per client (default: 1000)
  --log-dir PATH   Output directory (default: host-roce-bw-TIMESTAMP)
  --help

Each mapping row runs as one client process. --qp-count controls QPs within that
client's mapped NPU/RDMA link. Start one matching,
one-shot CPU service process per row before running this script. Run the script once
for WRITE and once for READ (restart the server processes between runs).
USAGE
}

die() { printf '[multi-roce-bw] ERROR: %s\n' "$*" >&2; exit 2; }

demo=""
mapping=""
operation=""
rdma_port=1
bytes=1048576
depth=16
qp_count=1
warmup=10
iterations=1000
log_dir=""

while (($#)); do
  case "$1" in
    --demo|--mapping|--operation|--rdma-port|--bytes|--depth|--qp-count|--warmup|--iterations|--log-dir)
      (($# >= 2)) || die "missing value for $1"
      option="$1"; value="$2"; shift 2
      case "$option" in
        --demo) demo="$value" ;;
        --mapping) mapping="$value" ;;
        --operation) operation="$value" ;;
        --rdma-port) rdma_port="$value" ;;
        --bytes) bytes="$value" ;;
        --depth) depth="$value" ;;
        --qp-count) qp_count="$value" ;;
        --warmup) warmup="$value" ;;
        --iterations) iterations="$value" ;;
        --log-dir) log_dir="$value" ;;
      esac
      ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$demo" && -x "$demo" ]] || die "--demo must name an executable client binary"
[[ -n "$mapping" && -r "$mapping" ]] || die "--mapping must name a readable mapping file"
[[ "$operation" == read || "$operation" == write ]] || die "--operation must be read or write"

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
for name in rdma_port bytes depth qp_count warmup iterations; do
  is_uint "${!name}" || die "$name must be an unsigned integer"
done
rdma_port=$((10#$rdma_port))
bytes=$((10#$bytes))
depth=$((10#$depth))
qp_count=$((10#$qp_count))
warmup=$((10#$warmup))
iterations=$((10#$iterations))
((rdma_port >= 1 && rdma_port <= 255)) || die "RDMA port must be in [1,255]"
((bytes >= 1 && bytes <= 4294967295)) || die "bytes must be in [1, UINT32_MAX]"
((depth >= 1 && depth <= 128)) || die "depth must be in [1,128]"
((qp_count >= 1 && qp_count <= 32)) || die "qp-count must be in [1,32]"
((iterations >= 1)) || die "iterations must be greater than zero"
((bytes * depth * qp_count <= 1073741824)) || die "bytes * depth * qp-count exceeds the client's 1 GiB buffer limit"

declare -a labels=() device_ids=() rdma_devices=() gids=() servers=()
declare -A seen_labels=() seen_devices=() seen_rdma=() seen_servers=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]] && continue
  label=""; device_id=""; rdma_device=""; gid=""; server=""; extra=""
  read -r label device_id rdma_device gid server extra <<< "$line"
  [[ -n "${label:-}" && -n "${device_id:-}" && -n "${rdma_device:-}" &&
     -n "${gid:-}" && -n "${server:-}" && -z "${extra:-}" ]] ||
    die "each mapping row must have exactly five whitespace-separated fields: $line"
  is_uint "$device_id" && is_uint "$gid" || die "invalid device id or GID index in row: $line"
  ((10#$gid <= 255)) || die "GID index must be in [0,255]: $line"
  [[ "$server" =~ ^[^[:space:]]+:[0-9]+$ ]] || die "server endpoint must be host:port: $line"
  [[ -z "${seen_labels[$label]+x}" ]] || die "duplicate label: $label"
  [[ -z "${seen_devices[$device_id]+x}" ]] || die "duplicate ACL device id: $device_id"
  [[ -z "${seen_rdma[$rdma_device]+x}" ]] || die "duplicate client RDMA device: $rdma_device"
  [[ -z "${seen_servers[$server]+x}" ]] || die "duplicate server endpoint: $server"
  seen_labels[$label]=1; seen_devices[$device_id]=1
  seen_rdma[$rdma_device]=1; seen_servers[$server]=1
  labels+=("$label"); device_ids+=("$device_id"); rdma_devices+=("$rdma_device")
  gids+=("$gid"); servers+=("$server")
done < "$mapping"

count=${#labels[@]}
((count > 0)) || die "mapping file contains no client rows"
if [[ -z "$log_dir" ]]; then
  log_dir="host-roce-bw-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$log_dir" || die "cannot create log directory: $log_dir"

operation_upper="${operation^^}"
declare -a pids=() log_files=() exit_codes=()
printf '[multi-roce-bw] operation=%s clients=%d bytes_per_wr=%s depth_per_qp=%s qp_count_per_client=%s warmup=%s iterations=%s\n' \
  "$operation_upper" "$count" "$bytes" "$depth" "$qp_count" "$warmup" "$iterations"
printf '[multi-roce-bw] per-client HBM/DDR region: %s bytes; aggregate region request: %s bytes\n' \
  "$((bytes * depth * qp_count))" "$((bytes * depth * qp_count * count))"

for ((i=0; i<count; ++i)); do
  log_file="$log_dir/${labels[i]}.log"
  log_files+=("$log_file")
  printf '[multi-roce-bw] START label=%s acl_device=%s rdma_device=%s gid=%s server=%s log=%s\n' \
    "${labels[i]}" "${device_ids[i]}" "${rdma_devices[i]}" "${gids[i]}" "${servers[i]}" "$log_file"
  "$demo" "${device_ids[i]}" "${servers[i]}" \
    --rdma-device "${rdma_devices[i]}" --rdma-port "$rdma_port" --gid-index "${gids[i]}" \
    --bytes "$bytes" --depth "$depth" --qp-count "$qp_count" \
    --warmup "$warmup" --iterations "$iterations" --operation "$operation" \
    >"$log_file" 2>&1 &
  pids+=("$!")
done

failed=0
for ((i=0; i<count; ++i)); do
  if wait "${pids[i]}"; then
    exit_codes+=(0)
    printf '[multi-roce-bw] DONE label=%s status=0\n' "${labels[i]}"
  else
    rc=$?
    exit_codes+=("$rc")
    failed=1
    printf '[multi-roce-bw] DONE label=%s status=%s (see %s)\n' "${labels[i]}" "$rc" "${log_files[i]}" >&2
  fi
done

summary="$log_dir/summary.tsv"
printf 'label\tacl_device\tclient_rdma_device\tserver\tstatus\tGBps\tGiBps\tGbps\tlog\n' > "$summary"
sum_gbps=0
sum_gibps=0
sum_gbits=0
for ((i=0; i<count; ++i)); do
  gbps="-"; gibps="-"; gbits="-"
  if [[ "${exit_codes[i]}" == 0 ]]; then
    result=$(awk -v expected="operation=$operation_upper" '
      $1 == "PERF_RESULT" {
        matched = 0; gb = ""; gi = ""; gbit = ""
        for (i = 2; i <= NF; ++i) {
          if ($i == expected) matched = 1
          if ($i ~ /^GBps=/) { sub(/^GBps=/, "", $i); gb = $i }
          if ($i ~ /^GiBps=/) { sub(/^GiBps=/, "", $i); gi = $i }
          if ($i ~ /^Gbps=/) { sub(/^Gbps=/, "", $i); gbit = $i }
        }
        if (matched && gb != "" && gi != "" && gbit != "") {
          ++matches; values = gb "\t" gi "\t" gbit
        }
      }
      END { if (matches != 1) exit 1; print values }
    ' "${log_files[i]}") || result=""
    if [[ -n "$result" ]]; then
      IFS=$'\t' read -r gbps gibps gbits <<< "$result"
      sum_gbps=$(awk -v a="$sum_gbps" -v b="$gbps" 'BEGIN { printf "%.6f", a + b }')
      sum_gibps=$(awk -v a="$sum_gibps" -v b="$gibps" 'BEGIN { printf "%.6f", a + b }')
      sum_gbits=$(awk -v a="$sum_gbits" -v b="$gbits" 'BEGIN { printf "%.6f", a + b }')
    else
      exit_codes[i]="no_perf_result"
      failed=1
      printf '[multi-roce-bw] ERROR label=%s exited successfully but has no unique %s PERF_RESULT\n' \
        "${labels[i]}" "$operation_upper" >&2
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${labels[i]}" "${device_ids[i]}" "${rdma_devices[i]}" "${servers[i]}" \
    "${exit_codes[i]}" "$gbps" "$gibps" "$gbits" "${log_files[i]}" >> "$summary"
done

printf '\n[multi-roce-bw] SUMMARY operation=%s successful_links=' "$operation_upper"
successes=0
for status in "${exit_codes[@]}"; do [[ "$status" == 0 ]] && ((successes+=1)); done
printf '%d/%d aggregate_GBps=%s aggregate_GiBps=%s aggregate_Gbps=%s\n' \
  "$successes" "$count" "$sum_gbps" "$sum_gibps" "$sum_gbits"
printf '[multi-roce-bw] per-link summary: %s\n' "$summary"
if ((failed)); then
  printf '[multi-roce-bw] INCOMPLETE: inspect failed client logs; aggregate includes only clients with valid results.\n' >&2
  exit 1
fi

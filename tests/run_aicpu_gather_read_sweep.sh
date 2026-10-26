#!/usr/bin/env bash
# Sweep AICPU GatherRead; the Python runner calls NpuRuntime.batch_get.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 build-aicpu-urma [benchmark-options...]" >&2
  exit 2
fi

build_dir="$1"
shift
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$repo_root/access_operator/demo/aicpu_batch_get/python/benchmark_gather_read.py"
export PYTHONPATH="$build_dir/python${PYTHONPATH:+:$PYTHONPATH}"

read -r -a token_sizes <<< "${KVAF_AICPU_TOKEN_BYTES:-128 256 512 656 1024 2048 4096}"
read -r -a miss_counts <<< "${KVAF_AICPU_MISS_COUNTS:-32 64 128 256 512 1024 2048 4096 8192}"
read -r -a worker_counts <<< "${KVAF_AICPU_WORKERS:-1 2 4 8}"
log_dir="${KVAF_AICPU_SWEEP_LOG_DIR:-reports/aicpu-gather-read/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$(dirname "$log_dir")"
mkdir "$log_dir"
csv_file="$log_dir/summary.csv"
total=$((${#token_sizes[@]} * ${#miss_counts[@]} * ${#worker_counts[@]}))
index=0

for token in "${token_sizes[@]}"; do
  for count in "${miss_counts[@]}"; do
    for workers in "${worker_counts[@]}"; do
      index=$((index + 1))
      log_file="$log_dir/token_${token}_miss_${count}_workers_${workers}.log"
      echo "[$index/$total] token_bytes=$token miss_count=$count workers=$workers log=$log_file"
      if "${PYTHON:-python3}" "$runner" "$@" \
          --token-bytes "$token" --miss-count "$count" --workers "$workers" \
          >"$log_file" 2>&1; then
        result="$(awk '$1 == "AICPU_GATHER_READ_RESULT" { print; found=1 }
                       END { if (!found) exit 1 }' "$log_file")"
      else
        tail -n 30 "$log_file" >&2
        exit 1
      fi
      echo "$result"
      printf '%s\n' "$result" | awk -v first="$index" '
        {
          header=""; row=""
          for (i=2; i<=NF; i++) {
            split($i, field, "=")
            sep=(i==2 ? "" : ",")
            header=header sep field[1]; row=row sep field[2]
          }
          if (first==1) print header
          print row
        }' >>"$csv_file"
    done
  done
done

awk -F, '
  NR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    print "token_bytes miss_count workers mean_us p50_us p90_us p99_us p50_GBps"
    next
  }
  { print $(col["token_bytes"]), $(col["miss_count"]), $(col["workers"]),
          $(col["mean_us"]), $(col["p50_us"]), $(col["p90_us"]),
          $(col["p99_us"]), $(col["p50_GBps"]) }
' "$csv_file"
echo "AICPU_GATHER_READ_SWEEP_PASS cases=$index csv=$csv_file"

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
read -r -a worker_counts <<< "${KVAF_AICPU_WORKERS:-1 2 4 5}"
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
      awk '/^AICPU_GATHER_READ_PROFILE / { profile=1 }
           /^AICPU_GATHER_READ_RESULT / { profile=0 }
           profile { print }' "$log_file"
      printf '%s\n' "$result" | awk -v first="$index" '
        BEGIN {
          label["token_bytes"]="单次读取(B)"
          label["miss_count"]="请求数"
          label["workers"]="worker数"
          label["device_p50_us"]="算子P50(us)"
          label["p50_GBps"]="p50_GBps"
          label["request_us"]="请求准备_含缓存维护和查表_均值(us)"
          label["source_write_us"]="发送地址清单_均值(us)"
          label["notify_us"]="通知CPU_均值(us)"
          label["read_us"]="流水READ总计_均值(us)"
          label["wr_build_us"]="READ内_组装WR及SGL_均值(us)"
          label["ready_wait_us"]="READ内_等待CPU就绪_均值(us)"
          label["post_us"]="READ内_提交READ_均值(us)"
          label["cq_wait_us"]="READ内_等待CQE_均值(us)"
          label["ack_us"]="等待CPU最终ACK_均值(us)"
        }
        {
          header=""; row=""
          for (i=2; i<=NF; i++) {
            split($i, field, "=")
            sep=(i==2 ? "" : ",")
            header=header sep label[field[1]]; row=row sep field[2]
          }
          if (first==1 && NR==1) print header
          print row
        }' >>"$csv_file"
    done
  done
done

awk -F, '
  NR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    print "单次读取(B) 请求数 worker数 算子P50(us) p50_GBps"
    next
  }
  {
    print $(col["单次读取(B)"]), $(col["请求数"]), $(col["worker数"]),
          $(col["算子P50(us)"]), $(col["p50_GBps"])
  }
' "$csv_file"
echo "AICPU_GATHER_READ_SWEEP_PASS cases=$index csv=$csv_file"

#!/usr/bin/env bash
# Diagnose fast paths without changing post routing, CQ polling or WQE filling.
# Builds must already exist; this script never builds or changes the CPU service.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${1:-$root}"
log_root="${2:-$root/logs/batch-get-fast-path-$(date +%Y%m%d-%H%M%S)}"
rounds="${KVAF_FAST_PATH_ROUNDS:-6}"
warmup="${KVAF_FAST_PATH_WARMUP:-100}"
iterations="${KVAF_FAST_PATH_ITERATIONS:-2000}"
clock_mhz="${KVAF_FAST_PATH_CLOCK_MHZ:-1000}"
python_bin="${PYTHON:-python3}"
die() { echo "ERROR: $*" >&2; exit 1; }
for value in "$rounds" "$iterations" "$clock_mhz"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "rounds, iterations and clock MHz must be positive integers"
done
[[ "$warmup" =~ ^(0|[1-9][0-9]*)$ ]] || die "warmup must be a non-negative integer"
[[ -n "${KVAF_CPU_SERVICE_ADDR:-}" ]] || die "export KVAF_CPU_SERVICE_ADDR before running"
[[ -n "${DEVICE_ID:-}" ]] || die "export DEVICE_ID before running"
command -v "$python_bin" >/dev/null || die "Python interpreter not found: $python_bin"
if [[ -d "$log_root" && -n "$(find "$log_root" -type f -name 'r*_n*_*.log' -print -quit)" ]]; then
  die "log directory already contains benchmark logs; choose a new directory: $log_root"
fi

check_cache() {
  local build="$1" key="$2" expected="$3" value
  value="$(awk -F= -v key="$key" '$1 ~ "^" key ":(BOOL|STRING)$" {value=$2} END {print value}' "$build/CMakeCache.txt")"
  case "$value" in
    ON|1) value=1 ;;
    OFF|0) value=0 ;;
  esac
  [[ "$value" == "$expected" ]] ||
    die "$build: $key=$value, expected explicit $expected; rebuild with the documented switches"
}
for variant in trim group post both; do
  build="$build_root/build-npu-fast-$variant"
  [[ -f "$build/CMakeCache.txt" && -f "$build/bin/kvaf_execute_batch_get_kernels.o" &&
     -d "$build/python/kvaf" ]] || die "missing NPU build: $build"
  group_fast=0
  post_fast=0
  if [[ "$variant" == group || "$variant" == both ]]; then group_fast=1; fi
  if [[ "$variant" == post || "$variant" == both ]]; then post_fast=1; fi
  check_cache "$build" KVAF_BATCH_GET_LEAN_SCHEDULE 1
  check_cache "$build" KVAF_BATCH_GET_MIN_TAIL_READS 0
  check_cache "$build" KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH "$group_fast"
  check_cache "$build" KVAF_BATCH_GET_SINGLE_POST_FAST_PATH "$post_fast"
done

# All configured Jettys carry READs. The post-only variant intentionally also
# runs cases with multiple posts: enabled=1, actual single-post calls must be 0.
cases=("32 1 1" "64 1 1" "128 1 1" "256 1 2" "512 1 2"
       "512 1 4" "1024 1 4" "1024 1 8" "64 2 1" "128 2 1"
       "256 2 1" "512 2 2" "1024 2 2")
for candidate in group post both; do
  pair_dir="$log_root/$candidate"
  mkdir -p "$pair_dir"
  proof_checked=0
  for ((round = 1; round <= rounds; ++round)); do
    variants=(baseline lean)
    if ((round % 2 == 0)); then variants=(lean baseline); fi
    for shape in "${cases[@]}"; do
      read -r n core jetty <<< "$shape"
      for variant in "${variants[@]}"; do
        build_variant=trim
        if [[ "$variant" == lean ]]; then build_variant="$candidate"; fi
        log="$pair_dir/r${round}_n${n}_c${core}_j${jetty}_${variant}.log"
        echo "[FAST_PATH_AB] candidate=$candidate round=$round miss_count=$n cores=$core jettys=$jetty variant=$variant"
        if ! bash "$root/tests/run_remote_cpu_batch_get_perf.sh" "$build_root/build-npu-fast-$build_variant" \
          --request-length 32768 --table-capacity 16384 \
          --tokens-per-block 128 --token-bytes 656 --miss-count "$n" \
          --aiv-cores "$core" --jettys-per-core "$jetty" --max-inflight-total 2048 \
          --source-access-pattern random --output-access-pattern contiguous \
          --no-sort-input-indices --seed 20260903 \
          --warmup "$warmup" --iterations "$iterations" --kernel-profile \
          --kernel-profile-clock-mhz "$clock_mhz" > "$log" 2>&1; then
          die "benchmark failed; inspect $log"
        fi
      done
      if [[ "$proof_checked" == 0 ]]; then
        "$python_bin" "$root/tests/diagnostics/summarize_batch_get.py" "$pair_dir" \
          --fast-path-ab --candidate-fast-paths "$candidate" --validate-only
        proof_checked=1
      fi
    done
  done
  "$python_bin" "$root/tests/diagnostics/summarize_batch_get.py" "$pair_dir" \
    --fast-path-ab --candidate-fast-paths "$candidate" --output-dir "$pair_dir/csv"
done
echo "[FAST_PATH_AB] complete: $log_root (three comparisons, six CSV files each)"

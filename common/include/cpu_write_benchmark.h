#pragma once
#include <cstdint>

namespace kvaf::cpu_push {
constexpr std::uint32_t kWriteBenchmarkSamples = 4096;
constexpr std::uint32_t kWriteBenchmarkMaxSges = 1U << 20;
// Shared deterministic pattern and random-fragment mapping. Payload validation
// is performed independently by the NPU controller after the measured writes.
inline std::uint64_t WriteBenchmarkMix(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}
#pragma pack(push, 1)
struct WriteBenchmarkOptions {
  std::uint64_t payload_bytes{}, source_bytes{}, seed{};
  // source_mode: 0 = contiguous, 1 = random SGL fragments.
  std::uint32_t message_bytes{}, fragment_bytes{}, queues{4}, post_batch_wrs{128};
  std::uint32_t warmup{20}, iterations{100}, source_mode{};
};
struct WriteBenchmarkSample {
  std::uint64_t comm_ns{}, post_ns{}, cq_wait_ns{}, tail_cqes{};
};
struct WriteBenchmarkResult {
  std::uint64_t hbm_address{}, payload_bytes{}, source_bytes{}, data_wrs{}, source_sges{};
  std::uint32_t message_bytes{}, max_sge{}, queues{}, post_batch_wrs{}, waves{}, optimized{}, count{};
  WriteBenchmarkSample samples[kWriteBenchmarkSamples]{};
};
// Separate diagnostic RPC keeps the original benchmark wire ABI unchanged.
struct WriteBenchmarkDiagnosticOptions {
  std::uint64_t source_window_bytes{}, target_window_bytes{}, flight_bytes{};
  std::int32_t cpu{-1};
  std::uint32_t idle_us{}, reserved{};
};
struct WriteBenchmarkDiagnosticSample {
  std::uint64_t thread_cpu_ns{}, voluntary_switches{}, involuntary_switches{}, minor_faults{}, major_faults{};
  std::int32_t cpu_start{-1}, cpu_end{-1};
  std::uint64_t inter_sample_gap_ns{};
};
struct WriteBenchmarkDiagnostics {
  std::uint64_t source_window_bytes{}, target_window_bytes{}, flight_bytes{};
  std::int32_t cpu_requested{-1};
  std::uint32_t idle_us{};
  // JSON snapshot: NUMA/page information and case-wide cgroup v2 counters.
  char environment[2048]{};
  WriteBenchmarkDiagnosticSample samples[kWriteBenchmarkSamples]{};
};
#pragma pack(pop)
static_assert(sizeof(WriteBenchmarkOptions) == 52);
static_assert(sizeof(WriteBenchmarkSample) == 32);
static_assert(sizeof(WriteBenchmarkResult) == 68 + 32 * kWriteBenchmarkSamples);
static_assert(sizeof(WriteBenchmarkDiagnosticOptions) == 36);
static_assert(sizeof(WriteBenchmarkDiagnosticSample) == 56);
static_assert(sizeof(WriteBenchmarkDiagnostics) == 2080 + 56 * kWriteBenchmarkSamples);
}  // namespace kvaf::cpu_push

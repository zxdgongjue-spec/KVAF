#pragma once

#include "abi.h"
#include "scatter_copy.h"
#include "scatter_probe_pmu.h"
#include <cstring>

namespace kvaf::aicpu_test {
constexpr char kScatterProbeKernelName[] = "KvafAicpuScatterProbeV1";
constexpr std::uint64_t kScatterProbeMagic = 0x4b56414653435031ULL;
constexpr std::uint64_t kScatterProbeRevision = 1;
constexpr std::uint64_t kMaxScatterProbeRounds = 2000;

enum class ScatterProbeMode : std::uint64_t {
  kPipeline, kPhased, kSerialCopy, kCopyFenced, kPacked, kPackedSerial,
  kRead, kReadSerial, kWrite, kWriteSerial, kCompute, kComputeSerial, kCount
};
inline const char* ScatterProbeModeName(ScatterProbeMode mode) {
  switch (mode) {
    case ScatterProbeMode::kPipeline: return "pipeline";
    case ScatterProbeMode::kPhased: return "phased";
    case ScatterProbeMode::kSerialCopy: return "serial_copy";
    case ScatterProbeMode::kCopyFenced: return "copy_fenced";
    case ScatterProbeMode::kPacked: return "packed";
    case ScatterProbeMode::kPackedSerial: return "packed_serial";
    case ScatterProbeMode::kRead: return "read";
    case ScatterProbeMode::kReadSerial: return "read_serial";
    case ScatterProbeMode::kWrite: return "write";
    case ScatterProbeMode::kWriteSerial: return "write_serial";
    case ScatterProbeMode::kCompute: return "compute";
    case ScatterProbeMode::kComputeSerial: return "compute_serial";
    default: return "invalid";
  }
}
inline bool ProbePacked(ScatterProbeMode m) {
  return m == ScatterProbeMode::kPacked || m == ScatterProbeMode::kPackedSerial;
}
inline bool ProbeSerial(ScatterProbeMode m) {
  return m == ScatterProbeMode::kSerialCopy || m == ScatterProbeMode::kPackedSerial ||
      m == ScatterProbeMode::kReadSerial || m == ScatterProbeMode::kWriteSerial ||
      m == ScatterProbeMode::kComputeSerial;
}
inline bool ProbeReadsOnly(ScatterProbeMode m) {
  return m == ScatterProbeMode::kRead || m == ScatterProbeMode::kReadSerial;
}
inline bool ProbeWritesOnly(ScatterProbeMode m) {
  return m == ScatterProbeMode::kWrite || m == ScatterProbeMode::kWriteSerial;
}
inline bool ProbeComputesOnly(ScatterProbeMode m) {
  return m == ScatterProbeMode::kCompute || m == ScatterProbeMode::kComputeSerial;
}
inline bool ProbeCopies(ScatterProbeMode m) {
  return !ProbeReadsOnly(m) && !ProbeWritesOnly(m) && !ProbeComputesOnly(m);
}

struct ScatterProbeConfig {
  std::uint64_t magic{kScatterProbeMagic};
  std::uint64_t revision{kScatterProbeRevision};
  std::uint64_t workers{4};
  std::uint64_t rows_per_worker{1024};
  std::uint64_t row_bytes{656};
  std::uint64_t warmup{5};
  std::uint64_t iterations{200};
  std::uint64_t timeout_ms{1000};
  ScatterProbeMode mode{ScatterProbeMode::kPipeline};
  std::uint64_t pmu{};
};
struct ScatterProbeLayout {
  std::uint64_t rows{}, stride{}, source_bytes{}, output_bytes{};
};
// Fixed per-worker work, cache-line-disjoint ranges, and bounded allocations.
// Read/write probes use exact 16-byte accesses; they do not read row padding.
inline bool MakeScatterProbeLayout(const ScatterProbeConfig& c, ScatterProbeLayout* l) {
  if (!l || c.magic != kScatterProbeMagic || c.revision != kScatterProbeRevision ||
      !c.workers || c.workers > kMaxScatterThreads || !c.rows_per_worker ||
      c.rows_per_worker > 65536 || c.row_bytes < 16 || c.row_bytes > 65536 ||
      c.row_bytes % 16 || c.rows_per_worker % RowsPerCacheGroup(c.row_bytes) ||
      !c.iterations || c.iterations > kMaxScatterProbeRounds ||
      c.warmup > kMaxScatterProbeRounds - c.iterations ||
      !c.timeout_ms || c.timeout_ms > 10000 || c.mode >= ScatterProbeMode::kCount ||
      c.pmu > kScatterProbePmuMask) return false;
  const auto rows = c.workers * c.rows_per_worker;
  const auto stride = ProbePacked(c.mode) ? c.row_bytes : ((c.row_bytes + 63) & ~63ULL) + 64;
  if (rows > (kMaxAllocation - 2 * kGuardBytes) / stride) return false;
  *l = {rows, stride, rows * c.row_bytes, rows * stride};
  return true;
}

// A separate cache-line-aligned record for every (round, worker). Raw intervals
// allow per-round overlap analysis; max(mean) is not mean(max).
struct alignas(64) ScatterProbeSample {
  std::uint64_t invalidate_begin{}, invalidate_issued{}, invalidate_end{};
  std::uint64_t op_begin{}, op_end{}, drain_begin{}, drain_end{};
  std::uint64_t clean_begin{}, clean_issued{}, clean_end{};
  std::uint64_t op_cpu_ns{}, cpu_time_errno{}, cpu_sample_errno{};
  std::int64_t first_cpu{-1}, last_cpu{-1};
  std::uint64_t checksum{};
  ScatterProbePmuSample pmu;
};
static_assert(sizeof(ScatterProbeSample) == 256);
struct alignas(64) ScatterProbeResult {
  std::uint64_t magic{kScatterProbeMagic}, revision{kScatterProbeRevision};
  std::uint64_t counter_frequency{}, completed_workers{};
  std::uint64_t reserved[4]{};
  CoreProbeSlot placement[kMaxScatterThreads];
};
struct alignas(64) ScatterProbeCoordination {
  CoordinationWord claimed, status, exited;
  CoordinationWord arrived[kMaxScatterThreads];
};
struct ScatterProbeArgs {
  std::uint64_t source{}, output{}, samples{}, result{}, coordination{};
  ScatterProbeConfig config;
};
static_assert(sizeof(ScatterProbeArgs) == 120 && sizeof(ScatterProbeResult) == 8256);

// Read-only checksum: XOR each byte lane of every 16-byte source vector. The
// scalar reduction is outside the operation interval on the device.
inline void ProbeReadVectors(const std::uint8_t* src, std::uint64_t bytes, std::uint8_t* checksum) {
#if defined(__aarch64__)
  auto value = vdupq_n_u8(0);
  for (std::uint64_t i = 0; i < bytes; i += 16) value = veorq_u8(value, vld1q_u8(src + i));
  vst1q_u8(checksum, value);
#else
  std::memset(checksum, 0, 16);
  for (std::uint64_t i = 0; i < bytes; ++i) checksum[i % 16] ^= src[i];
#endif
}
inline void ProbeWriteRows(std::uint8_t* dst, std::uint64_t rows,
                           std::uint64_t row_bytes, std::uint64_t stride) {
#if defined(__aarch64__)
  const auto value = vdupq_n_u8(0x3C);
  for (std::uint64_t row = 0; row < rows; ++row)
    for (std::uint64_t b = 0; b < row_bytes; b += 16) vst1q_u8(dst + row * stride + b, value);
#else
  for (std::uint64_t row = 0; row < rows; ++row) std::memset(dst + row * stride, 0x3C, row_bytes);
#endif
}
inline std::uint64_t ProbeCompute(std::uint64_t steps, std::uint64_t value) {
  for (std::uint64_t i = 0; i < steps; ++i) {
    value ^= value << 13; value ^= value >> 7; value ^= value << 17;
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+r"(value));
#endif
  }
  return value;
}
}  // namespace kvaf::aicpu_test

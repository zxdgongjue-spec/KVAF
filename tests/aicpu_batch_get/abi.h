#pragma once

#include <cstdint>
#include <limits>

namespace kvaf::aicpu_test {
constexpr std::uint64_t kMagic = 0x4b56414641424732ULL;
constexpr char kKernelName[] = "KvafAicpuBatchGetV2";
constexpr std::uint64_t kKernelRevision = 2;
constexpr std::uint64_t kMaxScatterThreads = 32;
constexpr std::uint64_t kRequestOffset = 0;
constexpr std::uint64_t kDoneOffset = 256;
constexpr std::uint64_t kStagingOffset = 4096;
constexpr std::uint64_t kGuardBytes = 64;
constexpr std::uint64_t kMaxAllocation = 8ULL << 30;
constexpr std::uint64_t kMaxRounds = 100000;

struct Config {
  std::uint64_t magic{kMagic};
  std::uint64_t rows{8192};
  std::uint64_t row_bytes{4096};
  std::uint64_t warmup{5};
  std::uint64_t iterations{100};
  std::uint64_t timeout_ms{30000};
  std::uint64_t scatter_threads{1};
};

struct Layout {
  std::uint64_t payload_bytes;
  std::uint64_t stride;
  std::uint64_t output_bytes;
  std::uint64_t region_bytes;
  std::uint64_t rounds;
};

// Keep every row on separate complete cache lines. The two control records
// have different producers and never share a line with data or each other.
inline bool MakeLayout(const Config& c, Layout* out) {
  if (!out || c.magic != kMagic || c.rows == 0 || c.rows > UINT32_MAX ||
      c.row_bytes < sizeof(std::uint64_t) || c.row_bytes > UINT32_MAX ||
      c.scatter_threads == 0 || c.scatter_threads > kMaxScatterThreads ||
      c.iterations == 0 || c.iterations > kMaxRounds ||
      c.warmup > kMaxRounds - c.iterations ||
      c.timeout_ms == 0 || c.timeout_ms > 600000) return false;
  const std::uint64_t stride = ((c.row_bytes + 63) & ~std::uint64_t{63}) + kGuardBytes;
  if (c.rows > (kMaxAllocation - kStagingOffset) / stride) return false;
  *out = {c.rows * c.row_bytes, stride, c.rows * stride,
          kStagingOffset + c.rows * c.row_bytes, c.warmup + c.iterations};
  return true;
}

struct alignas(64) Control {
  std::uint64_t sequence;
  std::uint64_t reserved[7];
};

enum class Result : std::uint64_t {
  kPending = 0, kOk = 1, kInvalidArgs = 2, kTimeout = 3,
  kSequenceError = 4, kDataMismatch = 5,
  kThreadError = 6,
};

struct Sample {
  std::uint64_t total_ticks;
  std::uint64_t wait_ticks;
  std::uint64_t scatter_ticks;
};

struct alignas(64) KernelResult {
  Result status;
  std::uint64_t completed_rounds;
  std::uint64_t failed_sequence;
  std::uint64_t counter_frequency;
  std::uint64_t requested_threads;
  std::uint64_t active_threads;
  std::uint64_t pthread_error;
  std::uint64_t kernel_revision;
};

struct KernelArgs {
  std::uint64_t region;
  std::uint64_t output;
  std::uint64_t samples;
  std::uint64_t result;
  Config config;
};

// Nontrivial row/byte pattern plus a changing per-round header detects stale
// data. CPU updates only the 8-byte header of each source row per round.
inline std::uint8_t Pattern(std::uint64_t row, std::uint64_t byte) {
  return static_cast<std::uint8_t>((row * 131 + byte * 17 + (byte >> 8)) % 251);
}
inline std::uint64_t RowTag(std::uint64_t sequence, std::uint64_t row) {
  return (sequence << 32) | row;
}
inline std::uint64_t OutputOffset(const Config& c, const Layout& l, std::uint64_t row) {
  return (c.rows - 1 - row) * l.stride;
}
// A packed source row can cross cache lines. Assign whole groups whose byte
// length is a multiple of 64, so worker invalidations never share a line.
inline std::uint64_t RowsPerCacheGroup(std::uint64_t row_bytes) {
  std::uint64_t a = row_bytes, b = 64;
  while (b) { const auto remainder = a % b; a = b; b = remainder; }
  return 64 / a;
}
inline std::uint32_t ActiveThreads(const Config& c) {
  const auto quantum = RowsPerCacheGroup(c.row_bytes);
  const auto groups = (c.rows + quantum - 1) / quantum;
  return static_cast<std::uint32_t>(groups < c.scatter_threads ? groups : c.scatter_threads);
}
struct RowRange { std::uint64_t begin, end; };
inline RowRange ScatterRange(const Config& c, std::uint32_t worker, std::uint32_t workers) {
  const auto quantum = RowsPerCacheGroup(c.row_bytes);
  const auto groups = (c.rows + quantum - 1) / quantum;
  const auto begin = (groups * worker / workers) * quantum;
  const auto end = (groups * (worker + 1) / workers) * quantum;
  return {begin, end < c.rows ? end : c.rows};
}
static_assert(sizeof(Config) == 56 && sizeof(Control) == 64);
static_assert(sizeof(Sample) == 24 && sizeof(KernelResult) == 64);
static_assert(sizeof(KernelArgs) == 88);
}  // namespace kvaf::aicpu_test

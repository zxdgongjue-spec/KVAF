#pragma once

#include <cstdint>
#include <limits>

namespace kvaf::aicpu_test {
constexpr std::uint64_t kMagic = 0x4b56414641424731ULL;
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
      c.row_bytes < 64 || c.row_bytes % 64 != 0 || c.row_bytes > UINT32_MAX ||
      c.iterations == 0 || c.iterations > kMaxRounds ||
      c.warmup > kMaxRounds - c.iterations ||
      c.timeout_ms == 0 || c.timeout_ms > 600000) return false;
  const std::uint64_t stride = c.row_bytes + kGuardBytes;
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
  std::uint64_t reserved[4];
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
static_assert(sizeof(Config) == 48 && sizeof(Control) == 64);
static_assert(sizeof(Sample) == 24 && sizeof(KernelResult) == 64);
static_assert(sizeof(KernelArgs) == 80);
}  // namespace kvaf::aicpu_test

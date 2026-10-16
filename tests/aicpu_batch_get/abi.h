#pragma once

#include <cstdint>

namespace kvaf::aicpu_test {
constexpr std::uint64_t kMagic = 0x4b56414641424735ULL;
constexpr char kKernelName[] = "KvafAicpuBatchGetV21";
constexpr std::uint64_t kKernelRevision = 21;
constexpr std::uint64_t kMaxScatterThreads = 32;
constexpr std::uint64_t kRequestOffset = 0;
constexpr std::uint64_t kDoneOffset = 256;
constexpr std::uint64_t kOutputReadyOffset = 320, kOutputVerifiedOffset = 384;
constexpr std::uint64_t kStagingOffset = 4096;
constexpr std::uint64_t kGuardBytes = 64;
constexpr std::uint64_t kMaxAllocation = 8ULL << 30;
constexpr std::uint64_t kMaxRounds = 100000;
constexpr std::uint64_t kChainReadyWorkerBytes = 128;

struct Config {
  std::uint64_t magic{kMagic};
  std::uint64_t rows{4096};
  std::uint64_t row_bytes{656};
  std::uint64_t warmup{50};
  std::uint64_t iterations{1000};
  std::uint64_t timeout_ms{30000};
  std::uint64_t scatter_threads{4};
  std::uint64_t pipeline_chunk_rows{1024}, pipeline_jettys{4};
  std::uint64_t payload_cache_maintenance{};
  std::uint64_t verify_payload{}, verify_each_round{};
};

struct Layout {
  std::uint64_t payload_bytes;
  std::uint64_t stride;
  std::uint64_t output_bytes;
  std::uint64_t region_bytes;
  std::uint64_t rounds;
};

// Packed staging, guarded output and per-worker/Jetty progress words are disjoint.
inline std::uint64_t Align64(std::uint64_t n) { return (n + 63) & ~std::uint64_t{63}; }
inline std::uint64_t RowsPerCacheGroup(std::uint64_t row_bytes);
inline std::uint64_t RegisteredOutputOffset(const Layout& l) {
  return Align64(kStagingOffset + l.payload_bytes);
}
inline bool MakeLayout(const Config& c, Layout* out) {
  if (!out || c.magic != kMagic || !c.rows || c.rows > UINT32_MAX ||
      c.row_bytes < 8 || c.row_bytes > UINT32_MAX || !c.scatter_threads ||
      c.scatter_threads > kMaxScatterThreads || !c.pipeline_jettys || c.pipeline_jettys > 16 ||
      !c.pipeline_chunk_rows || c.pipeline_chunk_rows > UINT32_MAX ||
      c.pipeline_chunk_rows % RowsPerCacheGroup(c.row_bytes) ||
      !c.iterations || c.iterations > kMaxRounds || c.warmup > kMaxRounds - c.iterations ||
      !c.timeout_ms || c.timeout_ms > 600000 || c.payload_cache_maintenance > 1 ||
      c.verify_payload > 1 || c.verify_each_round > 1 || (c.verify_payload && !c.verify_each_round)) return false;
  const auto stride = Align64(c.row_bytes) + kGuardBytes;
  const auto limit = kMaxAllocation - kStagingOffset - kGuardBytes - 63 -
      c.scatter_threads * kChainReadyWorkerBytes;
  if (c.rows > limit / (stride + c.row_bytes) || c.rows > limit / (2 * stride)) return false;
  *out = {c.rows * c.row_bytes, stride, c.rows * stride, 0, c.warmup + c.iterations};
  out->region_bytes = Align64(RegisteredOutputOffset(*out) + out->output_bytes + kGuardBytes) +
      c.scatter_threads * kChainReadyWorkerBytes;
  return true;
}
inline std::uint64_t ChainReadyOffset(const Config& c, const Layout& l,
                                      std::uint32_t worker, std::uint32_t queue = 0) {
  return l.region_bytes - c.scatter_threads * kChainReadyWorkerBytes +
      worker * kChainReadyWorkerBytes + queue * 8;
}
inline bool ShouldVerifyOutput(const Config& c, std::uint64_t sequence) {
  return c.verify_each_round || sequence == c.warmup + c.iterations;
}
inline std::uint64_t OutputVerificationToken(std::uint64_t sequence, bool matches) {
  return (sequence << 1) | (matches ? 0 : 1);
}

struct alignas(64) Control {
  std::uint64_t sequence;
  std::uint64_t reserved[7];
};

enum class Result : std::uint64_t {
  kPending = 0, kOk = 1, kInvalidArgs = 2, kTimeout = 3,
  kSequenceError = 4, kDataMismatch = 5,
  kCopyError = 6,
};

struct Sample {
  std::uint64_t total_ticks;
  std::uint64_t wait_ticks;
  std::uint64_t scatter_ticks;
  std::uint64_t prepare_ticks, verify_ticks;
};

struct alignas(64) KernelResult {
  Result status;
  std::uint64_t completed_rounds;
  std::uint64_t failed_sequence;
  std::uint64_t counter_frequency;
  std::uint64_t requested_threads;
  std::uint64_t active_threads;
  std::uint64_t kernel_revision;
  std::uint64_t reserved;
};

struct KernelArgs {
  std::uint64_t region;
  std::uint64_t samples;
  std::uint64_t result;
  std::uint64_t coordination;
  Config config;
};

struct alignas(64) CoordinationWord { std::uint64_t value{}, reserved[7]{}; };
struct alignas(64) MultiBlockCoordination {
  std::uint64_t magic{kMagic}, revision{kKernelRevision}, workers{}, reserved[5]{};
  CoordinationWord claimed, ready, initialized, request_sequence, request_start, ready_tick;
  CoordinationWord scatter_done, arrivals, round_complete, exited, status, failure_sequence;
};
inline bool IsValidMultiBlockCoordination(const MultiBlockCoordination& c, std::uint32_t workers) {
  return c.magic == kMagic && c.revision == kKernelRevision && workers &&
      workers <= kMaxScatterThreads && c.workers == workers;
}
inline MultiBlockCoordination MakeMultiBlockCoordination(std::uint32_t workers) {
  MultiBlockCoordination c{}; c.workers = workers; return c;
}

// Per-row byte pattern; changing round tags detect stale transfers.
inline std::uint8_t Pattern(std::uint64_t row, std::uint64_t byte) {
  return static_cast<std::uint8_t>((row * 131 + byte * 17 + (byte >> 8)) % 251);
}
inline std::uint8_t PayloadPattern(const Config& c, std::uint64_t sequence,
                                   std::uint64_t row, std::uint64_t byte) {
  return Pattern(row, byte) ^ (c.verify_payload ?
      static_cast<std::uint8_t>(sequence * 37 + (sequence >> 8) * 29) : 0);
}
inline std::uint64_t RowTag(std::uint64_t sequence, std::uint64_t row) {
  return (sequence << 32) | row;
}
inline std::uint64_t LogicalRowForOutputSlot(const Config& c, std::uint64_t slot) { return c.rows - 1 - slot; }
// Align worker boundaries to complete staging cache lines.
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
static_assert(sizeof(Config) == 96 && sizeof(Control) == 64);
static_assert(sizeof(Sample) == 40 && sizeof(KernelResult) == 64);
static_assert(sizeof(KernelArgs) == 128 && sizeof(CoordinationWord) == 64);
}  // namespace kvaf::aicpu_test

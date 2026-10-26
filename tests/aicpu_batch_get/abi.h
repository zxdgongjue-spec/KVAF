#pragma once

#include <cstdint>

namespace kvaf::aicpu_test {
constexpr std::uint64_t kMagic = 0x4b56414641424735ULL;
constexpr char kKernelName[] = "KvafAicpuBatchGetV25";
constexpr std::uint64_t kKernelRevision = 25;
constexpr std::uint32_t kDefaultAicpuBlocks = 4, kMaxAicpuBlocks = 32, kJettyCount = 4;
constexpr std::uint64_t kPipelineChunkRows = 1024;
constexpr std::uint64_t kRequestOffset = 0;
constexpr std::uint64_t kDoneOffset = 256;
constexpr std::uint64_t kOutputReadyOffset = 320, kOutputVerifiedOffset = 384;
constexpr std::uint64_t kStagingOffset = 4096;
constexpr std::uint64_t kGuardBytes = 64;
constexpr std::uint64_t kMaxAllocation = 8ULL << 30;
constexpr std::uint64_t kMaxRounds = 100000;
constexpr std::uint64_t kChainReadyWorkerBytes = 64;

enum class ScatterStart : std::uint64_t { kReady = 0, kDone = 1 };
inline const char* ScatterStartName(ScatterStart start) {
  return start == ScatterStart::kDone ? "done" : "ready";
}
struct Config {
  std::uint64_t magic{kMagic};
  std::uint64_t rows{4096};
  std::uint64_t row_bytes{656};
  std::uint64_t warmup{50};
  std::uint64_t iterations{1000};
  std::uint64_t timeout_ms{30000};
  std::uint64_t aicpu_blocks{kDefaultAicpuBlocks};
  ScatterStart scatter_start{ScatterStart::kReady};
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
inline std::uint64_t RegisteredOutputOffset(const Layout& l) {
  return Align64(kStagingOffset + l.payload_bytes);
}
inline bool MakeLayout(const Config& c, Layout* out) {
  if (!out || c.magic != kMagic || !c.rows || c.rows > UINT32_MAX ||
      c.row_bytes < 8 || c.row_bytes > UINT32_MAX || !c.aicpu_blocks || c.aicpu_blocks > kMaxAicpuBlocks ||
      (c.scatter_start != ScatterStart::kReady && c.scatter_start != ScatterStart::kDone) ||
      !c.iterations || c.iterations > kMaxRounds || c.warmup > kMaxRounds - c.iterations ||
      !c.timeout_ms || c.timeout_ms > 600000) return false;
  const auto stride = Align64(c.row_bytes) + kGuardBytes;
  const auto limit = kMaxAllocation - kStagingOffset - kGuardBytes - 63 -
      c.aicpu_blocks * kChainReadyWorkerBytes;
  if (c.rows > limit / (stride + c.row_bytes) || c.rows > limit / (2 * stride)) return false;
  *out = {c.rows * c.row_bytes, stride, c.rows * stride, 0, c.warmup + c.iterations};
  out->region_bytes = Align64(RegisteredOutputOffset(*out) + out->output_bytes + kGuardBytes) +
      c.aicpu_blocks * kChainReadyWorkerBytes;
  return true;
}
inline std::uint64_t ChainReadyOffset(const Layout& l,
                                      std::uint32_t worker, std::uint32_t queue = 0) {
  return Align64(RegisteredOutputOffset(l) + l.output_bytes + kGuardBytes) +
      worker * kChainReadyWorkerBytes + queue * 8;
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
inline MultiBlockCoordination MakeMultiBlockCoordination(std::uint32_t workers) {
  MultiBlockCoordination c{}; c.workers = workers; return c;
}

// Per-row byte pattern; changing round tags detect stale transfers.
inline std::uint8_t Pattern(std::uint64_t row, std::uint64_t byte) {
  return static_cast<std::uint8_t>((row * 131 + byte * 17 + (byte >> 8)) % 251);
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
  return static_cast<std::uint32_t>(groups < c.aicpu_blocks ? groups : c.aicpu_blocks);
}
struct RowRange { std::uint64_t begin, end; };
inline RowRange ScatterRange(const Config& c, std::uint32_t worker, std::uint32_t workers) {
  const auto quantum = RowsPerCacheGroup(c.row_bytes);
  const auto groups = (c.rows + quantum - 1) / quantum;
  const auto begin = (groups * worker / workers) * quantum;
  const auto end = (groups * (worker + 1) / workers) * quantum;
  return {begin, end < c.rows ? end : c.rows};
}
static_assert(sizeof(Config) == 64 && sizeof(Control) == 64);
static_assert(sizeof(Sample) == 40 && sizeof(KernelResult) == 64);
static_assert(sizeof(KernelArgs) == 96 && sizeof(CoordinationWord) == 64);
}  // namespace kvaf::aicpu_test

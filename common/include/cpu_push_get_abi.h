#pragma once

#include <cstdint>

namespace kvaf::cpu_push {
// V3 ports the a12bb16 V24 per-row pipeline and removes the V2 sort workspace.
constexpr std::uint64_t kMagic = 0x4b56414650555333ULL;
inline constexpr char kReferenceCommit[] = "a12bb16c120e885eb18ed61581739e1c212c32e8";
constexpr std::uint32_t kMaxRequests = 65536, kMaxWorkers = 8, kJettyCount = 4;
constexpr std::uint32_t kChunkRows = 1024, kTimeoutMs = 30000;
// Runtime additionally carries the opt-in profile bit with the request sequence.
constexpr std::uint64_t kProfileNotify = std::uint64_t{1} << 63;

enum class Result : std::uint64_t {
  kPending = 0, kOk = 1, kInvalidRequest = 2, kTimeout = 3,
  kSequenceError = 4, kTransportError = 5, kCopyError = 6,
};

// Different producers never share a control cache line.
struct alignas(64) Word { std::uint64_t value{}, reserved[7]{}; };
struct alignas(64) Request {
  std::uint64_t magic{kMagic}, sequence{};
  std::uint32_t count{}, token_bytes{}, workers{}, reserved{};
  std::uint64_t reserved_words[4]{};
};
struct alignas(64) Completion {
  // One aligned word publishes both sequence and result without a torn pair.
  std::uint64_t token{}, reserved[7]{};
};
// CPU owns all four words; each Jetty updates only its own 8-byte progress.
struct alignas(64) Ready { std::uint64_t jetty[kJettyCount]{}, reserved[4]{}; };
struct alignas(64) Coordination {
  Word claimed, built, arrivals, finished, result;
};
struct alignas(64) WorkerStatus {
  std::uint64_t magic{kMagic}, sequence{}, result{}, worker{}, consumed{}, reserved[3]{};
};
inline constexpr std::uint64_t Progress(std::uint64_t sequence, std::uint32_t chunks) {
  return (sequence << 32) | chunks;
}
static_assert(sizeof(Request) == 64 && sizeof(Completion) == 64 && sizeof(Ready) == 64);
static_assert(sizeof(Word) == 64 && sizeof(WorkerStatus) == 64);
}  // namespace kvaf::cpu_push

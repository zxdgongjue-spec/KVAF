#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "peer_protocol.h"

namespace kvaf::offload {
// Provider endpoint layout shared by CPU Pull and NPU Pull.
constexpr std::uint16_t kEndpointVersion = 3;
constexpr std::uint32_t kChannelCount = 4;
constexpr std::uint32_t kMaxSpans = 4096;
enum class Result : std::uint32_t { Ok, Invalid, NotFound, Busy, Failed, InDoubt };
struct Span { std::uint64_t source, destination, bytes; };
// Shared once per source connection. Route EID is independent of the opaque
// EIDs embedded in provider Jetty keys and MR segments.
struct ConnectionDesc {
  std::uint64_t tp{};
  std::uint32_t psn{}, token{}, key_bytes{};
  std::uint8_t eid[16]{}, key[kChannelCount][64]{};
};
// Per-MR logical range and provider export. Token ID remains in the opaque
// segment; token value is explicit even when registrations share its value.
struct MemoryDesc {
  std::uint64_t base{}, bytes{};
  std::uint32_t token{}, segment_bytes{};
  std::uint8_t segment[64]{};
};
static_assert(sizeof(Span) == 24 && sizeof(ConnectionDesc) == 296 && sizeof(MemoryDesc) == 88);
inline bool Contains(std::uint64_t size, std::uint64_t offset, std::uint64_t bytes) {
  return bytes && offset <= size && bytes <= size - offset;
}
inline bool ValidateSpans(const std::vector<Span>& spans, std::uint64_t source,
                          std::uint64_t destination) {
  if (spans.empty() || spans.size() > kMaxSpans) return false;
  std::vector<Span> sorted(spans);
  for (const auto& s : sorted)
    if (!Contains(source, s.source, s.bytes) || !Contains(destination, s.destination, s.bytes))
      return false;
  std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) { return a.destination < b.destination; });
  for (std::size_t i = 1; i < sorted.size(); ++i)
    if (sorted[i].destination - sorted[i-1].destination < sorted[i-1].bytes) return false;
  return true;
}
inline std::uint64_t Hash(const void* memory, std::size_t bytes, std::uint64_t hash = 14695981039346656037ULL) {
  const auto* p = static_cast<const std::uint8_t*>(memory);
  while (bytes--) { hash ^= *p++; hash *= 1099511628211ULL; }
  return hash;
}
} // namespace kvaf::offload

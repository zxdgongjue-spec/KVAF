#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include "peer_protocol.h"

namespace kvaf::offload {
// Provider endpoint layout shared by CPU Pull and NPU Pull.
constexpr std::uint16_t kEndpointVersion = 2;
constexpr std::uint32_t kChannelCount = 4;
constexpr std::uint32_t kMaxSpans = 4096;
enum class Result : std::uint32_t { Ok, Invalid, NotFound, Busy, Failed, InDoubt };
struct Span { std::uint64_t source, destination, bytes; };
struct Endpoint {
  std::uint64_t base{}, bytes{}, tp{};
  // The shared DDR ticket carries the authentication token; provider token
  // IDs remain local objects.
  std::uint32_t psn{}, token{}, key_bytes{}, segment_bytes{};
  // eid selects the CTP route. key and segment are opaque provider exports;
  // their embedded EIDs can differ from the route EID and must stay unchanged.
  std::uint8_t eid[16]{}, key[kChannelCount][64]{}, segment[64]{};
};
static_assert(sizeof(Span) == 24 && sizeof(Endpoint) == 376);
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

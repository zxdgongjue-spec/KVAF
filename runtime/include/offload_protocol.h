#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include "peer_protocol.h"

namespace kvaf::offload {
// Separate framing preserves the existing KVCP v6 messages. All peers must
// use the same little-endian deployment ABI; no native pointers cross the wire.
constexpr std::uint32_t kMagic = 0x4b564f46; // KVOF
constexpr std::uint16_t kVersion = 1;
constexpr std::uint32_t kMaxSpans = 4096;
enum class Command : std::uint16_t { Allocate = 1, Open, Submit, Query, ReleaseSource, Close, Checksum };
enum class Result : std::uint32_t { Ok, Invalid, NotFound, Busy, Failed, InDoubt };
enum class State : std::uint32_t { Empty, Open, Running, Done, Failed, InDoubt, Released };
struct Span { std::uint64_t source, destination, bytes; };
struct Endpoint {
  std::uint64_t base{}, bytes{}, tp{};
  std::uint32_t psn{}, token{}, key_bytes{}, segment_bytes{};
  std::uint8_t eid[16]{}, key[64]{}, segment[64]{};
};
struct Request {
  peer::MessageHeader header{};
  std::uint32_t reserved{};
  std::uint64_t region{}, epoch{}, task{}, bytes{}, slot_size{};
  std::uint32_t candidate{}, count{};
  Endpoint source{};
};
struct Response {
  peer::MessageHeader header{};
  Result result{Result::Invalid};
  State state{State::Empty};
  std::uint32_t reserved{};
  std::uint64_t region{}, epoch{}, task{}, base{}, bytes{}, checksum{};
  Endpoint endpoint{};
};
static_assert(sizeof(Span) == 24 && sizeof(Endpoint) == 184);
static_assert(sizeof(Request) == 248 && sizeof(Response) == 256);
static_assert(offsetof(Request, region) == 16 && offsetof(Response, region) == 24);
inline Request MakeRequest(Command command) {
  Request r{};
  r.header = {kMagic, kVersion, static_cast<std::uint16_t>(command), sizeof(Request)};
  return r;
}
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

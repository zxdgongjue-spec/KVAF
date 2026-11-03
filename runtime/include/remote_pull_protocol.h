#pragma once
#include "offload_protocol.h"

namespace kvaf::remote_pull {
// Shared DDR with identity-only Ready and application-defined Submit ranges.
constexpr std::uint32_t kMagic = 0x4b565250; // KVRP
constexpr std::uint16_t kVersion = 6;
constexpr std::uint32_t kMaxSourceMrs = 4096;
using MemoryDesc = offload::MemoryDesc;
using ConnectionDesc = offload::ConnectionDesc;
using Span = offload::Span;
using Result = offload::Result;
constexpr auto kMaxSpans = offload::kMaxSpans;
enum class Command : std::uint16_t { Describe = 1, Attach, Bind, Probe, Ready, Submit, Query, Complete, Cancel, Detach, Checksum, InspectSource, AddMemory };
enum class State : std::uint32_t { Connecting, Idle, Ready, Running, Done, Failed, InDoubt, Committed, Cancelled, Closed };
struct RegionInfo {
  std::uint64_t epoch{}, id{}, base{}, bytes{}, slot_size{};
  std::uint32_t candidate{}, token{}, closing{};
  std::uint8_t eid[16]{};
};
struct Request {
  peer::MessageHeader header{};
  std::uint32_t count{};
  std::uint64_t region{}, epoch{}, source_id{}, task{}; // task = B's ready sequence
  std::uint32_t token{}, flags{}; // Complete flags: 1=committed, 0=aborted
  std::uint64_t mr_id{1};
  std::uint32_t memory_count{1}, reserved{}; // Attach declares the immutable MR table size.
  MemoryDesc source{};
  ConnectionDesc connection{}; // Wire tail present only for Attach.
};
struct Response {
  peer::MessageHeader header{};
  Result result{Result::Invalid};
  std::uint64_t region{}, epoch{}, source_id{}, task{}, checksum{};
  State state{State::Connecting};
  std::uint32_t reserved{};
  std::uint64_t mr_id{1};
  RegionInfo pool{};
  MemoryDesc memory{};
  ConnectionDesc connection{}; // Wire tail present only for Attach.
};
inline std::uint32_t RequestBytes(std::uint16_t command) {
  return command == static_cast<std::uint16_t>(Command::Attach) ? sizeof(Request) : offsetof(Request, connection);
}
inline std::uint32_t ResponseBytes(std::uint16_t command) {
  return (command & 0x7fffU) == static_cast<std::uint16_t>(Command::Attach) ? sizeof(Response) : offsetof(Response, connection);
}
inline Request MakeRequest(Command command) {
  Request r{}; r.header = {kMagic, kVersion, static_cast<std::uint16_t>(command), RequestBytes(static_cast<std::uint16_t>(command))}; return r;
}
inline bool SameSpans(const std::vector<Span>& a, const std::vector<Span>& b) {
  return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size()*sizeof(Span)));
}
}

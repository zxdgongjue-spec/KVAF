#pragma once
#include "offload_protocol.h"

namespace kvaf::remote_pull {
// Shared-pool, third-party source control. Keep legacy KVOF and KVCP framing
// separate; old services cannot accidentally interpret these requests.
constexpr std::uint32_t kMagic = 0x4b565250; // KVRP
constexpr std::uint16_t kVersion = 1;
using Endpoint = offload::Endpoint;
using Span = offload::Span;
using Result = offload::Result;
constexpr auto kMaxSpans = offload::kMaxSpans;
enum class Command : std::uint16_t { Describe = 1, Attach, Bind, Probe, Ready, Submit, Query, Complete, Cancel, Detach, Checksum };
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
  Endpoint source{};
};
struct Response {
  peer::MessageHeader header{};
  Result result{Result::Invalid};
  std::uint64_t region{}, epoch{}, source_id{}, task{}, checksum{};
  State state{State::Connecting};
  std::uint32_t reserved{};
  RegionInfo pool{};
  Endpoint endpoint{};
};
inline Request MakeRequest(Command command) {
  Request r{}; r.header = {kMagic, kVersion, static_cast<std::uint16_t>(command), sizeof(r)}; return r;
}
inline bool SameSpans(const std::vector<Span>& a, const std::vector<Span>& b) {
  return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size()*sizeof(Span)));
}
}

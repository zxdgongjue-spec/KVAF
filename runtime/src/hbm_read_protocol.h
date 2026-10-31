#pragma once
#include "hbm_read.h"
#include "offload_protocol.h"

namespace kvaf::hbm_read {
constexpr std::uint32_t kMagic = 0x4b564852; // KVHR, independent of KVCP/KVRP
constexpr std::uint16_t kVersion = 4; // Only identity leases; ranges stay on the reader NPU.
enum class Command : std::uint16_t { Discover=1, Open, Channel, Confirm, Complete=6, Cancel, Close, AcquireRaw };
enum class State : std::uint32_t { Registered, Connected, Ready, Reading, Done };
using Span = offload::Span;
struct Request {
  peer::MessageHeader header{};
  std::uint32_t token{}, candidate{}, channel{}, count{};
  std::uint64_t source_id{}, epoch{}, client{}, sequence{};
  offload::Endpoint endpoint{};
};
struct Response {
  peer::MessageHeader header{};
  std::uint32_t code{}, candidates{}, state{}, reserved{};
  std::uint64_t source_id{}, epoch{}, client{}, sequence{}, checksum{};
  offload::Endpoint endpoint{};
  char detail[256]{};
};
static_assert(sizeof(Request) == 440 && sizeof(Response) == 704, "KVHR wire ABI changed");
inline Request Make(Command c, const HbmReadTicket& t, std::uint64_t client) {
  Request r{};
  r.header = {kMagic, kVersion, static_cast<std::uint16_t>(c), sizeof(Request)};
  r.token=t.token; r.source_id=t.source_id; r.epoch=t.epoch; r.client=client;
  return r;
}
Status Exchange(const std::string&, Request, const std::vector<Span>&, Response*);
int Connect(const std::string&);
Status Listen(const std::string&, int* fd, std::uint16_t* port);
}

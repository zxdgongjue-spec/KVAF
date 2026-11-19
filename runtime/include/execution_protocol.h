#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "execution_config.h"
#include "peer_link.h"

namespace kvaf::execution {
constexpr std::uint16_t kCommand = 20;
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kRegions = 1, kSessions = 2, kIdempotent = 4, kFabric = 8, kUrma = 16;
constexpr std::uint64_t kRequired = kRegions | kSessions | kIdempotent;
enum class Op : std::uint32_t {
  Capabilities,
  CreateRegion,
  OpenSession,
  Query,
  CloseSession,
  ReleaseRegion,
  SessionCall,
  Fence
};
enum class State : std::uint32_t { Creating, Ready, Failed, Recovering, Closed };
struct Request {
  Op op{Op::Capabilities};
  std::uint64_t client{0}, request{0}, region{0}, epoch{0}, session{0};
  TransportMode mode{TransportMode::Urma};
  Transport transport{Transport::Urma};
  Executor executor{Executor::Aiv};
  GetAlgorithm algorithm{GetAlgorithm::Direct};
  bool put{false};
  std::uint32_t candidate{0}, workers{1}, jettys{1};
  std::uint32_t request_capacity{65536}, token_bytes{0}, cpu_threads{0};
  std::uint64_t logical_bytes{0}, slot_size{0}, auxiliary_bytes{0};
  std::array<std::uint8_t, 16> local_eid{};
  std::vector<std::uint8_t> payload;
};
struct Response {
  Status status;
  State state{State::Failed};
  std::uint64_t capabilities{0}, region{0}, epoch{0}, session{0};
  std::uint64_t logical_bytes{0}, slot_size{0};
  peer::PeerRegion binding;
  std::vector<std::uint8_t> payload;
};
std::vector<std::uint8_t> Encode(const Request&);
std::vector<std::uint8_t> Encode(const Response&);
Status Decode(const std::vector<std::uint8_t>&, Request*);
Status Decode(const std::vector<std::uint8_t>&, Response*);
bool Exchange(const std::string& address, const Request&, Response*, std::string* error);
}  // namespace kvaf::execution

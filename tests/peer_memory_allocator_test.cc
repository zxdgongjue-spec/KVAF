#include "peer_memory_allocator.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/direct_hccp_remote_transport.h"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <exception>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

// Device bootstrap/teardown are outside this Host test. The real channel
// implementation below only uses the injected RA callbacks and PeerLink.
extern "C" int aclrtGetDevice(std::int32_t*) { return -1; }
namespace kvaf::direct_hccp {
void ReleaseDirectHccpSession(DirectHccpSession*) {}
}

namespace {
using namespace kvaf;
using namespace kvaf::peer;
namespace d = kvaf::direct_hccp;

void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

struct FakeRa {
  std::uint8_t expected_route;
  bool queried{false};
  bool imported{false};

  static int Query(void* context, d::GetTpCfg* cfg, d::TpInfo* info,
                   std::uint32_t* count, void** request) {
    auto& fake = *static_cast<FakeRa*>(context);
    Check(cfg->local_eid.raw[15] == 5U && cfg->peer_eid.raw[15] == fake.expected_route,
          "CTP query used the memory EID instead of the peer context EID");
    fake.queried = true;
    info->tp_handle = 11U;
    *count = 1U;
    *request = context;
    return 0;
  }
  static int Complete(void*, int* result) { *result = 0; return 0; }
  static int Random(d::RaInfo*, std::uint32_t* psn) { *psn = 7U; return 0; }
  static int Import(void* context, d::QpImportInfoT* info, void** target) {
    auto& fake = *static_cast<FakeRa*>(context);
    const auto& cfg = info->in.ub.exp_import_cfg;
    Check(fake.queried && cfg.tp_handle == 11U && cfg.peer_tp_handle == 12U &&
          cfg.tx_psn == 7U && cfg.rx_psn == 6U && info->in.key.size == 1U &&
          info->in.key.value[0] == 8U, "wrong CTP pair imported");
    fake.imported = true;
    info->out.ub.tpn = 42U;
    *target = context;
    return 0;
  }
};

template <typename T>
T Receive(int fd, Command command) {
  T request{};
  Check(recv(fd, &request, sizeof(request), MSG_WAITALL) == sizeof(request), "short request");
  Check(IsValidHeader(request.header, command, sizeof(request)), "wrong request command");
  return request;
}

template <typename T>
void Reply(int fd, Command command, T response) {
  response.header = MakeHeader(command, sizeof(response));
  response.header.command = ResponseCommand(command);
  Check(send(fd, &response, sizeof(response), 0) == sizeof(response), "short response");
}

// Real PeerLink TCP exchanges, with no CANN, NPU or liburma dependency.
class Server {
 public:
  explicit Server(std::vector<std::function<void(int)>> steps) {
    listener_ = socket(AF_INET, SOCK_STREAM, 0);
    Check(listener_ >= 0, "socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    Check(bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0 &&
          listen(listener_, 2) == 0, "listen failed");
    socklen_t length = sizeof(address);
    Check(getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &length) == 0,
          "getsockname failed");
    endpoint = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
    worker_ = std::thread([this, steps = std::move(steps)] {
      int fd = -1;
      try {
        for (const auto& step : steps) {
          pollfd ready{listener_, POLLIN, 0};
          Check(poll(&ready, 1, 3000) > 0, "missing control request");
          fd = accept(listener_, nullptr, nullptr);
          Check(fd >= 0, "accept failed");
          timeval timeout{3, 0};
          setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
          step(fd);
          close(fd);
          fd = -1;
        }
      } catch (...) {
        error_ = std::current_exception();
        if (fd >= 0) close(fd);
      }
    });
  }
  ~Server() {
    if (worker_.joinable()) worker_.join();
    close(listener_);
  }
  void Verify() {
    worker_.join();
    if (error_) std::rethrow_exception(error_);
  }
  std::string endpoint;

 private:
  int listener_{-1};
  std::thread worker_;
  std::exception_ptr error_;
};

void Lifecycle(DeviceKind kind) {
  const bool hbm = kind == DeviceKind::kNpu;
  std::vector<std::function<void(int)>> steps;
  if (hbm) {
    for (std::uint32_t i = 0; i < 2U; ++i) {
      steps.push_back([i](int fd) {
        const auto request = Receive<HbmEidCandidateRequest>(fd, Command::kGetHbmEidCandidate);
        Check(request.device_id == 3U && request.candidate_index == i, "wrong remote NPU candidate");
        HbmEidCandidateResponse response{};
        response.candidate_count = 2U;
        response.candidate_index = i;
        response.eid[15] = static_cast<std::uint8_t>(i + 1U);
        Reply(fd, Command::kGetHbmEidCandidate, response);
      });
    }
  } else {
    steps.push_back([](int fd) {
      Receive<CandidateCountRequest>(fd, Command::kGetCandidateCount);
      CandidateCountResponse response{};
      response.candidate_count = 2U;
      Reply(fd, Command::kGetCandidateCount, response);
    });
  }
  // First candidate fails; the next succeeds without leaking an allocation.
  for (std::uint32_t candidate = 0; candidate < 2U; ++candidate) {
    steps.push_back([hbm, candidate](int fd) {
      if (hbm) {
        const auto request = Receive<AllocateHbmRegionRequest>(fd, Command::kAllocateHbmRegion);
        Check(request.device_id == 3U && request.eid_index == candidate &&
              request.hbm_bytes == 8192U && request.slot_size == 4096U, "wrong HBM allocation");
        AllocateHbmRegionResponse response{};
        response.result = candidate == 0U ? static_cast<int>(Result::kResourceError) : 0;
        response.session_id = 77U;
        response.hbm_base = 0x100000000ULL;
        response.hbm_bytes = 8192U;
        response.token_id = 0x12300U;
        response.token_value = 0xABCFU;
        response.segment_key_bytes = response.jetty_key_bytes = 1U;
        response.segment_key[0] = 0xA5U;
        response.jetty_key[0] = 0xB5U;
        response.peer_eid[15] = 2U;
        response.route_eid[15] = 0xCBU;
        Reply(fd, Command::kAllocateHbmRegion, response);
      } else {
        const auto request = Receive<AllocateRegionRequest>(fd, Command::kAllocateRegion);
        Check(request.candidate_id == candidate && request.ddr_bytes == 8192U &&
              request.slot_size == 4096U, "wrong DDR allocation");
        AllocateRegionResponse response{};
        response.result = candidate == 0U ? static_cast<int>(Result::kResourceError) : 0;
        response.session_id = 77U;
        response.peer_base = 0x100000000ULL;
        response.peer_length = 8192U;
        response.token_id = 0x12300U;
        response.token_value = 0xABCFU;
        response.segment_key_bytes = 1U;
        response.segment_key[0] = 0xA5U;
        response.peer_eid[15] = 2U;
        Reply(fd, Command::kAllocateRegion, response);
      }
    });
  }
  steps.push_back([](int fd) {
    const auto request = Receive<CreateChannelRequest>(fd, Command::kCreateChannel);
    Check(request.session_id == 77U && request.local_qp_key_bytes == 1U &&
          request.local_qp_key[0] == 9U && request.local_eid[15] == 5U &&
          request.peer_token_value == 0xABCFU && request.local_tp_handle == 11U &&
          request.local_psn == 7U, "wrong channel descriptor");
    CreateChannelResponse response{};
    response.peer_qp_key_bytes = 1U;
    response.peer_qp_key[0] = 8U;
    response.peer_tp_handle = 12U;
    response.peer_psn = 6U;
    Reply(fd, Command::kCreateChannel, response);
  });
  for (bool fail : {true, false}) {
    steps.push_back([hbm, fail](int fd) {
      const auto command = hbm ? Command::kReleaseHbmRegion : Command::kReleaseRegion;
      const auto session = hbm ? Receive<ReleaseHbmRegionRequest>(fd, command).session_id
                               : Receive<ReleaseRegionRequest>(fd, command).session_id;
      Check(session == 77U, "release must use the owning session");
      StatusResponse response{};
      response.result = fail ? static_cast<int>(Result::kResourceError) : 0;
      Reply(fd, command, response);
    });
  }

  Server server(std::move(steps));
  auto link = std::make_shared<PeerLink>(server.endpoint);
  PeerMemoryAllocator allocator(link, kind, 3U);
  std::uint32_t count = 0U;
  Check(allocator.GetCandidateCount(&count).ok() && count == 2U, "discovery failed");
  const RuntimeInitRequest request{8192U, 4096U, kind};
  RemoteRegionDesc region;
  Check(!allocator.Allocate({8192U, 4096U, hbm ? DeviceKind::kCpu : DeviceKind::kNpu}, &region).ok(),
        "wrong memory kind accepted");
  Check(!allocator.Allocate({8191U, 4096U, kind}, &region).ok(), "invalid size accepted");
  Check(!allocator.Allocate(request, &region).ok() && !allocator.has_allocation() &&
        !link->has_region(), "failed candidate left a region");
  allocator.SetCandidate(1U);
  Check(allocator.Allocate(request, &region).ok(), "second candidate failed");
  Check(region.device == kind && region.base == 0x100000000ULL && region.length == 8192U &&
        region.slot_size == 4096U, "wrong remote region");
  Check(!allocator.Allocate(request, &region).ok(), "duplicate allocation accepted");
  PeerRegion descriptor;
  Check(link->LookupRegion(region.base, &descriptor).ok() && descriptor.device_memory == hbm &&
        descriptor.token_id == 0x12300U && descriptor.token_value == 0xABCFU &&
        descriptor.segment_key[0] == 0xA5U && descriptor.eid[15] == 2U,
        "exported descriptor lost");
  Check(descriptor.RoutingEid()[15] == (hbm ? 0xCBU : 2U) && descriptor.eid[15] == 2U,
        "CTP routing EID must not replace the exported memory EID");
  d::DirectMemory memory{};
  std::array<std::uint8_t, kEidBytes> local_eid{}, memory_eid{};
  Check(d::LoadPeerMemory(link.get(), region.base, &memory, &memory_eid) &&
        memory_eid[15] == 2U && memory.token_id == 0x123U && memory.key[0] == 0xA5U,
        "WQE memory descriptor changed while selecting route EID");
  local_eid[15] = 5U;
  d::QpCreateInfo local_jetty{};
  local_jetty.key.size = 1U;
  local_jetty.key.value[0] = 9U;
  FakeRa fake{static_cast<std::uint8_t>(hbm ? 0xCBU : 2U)};
  d::DirectHccpApi api{};
  api.get_tp_info_async = FakeRa::Query;
  api.get_async_req_result = FakeRa::Complete;
  api.get_random = FakeRa::Random;
  api.qp_import = FakeRa::Import;
  d::PeerChannel channel{};
  Check(d::CreatePeerChannel({link.get(), &api, &fake, 7U, region.base, &memory,
                             &local_eid, &memory_eid, &local_jetty, "host_test"}, &channel) &&
        fake.imported && channel.local_import_info.out.ub.tpn == 42U, "channel failed");
  Check(!allocator.Release({region.base + 1U, region.length, region.slot_size, kind}).ok(),
        "wrong allocation released");
  Check(!allocator.Release(region).ok() && allocator.has_allocation() && link->has_region(),
        "failed release lost ownership");
  Check(allocator.Release(region).ok() && !allocator.has_allocation() && !link->has_region(),
        "release retry failed");
  server.Verify();
}
}  // namespace

int main() {
  try {
    Lifecycle(DeviceKind::kNpu);
    Lifecycle(DeviceKind::kCpu);
    std::cout << "Peer memory allocator lifecycle: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

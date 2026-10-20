#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "peer_link.h"

namespace {
using namespace kvaf::peer;
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void ExchangeChannel(bool shared, std::uint32_t acknowledgement, bool succeeds) {
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  Check(listener >= 0, "socket failed");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  Check(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0 &&
            listen(listener, 1) == 0,
        "listen failed");
  socklen_t size = sizeof(address);
  Check(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) == 0,
        "getsockname failed");
  bool request_ok = false;
  std::thread server([&] {
    pollfd first_ready{listener, POLLIN, 0};
    if (poll(&first_ready, 1, 2000) <= 0) return;
    const int allocation_fd = accept(listener, nullptr, nullptr);
    if (allocation_fd < 0) return;
    AllocateRegionRequest allocate{};
    if (recv(allocation_fd, &allocate, sizeof(allocate), MSG_WAITALL) != sizeof(allocate) ||
        !IsValidHeader(allocate.header, Command::kAllocateRegion, sizeof(allocate))) {
      close(allocation_fd);
      return;
    }
    AllocateRegionResponse allocated{};
    allocated.header = MakeHeader(Command::kAllocateRegion, sizeof(allocated));
    allocated.header.command = ResponseCommand(Command::kAllocateRegion);
    allocated.session_id = 9;
    allocated.peer_base = 4096;
    allocated.peer_length = 4096;
    allocated.token_id = 7;
    allocated.token_value = 123;
    allocated.segment_key_bytes = 48;
    if (send(allocation_fd, &allocated, sizeof(allocated), 0) != sizeof(allocated)) {
      close(allocation_fd);
      return;
    }
    close(allocation_fd);
    pollfd ready{listener, POLLIN, 0};
    if (poll(&ready, 1, 2000) <= 0) return;
    const int fd = accept(listener, nullptr, nullptr);
    if (fd < 0) return;
    timeval timeout{2, 0};
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    CreateChannelRequest request{};
    request_ok = recv(fd, &request, sizeof(request), MSG_WAITALL) == sizeof(request) &&
                 IsValidHeader(request.header, Command::kCreateChannel, sizeof(request)) &&
                 request.session_id == 9 && request.local_tp_handle == 42 &&
                 request.local_psn == 1 && request.local_qp_key_bytes == 28 &&
                 request.peer_token_value == 123 && request.local_eid[15] == 7 &&
                 request.reserved == (shared ? kChannelSharedTp : 0U);
    CreateChannelResponse response{};
    response.header = MakeHeader(Command::kCreateChannel, sizeof(response));
    response.header.command = ResponseCommand(Command::kCreateChannel);
    response.peer_qp_key_bytes = 28;
    response.peer_qp_key[0] = 17;
    response.peer_tp_handle = 43;
    response.peer_psn = 5;
    response.reserved = acknowledgement;
    request_ok = request_ok && send(fd, &response, sizeof(response), 0) == sizeof(response);
    close(fd);
  });
  PeerLink link("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
  PeerRegion region{};
  region.base = 4096;
  region.length = 4096;
  region.session_id = 9;
  Check(link.AllocateRegion(4096, 64, 0, &region).ok(), "allocate region failed");
  std::array<std::uint8_t, kQpKeyBytes> local{}, peer{};
  std::array<std::uint8_t, kEidBytes> eid{};
  eid[15] = 7;
  std::uint32_t key_bytes = 0, psn = 0;
  std::uint64_t tp = 0;
  const auto status = shared ? link.CreateChannel(region.base, local.data(), 28, eid.data(), 123,
                                                  42, 1, &peer, &key_bytes, &tp, &psn, true)
                             : link.CreateChannel(region.base, local.data(), 28, eid.data(), 123,
                                                  42, 1, &peer, &key_bytes, &tp, &psn);
  server.join();
  close(listener);
  Check(request_ok, "channel negotiation changed legacy fields or lost shared-TP request");
  Check(status.ok() == succeeds, "shared TP acknowledgement was not enforced");
  if (succeeds)
    Check(tp == 43 && psn == 5 && key_bytes == 28 && peer[0] == 17,
          "channel response was decoded incorrectly");
  else
    Check(tp == 0 && psn == 0 && key_bytes == 0 &&
              status.message().find("rebuild and restart") != std::string::npos,
          "unsupported shared TP exposed partial connection data or an unclear error");
}
}  // namespace

int main() {
  try {
    ExchangeChannel(false, 0U, true);  // Existing callers keep the original contract.
    ExchangeChannel(true, kChannelSharedTp, true);
    ExchangeChannel(true, 0U, false);  // An old service must not silently allocate separate TPs.
    std::cout << "Peer channel shared TP negotiation: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

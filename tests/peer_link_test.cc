#include "peer_link.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << __LINE__ << ": " #c "\n"; std::exit(1); } } while (false)
using namespace kvaf::peer;

template<class T> T Receive(int fd, MessageHeader header) {
  T request{};
  request.header = header;
  CHECK(IsValidHeader(header, static_cast<Command>(header.command), sizeof(T)));
  CHECK(recv(fd, reinterpret_cast<char*>(&request) + sizeof(header),
             sizeof(T) - sizeof(header), MSG_WAITALL) == sizeof(T) - sizeof(header));
  return request;
}
template<class T> void Reply(int fd, Command command, T response) {
  response.header = MakeHeader(command, sizeof(T));
  response.header.command = ResponseCommand(command);
  CHECK(send(fd, &response, sizeof(T), MSG_NOSIGNAL) == sizeof(T));
}

int main() {
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(listener >= 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t size = sizeof(address);
  CHECK(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) == 0);
  CHECK(listen(listener, 4) == 0);
  std::vector<uint64_t> released;
  std::thread service([&] {
    uint64_t next_session = 1;
    for (int exchange = 0; exchange < 11; ++exchange) {
      const int fd = accept(listener, nullptr, nullptr);
      CHECK(fd >= 0);
      MessageHeader header{};
      CHECK(recv(fd, &header, sizeof(header), MSG_WAITALL) == sizeof(header));
      const auto command = static_cast<Command>(header.command);
      switch (command) {
        case Command::kGetCandidateInfo: {
          const auto request = Receive<CandidateInfoRequest>(fd, header);
          CHECK(request.candidate_id == 1);
          CandidateInfoResponse response{};
          response.candidate_id = 1;
          response.device_id = 1;
          response.eid_index = 3;
          std::strcpy(response.device_name, "udmac0d1e3");
          response.eid[15] = 2;
          Reply(fd, command, response);
          break;
        }
        case Command::kAllocateRegion: {
          const auto request = Receive<AllocateRegionRequest>(fd, header);
          CHECK(request.ddr_bytes == 4096 && request.slot_size == 64);
          AllocateRegionResponse response{};
          if (request.candidate_id == 99) {
            CHECK(request.shared_session_id == 3);
            response.result = static_cast<int32_t>(Result::kResourceError);
          } else {
            CHECK(request.shared_session_id == (request.candidate_id == 1 ? 1U : 0U));
            response.session_id = next_session++;
            response.peer_base = 0x100000;
            response.peer_length = 4096;
            response.token_id = 0x100 + request.candidate_id;
            response.token_value = 0x200 + request.candidate_id;
            response.segment_key_bytes = 16;
            response.peer_eid[15] = request.candidate_id + 1;
          }
          Reply(fd, command, response);
          break;
        }
        case Command::kCreateChannel: {
          const auto request = Receive<CreateChannelRequest>(fd, header);
          CHECK(request.session_id == (exchange == 3 ? 1U : 2U));
          CHECK(request.peer_token_value == 0x1ff + request.session_id);
          CreateChannelResponse response{};
          response.peer_qp_key_bytes = 16;
          response.peer_qp_key[0] = static_cast<uint8_t>(request.session_id);
          response.peer_tp_handle = 1;
          response.peer_psn = 2;
          Reply(fd, command, response);
          break;
        }
        case Command::kReleaseRegion: {
          const auto request = Receive<ReleaseRegionRequest>(fd, header);
          released.push_back(request.session_id);
          StatusResponse response{};
          if (released.size() == 1) response.result = static_cast<int32_t>(Result::kResourceError);
          Reply(fd, command, response);
          break;
        }
        default: CHECK(false);
      }
      close(fd);
    }
  });
  PeerLink link("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
  PeerCandidateInfo info;
  CHECK(link.GetCandidateInfo(1, &info).ok());
  CHECK(info.eid[15] == 2 && info.device_name == "udmac0d1e3");
  PeerRegion primary, alias;
  CHECK(link.AllocateRegion(4096, 64, 0, &primary).ok());
  CHECK(link.RegisterRegionPath(primary.base, 64, 1).ok());
  CHECK(link.RegionPathCount(primary.base) == 2);
  CHECK(link.LookupRegion(primary.base, &alias, 1).ok());
  CHECK(alias.base == primary.base && alias.eid[15] == 2 && alias.token_id != primary.token_id);
  CHECK(!link.LookupRegion(primary.base, &alias, 2).ok());
  for (uint32_t path = 0; path < 2; ++path) {
    std::array<uint8_t, kQpKeyBytes> key{}, peer_key{};
    std::array<uint8_t, kEidBytes> eid{};
    eid[15] = 3;
    uint32_t key_bytes = 0, psn = 0;
    uint64_t handle = 0;
    CHECK(link.CreateChannel(primary.base, key.data(), 16, eid.data(), 0x200 + path,
                             1, 1, &peer_key, &key_bytes, &handle, &psn, path).ok());
    CHECK(peer_key[0] == path + 1);
  }
  CHECK(!link.ReleaseRegion(primary.base).ok());
  CHECK(link.has_region());
  CHECK(link.ReleaseRegion(primary.base).ok());
  CHECK(!link.has_region());
  CHECK(link.AllocateRegion(4096, 64, 0, &primary).ok());
  CHECK(!link.RegisterRegionPath(primary.base, 64, 99).ok());
  CHECK(link.RegionPathCount(primary.base) == 1);
  CHECK(link.ReleaseRegion(primary.base).ok());
  service.join();
  close(listener);
  CHECK(released == std::vector<uint64_t>({2, 2, 1, 3}));
  std::cout << "PeerLink: exact EIDs, shared registration, per-peer channels, release retry passed\n";
}

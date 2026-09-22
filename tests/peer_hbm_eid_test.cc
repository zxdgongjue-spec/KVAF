#include "peer_link.h"
#include "npu_eid_discovery.h"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace kvaf::peer;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

HbmEidCandidateResponse Candidate(std::uint32_t index, std::uint8_t value) {
  HbmEidCandidateResponse response{};
  response.header = MakeHeader(Command::kGetHbmEidCandidate, sizeof(response));
  response.header.command = ResponseCommand(Command::kGetHbmEidCandidate);
  response.candidate_count = 2U;
  response.candidate_index = index;
  response.eid[15] = value;
  return response;
}

void Query(const std::vector<HbmEidCandidateResponse>& replies, bool succeeds) {
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  Check(listener >= 0, "socket failed");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  Check(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0 &&
            listen(listener, 2) == 0, "listen failed");
  socklen_t length = sizeof(address);
  Check(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0,
        "getsockname failed");
  bool requests_ok = true;
  std::thread server([&] {
    for (std::size_t i = 0; i < replies.size(); ++i) {
      pollfd ready{listener, POLLIN, 0};
      if (poll(&ready, 1, 2000) <= 0) { requests_ok = false; break; }
      const int fd = accept(listener, nullptr, nullptr);
      if (fd < 0) { requests_ok = false; break; }
      timeval timeout{2, 0};
      (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      HbmEidCandidateRequest request{};
      requests_ok = requests_ok &&
          recv(fd, &request, sizeof(request), MSG_WAITALL) == sizeof(request) &&
          IsValidHeader(request.header, Command::kGetHbmEidCandidate, sizeof(request)) &&
          request.device_id == 3U && request.candidate_index == i;
      if (send(fd, &replies[i], sizeof(replies[i]), 0) != sizeof(replies[i])) requests_ok = false;
      close(fd);
    }
  });
  PeerLink link("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
  std::vector<std::array<std::uint8_t, kEidBytes>> candidates(1);
  const auto status = link.GetHbmEidCandidates(3U, &candidates);
  server.join();
  close(listener);
  Check(requests_ok, "candidate query sent wrong device or index");
  Check(status.ok() == succeeds, "candidate response validation failed");
  if (succeeds) {
    Check(candidates.size() == 2U && candidates[0][15] == 1U && candidates[1][15] == 2U,
          "candidate order or EID bytes changed");
  } else {
    Check(candidates.empty(), "failed discovery exposed a partial candidate list");
  }
}

void DiscoveryProtocol() {
  Query({Candidate(0, 1), Candidate(1, 2)}, true);
  Query({Candidate(0, 1), Candidate(1, 1)}, false);  // duplicate EID
  Query({Candidate(0, 1), Candidate(1, 0)}, false);  // zero EID
  Query({Candidate(1, 1)}, false);                 // wrong index
  auto changed = Candidate(1, 2);
  changed.candidate_count = 3U;
  Query({Candidate(0, 1), changed}, false);
  auto old = Candidate(0, 1);
  old.header.version = kProtocolVersion - 1U;
  Query({old}, false);
  auto unavailable = Candidate(0, 1);
  unavailable.result = static_cast<std::int32_t>(Result::kNotFound);
  Query({unavailable}, false);
}

void ClosCandidateOrder() {
  char path[] = "/tmp/kvaf-eid-test-XXXXXX";
  const int fd = mkstemp(path);
  Check(fd >= 0, "temporary file failed");
  close(fd);
  {
    std::ofstream output(path);
    output << "\"device_id\": 7\n\"net_layer\": 1\n\"net_type\": \"CLOS\"\n";
    for (const char* eid : {"00000000000000000000000000000002",
                            "00000000000000000000000000000001",
                            "00000000000000000000000000000002"}) {
      output << "\"addr_type\": \"EID\"\n\"addr\": \"" << eid << "\"\n";
    }
    output << "\"device_id\": 8\n\"net_layer\": 1\n\"net_type\": \"CLOS\"\n"
              "\"addr_type\": \"EID\"\n\"addr\": \"00000000000000000000000000000003\"\n";
  }
  std::vector<std::array<std::uint8_t, kEidBytes>> candidates;
  std::string error;
  const bool found = kvaf::npu_eid_detail::ReadHcclClosEids(path, 7U, &candidates, &error);
  unlink(path);
  Check(found && candidates.size() == 2U && candidates[0][15] == 2U && candidates[1][15] == 1U,
        "CLOS discovery must filter physical device, deduplicate and preserve order");
}
}  // namespace

int main() {
  try {
    DiscoveryProtocol();
    ClosCandidateOrder();
    std::cout << "HBM EID discovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

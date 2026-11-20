#include "src/npu_eid_selection.h"
#include "peer_link.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace {
using namespace kvaf;
using detail::NpuEid;
void Check(bool ok, const char* description) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", description);
    std::exit(1);
  }
}
NpuEid Eid(unsigned id) {
  NpuEid eid{};
  eid.back() = id;
  return eid;
}

void NpuSelection() {
  const std::string channel = "Dev id info. (dev_id=2;slot_id=1)\nchan_id=0;flag=1;hop=1\n";
  const std::string cpu = "local_eid: 0000:0000:0000:0000:0000:0000:0000:0001\n";
  const std::string npu = "remote_eid: " + detail::EidToHex(Eid(2)) + "\n";
  const auto parse = [&](const std::string& content) {
    std::istringstream input(content);
    return detail::ParsePairInfo(input, 2);
  };
  auto pair = parse(channel + cpu + npu + "Dev id info. (dev_id=3;slot_id=1)\n" + cpu);
  Check(pair.cpu == Eid(1) && pair.npu == Eid(2), "both sides of the selected card/channel");
  Check(!parse(channel + cpu + npu + channel + cpu).cpu, "ambiguous active channels");
  auto partial = parse(channel + cpu + "remote_eid: invalid\n");
  Check(partial.cpu == Eid(1) && !partial.npu, "independent CPU and NPU fallback");
  Check(!parse(channel + cpu + cpu + npu).cpu, "duplicate CPU fields");

  char path[] = "/tmp/kvaf-clos-test-XXXXXX";
  int fd = mkstemp(path);
  Check(fd >= 0, "create rootinfo fixture");
  close(fd);
  {
    std::ofstream root(path);
    root << "\"device_id\": 2\n\"net_layer\": 1\n\"net_type\": \"CLOS\"\n";
    for (unsigned id : {0, 3, 4})
      root << "\"addr_type\": \"EID\"\n\"addr\": \"" << detail::EidToHex(Eid(id)) << "\"\n";
  }
  setenv("KVAF_HCCL_ROOTINFO_PATH", path, 1);
  std::vector<NpuEid> selected;
  Check(detail::SelectNpuEids(2, {}, pair, &selected).ok() && selected == std::vector<NpuEid>{Eid(2)},
        "local pair precedes CLOS");
  Check(detail::SelectNpuEids(2, {}, partial, &selected).ok() && selected == std::vector<NpuEid>{Eid(3)},
        "first nonzero CLOS when NPU pair field is missing");
  Check(detail::SelectNpuEids(2, {}, {}, &selected).ok() && selected.front() == Eid(3),
        "remote automatic CLOS");
  unlink(path);
  Check(!detail::SelectNpuEids(2, {}, {}, &selected).ok(), "missing remote CLOS fails");
  Check(detail::SelectNpuEids(2, {detail::EidToHex(Eid(5)), detail::EidToHex(Eid(6))},
                              pair, &selected).ok() && selected == std::vector<NpuEid>{Eid(5), Eid(6)},
        "explicit NPU EIDs override discovery and preserve order");
}

void CpuSelection() {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t size = sizeof(addr);
  Check(listener >= 0 && bind(listener, reinterpret_cast<sockaddr*>(&addr), size) == 0 &&
        listen(listener, 4) == 0 && getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &size) == 0,
        "start loopback CPU service");
  std::thread server([&] {
    for (;;) {
      int fd = accept(listener, nullptr, nullptr);
      if (fd < 0) break;
      peer::MessageHeader h{};
      Check(recv(fd, &h, sizeof(h), MSG_WAITALL) == sizeof(h), "read request header");
      const auto command = static_cast<peer::Command>(h.command);
      if (command == peer::Command::kGetCandidateCount) {
        peer::CandidateCountResponse reply{peer::MakeHeader(command, sizeof(reply)), 0, 3, 2, 0};
        reply.header.command = peer::ResponseCommand(command);
        Check(send(fd, &reply, sizeof(reply), 0) == sizeof(reply), "reply candidate count");
      } else {
        peer::CandidateEidRequest request{};
        Check(h.bytes == sizeof(request) && command == peer::Command::kGetCandidateInfo &&
              recv(fd, reinterpret_cast<char*>(&request) + sizeof(h), sizeof(request) - sizeof(h),
                   MSG_WAITALL) == sizeof(request) - sizeof(h), "read EID query");
        const auto id = request.info.candidate_id;
        peer::CandidateEidResponse record{};
        auto& reply = record.info;
        Check(request.response_bytes == sizeof(record) && id < 3, "negotiated EID metadata");
        reply.header = peer::MakeHeader(command, sizeof(record));
        reply.header.command = peer::ResponseCommand(command);
        reply.candidate_id = id;
        reply.device_id = id == 2;
        reply.eid_index = id == 1 ? 5 : 2;
        std::strcpy(reply.device_name, id == 2 ? "devB" : "devA");
        record.eid[15] = id + 1;
        Check(send(fd, &record, sizeof(record), 0) == sizeof(record), "reply candidate EID");
      }
      close(fd);
    }
  });
  peer::PeerLink link("localhost:" + std::to_string(ntohs(addr.sin_port)));
  peer::PeerCandidateTable table;
  Check(link.DiscoverCandidates(&table).ok() && table.local, "actual connected peer is local");
  peer::PeerCandidateInfo selected;
  const auto choose = [&](const std::string& device, NpuEid eid, unsigned expected) {
    Check(link.SelectCpuCandidate(table, device, eid, &selected).ok() && selected.candidate_id == expected,
          "CPU endpoint selection");
  };
  choose("", Eid(2), 1);
  choose("devA", Eid(2), 0);
  choose("devB", Eid(2), 2);
  choose("", Eid(99), 0);
  Check(!link.SelectCpuCandidate(table, "missing", Eid(2), &selected).ok(), "missing fixed device");
  table.device_fixed = true;
  choose("", Eid(2), 0);
  table.device_fixed = false;
  table.local = false;
  choose("", Eid(2), 0);
  shutdown(listener, SHUT_RDWR);
  server.join();
  close(listener);
}
}  // namespace

int main() {
  NpuSelection();
  CpuSelection();
}

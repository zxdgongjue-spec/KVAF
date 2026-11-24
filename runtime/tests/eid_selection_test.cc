#include "src/npu_eid_selection.h"
#include "peer_link.h"
#include "cache_region_service.h"
#include "peer_execution_resources.h"

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
  const std::string device = "Dev id info. (dev_id=2;slot_id=1)\n";
  const std::string channel = device + "chan_id=0;flag=1;hop=1\n";
  const std::string cpu = "local_eid: 0000:0000:0000:0000:0000:0000:0000:0001\n";
  const std::string npu = "remote_eid: " + detail::EidToHex(Eid(2)) + "\n";
  const auto parse = [&](const std::string& content) {
    std::istringstream input(content);
    return detail::ParsePairInfo(input, 2);
  };
  auto pair = parse(channel + cpu + npu + "Dev id info. (dev_id=3;slot_id=1)\n" + cpu);
  Check(pair.cpu == Eid(1) && pair.npu == Eid(2), "both sides of the selected card/channel");
  for (const char* fields : {"chan_id=0, flag=1", "chan_id=0\tflag=1 hop=1",
                             "flag = 1; chan_id = 0"}) {
    const auto parsed = parse(device + fields + "\n" + cpu + npu);
    Check(parsed.cpu == Eid(1) && parsed.npu == Eid(2), fields);
  }
  for (const char* fields : {"chan_id=0, flag=10", "chan_id=0, flag=1x",
                             "chan_id=0, other_flag=1", "chan_id=0, flag=0"}) {
    const auto parsed = parse(device + fields + "\n" + cpu + npu);
    Check(!parsed.cpu && !parsed.npu, fields);
  }
  const auto mixed = parse("Dev id info. (dev_id=20, slot_id=1)\nchan_id=0, flag=1\n" + cpu +
                           "Dev id info. (slot_id=1, dev_id = 2)\nchan_id=0; flag=1\n" + cpu + npu);
  Check(mixed.cpu == Eid(1) && mixed.npu == Eid(2), "exact device ID with comma-separated fields");
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

struct TestAllocation : execution::Allocation {
  Status Close() override { return Status::OK(); }
};
struct TestRegistration : execution::Session {
  Status Open(const execution::Request& q, execution::Allocation& allocation, peer::PeerRegion* out) override {
    *out = allocation.descriptor;
    out->eid = Eid(q.candidate + 1);
    out->token_id = (q.candidate + 1) << 8;
    return Status::OK();
  }
  Status Call(const std::vector<uint8_t>&, std::vector<uint8_t>*) override { return Status::NotImplemented("unused"); }
  Status Close() override { return Status::OK(); }
};
struct TestOperator : AccessOperator {
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::NotImplemented("unused"); }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::NotImplemented("unused"); }
};

void CheckSessions(const std::string& address, const std::vector<peer::PeerCandidateInfo>& candidates) {
  auto root = std::make_shared<peer::PeerLink>(address);
  peer::PeerRegion region;
  Check(root->AllocateCacheRegion(4096, 64, 0, TransportMode::Urma, &region).ok(), "one logical DDR region");
  Check(root->LookupRegion(region.base, &region).ok() && region.region_id, "logical DDR identity");
  int published = 0;
  const auto factory = [&](const PeerExecutionResources::Links& links) {
    ++published;
    Check(links.size() == candidates.size(), "all sessions prepared before publishing");
    for (std::size_t i = 0; i < links.size(); ++i) {
      peer::PeerRegion binding;
      Check(links[i]->LookupRegion(region.base, &binding).ok() && binding.region_id == region.region_id &&
            binding.length == region.length && binding.eid == candidates[i].eid &&
            binding.token_id == (candidates[i].candidate_id + 1) * 256, "per-FE DDR binding");
    }
    return std::make_unique<TestOperator>();
  };
  // A stale candidate EID on the second session must leave both opened children closable.
  auto stale = candidates;
  stale[1].eid = Eid(99);
  PeerExecutionResources failed(root, {}, stale, factory, 128);
  Check(!failed.Initialize({region.base, region.length, 64}).ok() && !published, "reject mismatched second binding");
  Check(failed.Close().ok(), "retire partial Get");
  PeerExecutionResources ready(root, {}, candidates, factory, 128);
  Check(ready.Initialize({region.base, region.length, 64}).ok() && published == 1, "recover all Get sessions");
  Check(root->ReleaseRegion(region.base).code() == StatusCode::kBusy, "live sessions hold shared DDR");
  Check(ready.Close().ok() && ready.Close().ok(), "idempotent close of all FE sessions");
  Check(root->ReleaseRegion(region.base).ok(), "no child sessions left after recovery");
}

void CpuSelection() {
  execution::CacheRegionService service(execution::kUrma,
      [](const execution::Request& q, std::unique_ptr<execution::Allocation>* out) {
        *out = std::make_unique<TestAllocation>();
        (*out)->descriptor.base = 0x10000;
        (*out)->descriptor.length = q.logical_bytes;
        return Status::OK();
      }, [](const execution::Request&) { return std::make_unique<TestRegistration>(); });
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
      if (command == peer::Command::kGetCacheCandidateCount) {
        peer::CandidateCountResponse reply{peer::MakeHeader(command, sizeof(reply)), 0, 5, 4, 0};
        reply.header.command = peer::ResponseCommand(command);
        Check(send(fd, &reply, sizeof(reply), 0) == sizeof(reply), "reply candidate count");
      } else if (h.command == execution::kCommand) {
        std::vector<uint8_t> body(h.bytes - sizeof(h));
        execution::Request q;
        Check(recv(fd, body.data(), body.size(), MSG_WAITALL) == static_cast<ssize_t>(body.size()) &&
              execution::Decode(body, &q).ok(), "read execution request");
        body = execution::Encode(service.Handle(q));
        h.command = peer::ResponseCommand(command);
        h.bytes = sizeof(h) + body.size();
        Check(send(fd, &h, sizeof(h), 0) == sizeof(h) &&
              send(fd, body.data(), body.size(), 0) == static_cast<ssize_t>(body.size()), "reply execution request");
      } else {
        peer::CandidateEidRequest request{};
        Check(h.bytes == sizeof(request) && command == peer::Command::kGetCandidateInfo &&
              recv(fd, reinterpret_cast<char*>(&request) + sizeof(h), sizeof(request) - sizeof(h),
                   MSG_WAITALL) == sizeof(request) - sizeof(h), "read EID query");
        const auto id = request.info.candidate_id;
        peer::CandidateEidResponse record{};
        auto& reply = record.info;
        Check(request.response_bytes == sizeof(record) && id < 5, "negotiated EID metadata");
        reply.header = peer::MakeHeader(command, sizeof(record));
        reply.header.command = peer::ResponseCommand(command);
        reply.candidate_id = id;
        reply.device_id = id == 0 ? 0 : id - 1;
        const unsigned indices[] = {0, 5, 0, 7, 0};
        reply.eid_index = indices[id];
        std::strcpy(reply.device_name, "devA");
        reply.device_name[3] += reply.device_id;
        record.eid[15] = id + 1;
        Check(send(fd, &record, sizeof(record), 0) == sizeof(record), "reply candidate EID");
      }
      close(fd);
    }
  });
  peer::PeerLink link("localhost:" + std::to_string(ntohs(addr.sin_port)));
  peer::PeerCandidateTable table;
  Check(link.DiscoverCandidates(&table).ok() && table.local, "actual connected peer is local");
  peer::CpuFeSelection selected;
  const auto choose = [&](const std::string& device, NpuEid eid, unsigned expected) {
    Check(link.SelectCpuFes(table, device, eid, false, &selected).ok() && selected.primary.candidate_id == expected,
          "CPU endpoint selection");
  };
  choose("", Eid(2), 1);
  choose("devA", Eid(2), 0);
  choose("devB", Eid(2), 2);
  choose("", Eid(99), 0);
  Check(!link.SelectCpuFes(table, "missing", Eid(2), false, &selected).ok(), "missing fixed device");
  Check(!link.SelectCpuFes(table, "devA,devB", {}, false, &selected).ok(), "no multi-FE on local links");
  table.device_fixed = true;
  choose("", Eid(2), 0);
  table.device_fixed = false;
  table.local = false;
  unsetenv("KVAF_DEFAULT_PEER_FE_COUNT");
  choose("", Eid(2), 0);
  std::vector<peer::PeerCandidateInfo> active;
  Check(selected.ForDirectGet(4, &active).ok() && active.size() == 2 &&
        active[0].candidate_id == 0 && active[1].candidate_id == 2, "two distinct FE index-zero rows by default");
  Check(selected.ForDirectGet(1, &active).ok() && active.size() == 1, "automatic Jetty cap");
  setenv("KVAF_DEFAULT_PEER_FE_COUNT", "8", 1);
  choose("", {}, 0);
  Check(selected.direct_get.size() == 3 && selected.direct_get.back().device_name == "devD",
        "automatic selection skips devices without index zero and caps at availability");
  setenv("KVAF_DEFAULT_PEER_FE_COUNT", "1", 1);
  choose("", {}, 0);
  Check(selected.direct_get.size() == 1, "configurable single FE default");
  choose("devD, devA ,devB", {}, 4);
  Check(selected.ForDirectGet(3, &active).ok() && active.size() == 3 &&
        active[1].candidate_id == 0 && active[2].candidate_id == 2, "explicit three FEs override count");
  CheckSessions(link.address(), active);
  Check(!selected.ForDirectGet(2, &active).ok(), "explicit list exceeds worker count");
  for (const auto* csv : {"devA,", ",devA", "devA,  ,devB", "devB,devB", "devC,devD"})
    Check(!link.SelectCpuFes(table, csv, {}, false, &selected).ok(), "invalid explicit FE list");
  choose("devC", {}, 3);
  Check(selected.primary.eid_index == 7, "single fixed device preserves first advertised index");
  setenv("KVAF_DEFAULT_PEER_FE_COUNT", "0", 1);
  Check(!link.SelectCpuFes(table, "", {}, false, &selected).ok(), "invalid automatic count");
  choose("devA,devB", {}, 0);  // Explicit selection ignores the environment.
  Check(link.SelectCpuFes(table, "", {}, true, &selected).ok() && selected.direct_get.size() == 1,
        "multiple local EIDs use one CPU FE");
  Check(!link.SelectCpuFes(table, "devA,devB", {}, true, &selected).ok(), "ambiguous many-to-many list");
  table.device_fixed = true;
  choose("", {}, 0);
  Check(selected.direct_get.size() == 1, "respect service pin on remote links");
  shutdown(listener, SHUT_RDWR);
  server.join();
  close(listener);
}
}  // namespace

int main() {
  NpuSelection();
  CpuSelection();
}

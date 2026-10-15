#include "host_memory.h"
#include <cstdio>

using namespace kvaf::aicpu_test;
void NodeDiscovery() {
  const auto find = [](const std::string& ubcore, const std::string& infiniband, unsigned* reads) {
    return FindUrmaNumaNode("test_device", [&](const std::string& path) {
      ++*reads;
      return path.find("/ubcore/") != std::string::npos ? ubcore : infiniband;
    });
  };
  unsigned reads = 0;
  Require(find("3", "2", &reads) == 3 && reads == 1, "UB node discovery failed");
  reads = 0;
  Require(find("-1", "0", &reads) == 0 && reads == 2, "alternate topology lookup failed");
  reads = 0;
  Require(find("", "", &reads) == -1 && reads == 2, "missing topology did not use default policy");
  reads = 0;
  Require(find("-1", "-1", &reads) == -1 && reads == 2, "unknown topology did not use default policy");
  for (const char* text : {"garbage", "4096"}) {
    bool rejected = false;
    try { (void)find(text, "", &reads); } catch (...) { rejected = true; }
    Require(rejected, "malformed topology accepted");
  }
  bool rejected = false;
  try { (void)FindUrmaNumaNode("../invalid", [](const std::string&) { return std::string{}; }); }
  catch (...) { rejected = true; }
  Require(rejected, "invalid device path accepted");
}
int main() {
  try {
    NodeDiscovery();
    Require(ParseCpuList("0-2,7,10-11") == std::vector<unsigned>({0,1,2,7,10,11}), "CPU list parsing failed");
    for (const char* text : {"2-1", "4-", "1,garbage", "1024", "0-1000000000"}) {
      bool rejected = false; try { (void)ParseCpuList(text); } catch (...) { rejected = true; }
      Require(rejected, "invalid CPU list accepted");
    }
    cpu_set_t before{}, after{};
    Require(sched_getaffinity(0, sizeof(before), &before) == 0, "read initial affinity");
    int cpu = 0; while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &before)) ++cpu;
    Require(cpu < CPU_SETSIZE, "no available CPU");
    {
      HostMemory allocation; HostMemoryOptions options; options.cpu = cpu;
      allocation.Allocate(8193, options, "test_device");
      auto* data = static_cast<unsigned char*>(allocation.data());
      for (std::size_t i = 0; i < allocation.bytes(); ++i) Require(data[i] == 0, "host first touch not zeroed");
      Require(reinterpret_cast<std::uintptr_t>(data) % sysconf(_SC_PAGESIZE) == 0, "host allocation not page aligned");
      Require(sched_getaffinity(0, sizeof(after), &after) == 0 && CPU_COUNT(&after) == 1 && CPU_ISSET(cpu, &after), "CPU bind failed");
      data[8192] = 0xA5;
    }
    Require(sched_getaffinity(0, sizeof(after), &after) == 0 && CPU_EQUAL(&before, &after), "affinity not restored");
    std::puts("host memory: PASS"); return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}

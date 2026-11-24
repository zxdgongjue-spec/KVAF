#include "cpu_service/urma/cpu_numa_scope.h"

#include <sched.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
int main() {
  char root[] = "/tmp/kvaf-numa-test-XXXXXX";
  if (!mkdtemp(root)) return 1;
  const auto write = [&](const std::string& file, const std::string& contents) {
    const auto path = fs::path(root) / file;
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << contents;
  };
  const auto local = std::to_string(sched_getcpu());
  const auto other = std::to_string(sysconf(_SC_NPROCESSORS_CONF) + CPU_SETSIZE);
  const auto remote = std::to_string(std::stoi(other) + 1);
  const std::string system = "devices/system/";
  write(system + "cpu/possible", local + "," + other + "," + remote);
  write(system + "cpu/online", local + "," + other + "," + remote);
  write(system + "node/online", "0-2");
  write(system + "node/node0/cpulist", local);
  write(system + "node/node1/cpulist", other);
  write(system + "node/node2/cpulist", remote);
  write(system + "cpu/cpu" + local + "/topology/physical_package_id", "0");
  write(system + "cpu/cpu" + other + "/topology/physical_package_id", "0");
  write(system + "cpu/cpu" + remote + "/topology/physical_package_id", "1");
  write(system + "cpu/cpu" + local + "/firmware_node/uid", "10");
  write(system + "cpu/cpu" + other + "/firmware_node/uid", "0x0b");
  write(system + "cpu/cpu" + remote + "/firmware_node/uid", "12");
  // Nodes 0 and 1 share chip 0. The third node belongs to chip 1.
  std::string srat(48 + 3 * 18, '\0');
  srat.replace(0, 4, "SRAT");
  srat[4] = static_cast<char>(srat.size());
  for (int i = 0; i < 3; ++i) {
    const auto pos = 48 + 18 * i;
    srat[pos] = 3;
    srat[pos + 1] = 18;
    srat[pos + 2] = 4 + 3 * i;
    srat[pos + 6] = 10 + i;
    srat[pos + 10] = 1;
  }
  unsigned sum = 0;
  for (unsigned char c : srat) sum += c;
  srat[9] = static_cast<char>(0u - sum);
  write("firmware/acpi/tables/SRAT", srat);
  write("class/ubcore/near/device/numa_node", "0");
  write("class/ubcore/sibling/device/numa_node", "1");
  write("class/ubcore/far/device/numa_node", "2");
  write("class/ubcore/near-pxm/device/numa", "0x04");
  write("class/ubcore/sibling-pxm/device/numa", "0x07");
  write("class/ubcore/far-pxm/device/numa", "0x0a");

  const auto chip_zero = [&] {
    kvaf::CpuNumaScope scope;
    return scope.Capture(root).ok() && scope.Allows("near") && scope.Allows("sibling") &&
           scope.Allows("near-pxm") && scope.Allows("sibling-pxm") && !scope.Allows("far") &&
           !scope.Allows("far-pxm") && !scope.Allows("unknown") &&
           scope.Allows("udmac0d1e2") && !scope.Allows("udmac1d1e2");
  };
  bool ok = chip_zero();
  write(system + "node/node0/cpulist", other);
  write(system + "node/node1/cpulist", local);
  ok = ok && chip_zero();
  write(system + "cpu/online", local);
  kvaf::CpuNumaScope unbound;
  ok = ok && unbound.Capture(root).ok() && unbound.Allows("unknown");
  fs::remove_all(root);
  if (!ok) std::cerr << "NUMA to CPU chip scoping failed\n";
  return ok ? 0 : 1;
}

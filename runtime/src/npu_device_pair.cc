#include "npu_eid_selection.h"
#include "npu_eid_discovery.h"

namespace kvaf::detail {
namespace {
constexpr char kUbDevIdPath[] = "/proc/asdrv_ub/dev_id";
constexpr char kUbPairInfoPath[] = "/proc/asdrv_ub/pair_info";
constexpr char kUbSelectorLockPath[] = "/tmp/kvaf_npu_runtime_asdrv_ub.lock";
}  // namespace

bool ResolvePhysicalDeviceId(std::uint32_t device, std::uint32_t* physical) {
  return npu_eid::ResolvePhysicalDeviceId(device, physical);
}

PairEids ReadDevicePair(std::uint32_t physical_device) {
  PairEids pair;
  const int fd = open(kUbSelectorLockPath, O_CREAT | O_RDWR, 0600);
  if (fd >= 0 && flock(fd, LOCK_EX) == 0) {
    std::ofstream selector(kUbDevIdPath);
    selector << physical_device << '\n';
    selector.close();
    if (selector) {
      std::ifstream input(kUbPairInfoPath);
      pair = ParsePairInfo(input, physical_device);
    }
  }
  if (fd >= 0) close(fd);  // Releases the selector lock.
  std::fprintf(stderr, "[KVAF] PAIR_INFO device=%u cpu=%s npu=%s\n", physical_device,
               pair.cpu ? EidToHex(*pair.cpu).c_str() : "unavailable",
               pair.npu ? EidToHex(*pair.npu).c_str() : "unavailable");
  return pair;
}

}  // namespace kvaf::detail

#pragma once

#include <string>
#include <vector>

#include "status.h"

namespace kvaf {

enum class Transport { kUrma, kUbmem };

namespace detail {
enum class RuntimeBackend { kUrmaAiv, kUrmaAicpuGather, kUrmaAicpuScatter, kUbmem };

Status ResolveRuntimeBackend(Transport transport, const std::string& mode,
                             RuntimeBackend* backend);
inline bool UsesAicpu(RuntimeBackend backend) {
  return backend == RuntimeBackend::kUrmaAicpuGather ||
         backend == RuntimeBackend::kUrmaAicpuScatter;
}
// Only AIV groups explicit EIDs into simultaneous paths. Both AICPU modes
// try one candidate at a time, including when the caller supplies a list.
inline bool StripesEids(RuntimeBackend backend, bool explicit_eids) {
  return backend == RuntimeBackend::kUrmaAiv && explicit_eids;
}

struct AicpuPackageSearch {
  std::string ascend_home_path;
  std::string ascend_cann_package_path;
  std::vector<std::string> default_roots{
      "/usr/local/Ascend/ascend-toolkit/latest",
      "/usr/local/Ascend/cann/aarch64-linux/asc"};
};

struct AicpuKernelPackage {
  std::string cann_root;
  std::string json_path;
};

// No ACL calls or device/session state: package discovery can fail before
// allocating remote DDR. Explicit roots never fall back to another install.
Status ResolveAicpuKernelPackage(const AicpuPackageSearch& search,
                                AicpuKernelPackage* package);
}  // namespace detail
}  // namespace kvaf

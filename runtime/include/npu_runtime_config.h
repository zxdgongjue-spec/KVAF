#pragma once

#include <array>
#include <string>
#include <vector>

#include "status.h"
#include "execution_config.h"
namespace kvaf {
namespace detail {
using NpuEid = std::array<std::uint8_t, 16>;

// Apply routing only after selecting a provider and resolving its own budget.
// An explicit AIV Get list is strict: never truncate it to fit worker capacity.
Status SelectProviderEids(Transport transport, Executor executor, bool get,
                          bool explicit_eids, const ExecutionResources& resources,
                          const std::vector<NpuEid>& candidates,
                          std::vector<NpuEid>* selected);

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

// No ACL calls or device/session state. Called on first AICPU preparation;
// explicit roots never fall back to another install.
Status ResolveAicpuKernelPackage(const AicpuPackageSearch& search,
                                AicpuKernelPackage* package);
}  // namespace detail
}  // namespace kvaf

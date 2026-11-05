#include "npu_runtime_config.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace kvaf::detail {
Status ResolveRuntimeBackend(Transport transport, const std::string& mode,
                             RuntimeBackend* backend) {
  if (!backend) return Status::InvalidArgument("backend output is null");
  if (transport != Transport::kUrma && transport != Transport::kUbmem)
    return Status::InvalidArgument("transport must be urma or ubmem");
  if (!mode.empty() && mode != "gather_read" && mode != "cpu_push_scatter")
    return Status::InvalidArgument(
        "aicpu_batch_get_mode must be empty, gather_read or cpu_push_scatter");
  if (transport == Transport::kUbmem) {
    if (!mode.empty()) return Status::InvalidArgument(
        "ubmem requires an empty aicpu_batch_get_mode");
    *backend = RuntimeBackend::kUbmem;
  } else if (mode.empty()) {
    *backend = RuntimeBackend::kUrmaAiv;
  } else {
    *backend = mode == "gather_read" ? RuntimeBackend::kUrmaAicpuGather
                                     : RuntimeBackend::kUrmaAicpuScatter;
  }
  return Status::OK();
}

namespace {
bool ReadableFile(const std::string& path) {
  struct stat info{};
  return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
         info.st_size > 0 && access(path.c_str(), R_OK) == 0;
}

Status CanonicalRoot(const std::string& input, std::string* root) {
  char resolved[PATH_MAX];
  struct stat info{};
  if (!realpath(input.c_str(), resolved) || stat(resolved, &info) != 0 ||
      !S_ISDIR(info.st_mode))
    return Status::NotFound("invalid CANN root: " + input);
  *root = resolved;
  return Status::OK();
}

Status CheckLoaderConfig(const std::string& path) {
  if (!ReadableFile(path)) return Status::NotFound("missing CANN loader config: " + path);
  const std::array<std::string, 5> expected{
      "name:kvaf-aicpu-urma.tar.gz", "install_path:2", "optional:true",
      "package_path:opp/vendors/cust/op_impl/aicpu/kernel", "load_as_per_soc:false"};
  std::ifstream file(path);
  std::string line;
  unsigned count = 0;
  while (std::getline(file, line)) {
    if (line != expected[0]) continue;
    ++count;
    for (std::size_t i = 1; i < expected.size(); ++i) {
      if (!std::getline(file, line) || line != expected[i])
        return Status::InvalidArgument("invalid KVAF AICPU registration in " + path);
    }
  }
  if (count != 1)
    return Status::InvalidArgument("expected one KVAF AICPU registration in " + path);
  return Status::OK();
}
}  // namespace

Status ResolveAicpuKernelPackage(const AicpuPackageSearch& search,
                                AicpuKernelPackage* package) {
  if (!package) return Status::InvalidArgument("package output is null");
  AicpuKernelPackage found;
  const bool has_home = !search.ascend_home_path.empty();
  const bool has_cann = !search.ascend_cann_package_path.empty();
  if (has_home || has_cann) {
    auto status = CanonicalRoot(has_home ? search.ascend_home_path
                                        : search.ascend_cann_package_path, &found.cann_root);
    if (!status.ok()) return status;
    if (has_home && has_cann) {
      std::string other;
      status = CanonicalRoot(search.ascend_cann_package_path, &other);
      if (!status.ok()) return status;
      if (other != found.cann_root)
        return Status::InvalidArgument("CANN roots differ: ASCEND_HOME_PATH=" + found.cann_root +
            " ASCEND_CANN_PACKAGE_PATH=" + other);
    }
  } else {
    for (const auto& candidate : search.default_roots) {
      std::string root;
      if (CanonicalRoot(candidate, &root).ok() &&
          ReadableFile(root + "/conf/ascend_package_load.ini")) {
        found.cann_root = std::move(root);
        break;
      }
    }
    if (found.cann_root.empty()) return Status::NotFound(
        "CANN root not found; source CANN set_env.sh or set ASCEND_HOME_PATH");
  }
  found.json_path = found.cann_root + "/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json";
  if (!ReadableFile(found.json_path))
    return Status::NotFound("missing installed AICPU kernel JSON: " + found.json_path);
  const auto archive = found.cann_root + "/opp/vendors/cust/op_impl/aicpu/kernel/kvaf-aicpu-urma.tar.gz";
  if (!ReadableFile(archive)) return Status::NotFound("missing AICPU device package: " + archive);
  auto status = CheckLoaderConfig(found.cann_root + "/conf/ascend_package_load.ini");
  if (!status.ok()) return status;
  char resolved[PATH_MAX];
  if (!realpath(found.json_path.c_str(), resolved))
    return Status::NotFound("cannot resolve AICPU kernel JSON: " + found.json_path);
  found.json_path = resolved;
  *package = std::move(found);
  return Status::OK();
}
}  // namespace kvaf::detail

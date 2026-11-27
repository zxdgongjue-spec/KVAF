#include "npu_runtime_config.h"
#include "npu_kernel_paths.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace kvaf::detail {

ExecutionResources ResolveExecutionResources(Transport transport, Executor executor, bool get, bool peer_local,
                                             const ExecutionResources* configured) {
  ExecutionResources resources;
  resources.jettys = 2U;
  if (transport == Transport::Ubmem) {
    resources.cores = 16U;
  } else if (executor == Executor::Aicpu) {
    resources.cores = get ? 5U : 1U;
  } else {
    const auto defaults = DefaultUrmaAivParallelism(get, peer_local);
    resources.cores = defaults.cores;
    resources.jettys = defaults.jettys_per_core;
  }
  resources.max_inflight_total = 2048U;
  if (executor == Executor::Aiv)
    resources.kernel_path = get ? kDefaultBatchGetKernelPath : kDefaultBatchPutKernelPath;
  if (configured) {
    if (configured->cores) resources.cores = configured->cores;
    if (configured->jettys) resources.jettys = configured->jettys;
    if (configured->max_inflight_total)
      resources.max_inflight_total = configured->max_inflight_total;
    resources.cpu_threads = configured->cpu_threads;
    resources.sge_per_wr = configured->sge_per_wr;
    resources.sgl_payload_only = configured->sgl_payload_only;
    resources.sgl_pipeline = configured->sgl_pipeline;
    resources.sgl_push_chunked = configured->sgl_push_chunked;
    resources.sgl_push_head_rows = configured->sgl_push_head_rows;
    resources.sgl_push_cooperative = configured->sgl_push_cooperative;
    resources.sgl_push_publisher_only = configured->sgl_push_publisher_only;
    resources.sgl_push_head_cooperative = configured->sgl_push_head_cooperative;
    resources.request_capacity = configured->request_capacity;
    if (!configured->kernel_path.empty()) resources.kernel_path = configured->kernel_path;
  }
  return resources;
}

Status SelectProviderEids(Transport transport, Executor executor, bool get,
                          bool explicit_eids, const ExecutionResources& resources,
                          const std::vector<NpuEid>& candidates,
                          std::vector<NpuEid>* selected) {
  if (!selected) return Status::InvalidArgument("EID output is null");
  if (transport == Transport::Ubmem) {
    selected->clear();
    return Status::OK();
  }
  if (candidates.empty()) return Status::NotInitialized("URMA local EIDs are not initialized");
  if (get && executor == Executor::Aiv && explicit_eids) {
    const auto workers = static_cast<std::uint64_t>(resources.cores) * resources.jettys;
    if (candidates.size() > workers)
      return Status::InvalidArgument("npu_eids count exceeds selected AIV Get cores * jettys workers");
    *selected = candidates;
  } else {
    selected->assign(1, candidates.front());
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

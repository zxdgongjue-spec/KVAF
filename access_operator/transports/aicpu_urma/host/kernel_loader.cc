#include "aicpu_urma/kernel_loader.h"

#include <limits.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
namespace kvaf::aicpu_urma {
namespace {
std::mutex kernel_mutex;
aclrtBinHandle binary = nullptr;
std::string loaded_path;
struct KernelEntry {
  const char* name;
  std::atomic<aclrtFuncHandle> function{nullptr};
};
KernelEntry functions[] = {
    {"KvafAicpuUmdkControlSessionsV5"}, {"KvafAicpuBatchPut"},
    {"KvafAicpuGatherReadExecutionV5"}, {"KvafAicpuCpuPushScatterExecutionV4"}};

aclrtFuncHandle ResolveFunction(const char* name) {
  for (auto& entry : functions) {
    if (std::strcmp(entry.name, name) != 0) continue;
    auto function = entry.function.load(std::memory_order_acquire);
    if (function) return function;
    std::lock_guard<std::mutex> lock(kernel_mutex);
    if (!binary) return nullptr;
    function = entry.function.load(std::memory_order_relaxed);
    if (!function) {
      if (aclrtBinaryGetFunction(binary, name, &function) != ACL_SUCCESS || !function) return nullptr;
      // The process keeps the loaded binary alive; published handles never change.
      entry.function.store(function, std::memory_order_release);
    }
    return function;
  }
  return nullptr;
}
int32_t Launch(aclrtFuncHandle function, void* arguments, std::size_t bytes, aclrtStream stream,
               std::uint32_t blocks = 1U) {
  if (function == nullptr || arguments == nullptr || stream == nullptr) return -1;
  aclrtLaunchKernelAttr attribute{};
  attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  attribute.value.timeout = 180U;
  aclrtLaunchKernelCfg config{};
  config.numAttrs = 1U;
  config.attrs = &attribute;
  const aclError result = aclrtLaunchKernelWithHostArgs(function, blocks, stream, &config,
                                                        arguments, bytes, nullptr, 0U);
  if (result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] kernel launch failed acl=%d\n",
                 static_cast<int>(result));
    return -1;
  }
  return 0;
}

}  // namespace
bool LoadUmdkKernel(const char* json_path) {
  if (json_path == nullptr || json_path[0] == '\0') return false;
  std::lock_guard<std::mutex> lock(kernel_mutex);
  char resolved[PATH_MAX];
  if (realpath(json_path, resolved) == nullptr) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] missing kernel JSON path=%s\n", json_path);
    return false;
  }
  if (binary != nullptr) {
    if (loaded_path == resolved) return true;
    std::fprintf(stderr, "[KVAF_AICPU_URMA] kernel already loaded from %s; requested %s\n",
                 loaded_path.c_str(), resolved);
    return false;
  }
  aclrtBinaryLoadOption option{};
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  option.value.cpuKernelMode = 0;
  aclrtBinaryLoadOptions options{};
  options.options = &option;
  options.numOpt = 1;
  aclrtBinHandle loaded = nullptr;
  if (aclrtBinaryLoadFromFile(resolved, &options, &loaded) != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] failed to load AICPU kernel path=%s\n", resolved);
    if (loaded) (void)aclrtBinaryUnLoad(loaded);
    return false;
  }
  loaded_path = resolved;
  binary = loaded;
  return true;
}
Status PrepareUmdkKernel(const std::string& json, UmdkKernelKind kind) {
  const char* mode = nullptr;
  const char* entry = nullptr;
  switch (kind) {
    case UmdkKernelKind::Control: mode = "control"; break;
    case UmdkKernelKind::Put: mode = "put"; entry = "KvafAicpuBatchPut"; break;
    case UmdkKernelKind::Gather: mode = "gather_read"; entry = "KvafAicpuGatherReadExecutionV5"; break;
    case UmdkKernelKind::Scatter: mode = "cpu_push_scatter"; entry = "KvafAicpuCpuPushScatterExecutionV4"; break;
    default: return Status::InvalidArgument("unknown AICPU kernel kind");
  }
  const std::string context = "AICPU mode=" + std::string(mode) + " json=" + json;
  if (!LoadUmdkKernel(json.c_str()))
    return Status::Internal(context + ": failed to load installed device package");
  // Scatter uses its Host RA transport, not the AICPU URMA control session.
  for (const char* name : {kind == UmdkKernelKind::Scatter ? nullptr : "KvafAicpuUmdkControlSessionsV5",
                          entry}) {
    if (!name) continue;
    if (!ResolveFunction(name))
      return Status::NotFound(context + ": missing kernel entry " + name +
          "; rebuild and reinstall the matching AICPU package, then restart the client");
  }
  return Status::OK();
}
int32_t LaunchUmdkKernel(const char* name, const void* args, std::size_t bytes, aclrtStream stream,
                         std::uint32_t blocks) {
  if (!name || !args) return -1;
  const auto function = ResolveFunction(name);
  return Launch(function, const_cast<void*>(args), bytes, stream, blocks);
}
}  // namespace kvaf::aicpu_urma

#include "aicpu_urma/kernel_loader.h"

#include <limits.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
namespace kvaf::aicpu_urma {
namespace {
std::mutex kernel_mutex;
aclrtBinHandle binary = nullptr;
std::string loaded_path;
std::unordered_map<std::string, aclrtFuncHandle> functions;
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
Status PrepareUmdkKernel(const std::string& json, bool scatter) {
  const std::string context = "AICPU mode=" + std::string(scatter ? "cpu_push_scatter" : "gather_read") +
                              " json=" + json;
  if (!LoadUmdkKernel(json.c_str()))
    return Status::Internal(context + ": failed to load installed device package");
  std::lock_guard<std::mutex> lock(kernel_mutex);
  for (const char* name : {"KvafAicpuUmdkControl", "KvafAicpuBatchPut",
                          scatter ? "KvafAicpuCpuPushScatterA12V24" : "KvafAicpuGatherRead"}) {
    if (functions.find(name) != functions.end()) continue;
    aclrtFuncHandle function = nullptr;
    if (aclrtBinaryGetFunction(binary, name, &function) != ACL_SUCCESS || !function)
      return Status::NotFound(context + ": missing kernel entry " + name +
          "; rebuild and reinstall the matching AICPU package, then restart the client");
    functions.emplace(name, function);
  }
  return Status::OK();
}
int32_t LaunchUmdkKernel(const char* name, const void* args, std::size_t bytes, aclrtStream stream,
                         std::uint32_t blocks) {
  if (!name || !args) return -1;
  aclrtFuncHandle function = nullptr;
  {
    std::lock_guard<std::mutex> lock(kernel_mutex);
    if (!binary) return -1;
    auto found = functions.find(name);
    if (found != functions.end())
      function = found->second;
    else {
      if (aclrtBinaryGetFunction(binary, name, &function) != ACL_SUCCESS) return -1;
      functions.emplace(name, function);
    }
  }
  return Launch(function, const_cast<void*>(args), bytes, stream, blocks);
}
}  // namespace kvaf::aicpu_urma

#include "host.h"

#include <limits.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
namespace kvaf::aicpu_urma {
namespace {
std::mutex kernel_mutex;
aclrtBinHandle binary = nullptr;
aclrtFuncHandle gather_function = nullptr, scatter_function = nullptr;
aclrtFuncHandle control_function = nullptr;
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
bool Load(const char* json_path) {
  if (json_path == nullptr || json_path[0] == '\0') return false;
  std::lock_guard<std::mutex> lock(kernel_mutex);
  if (binary != nullptr) return true;
  char resolved[PATH_MAX];
  if (realpath(json_path, resolved) == nullptr) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] missing kernel JSON path=%s\n", json_path);
    return false;
  }
  aclrtBinaryLoadOption option{};
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  option.value.cpuKernelMode = 0;
  aclrtBinaryLoadOptions options{};
  options.options = &option;
  options.numOpt = 1;
  aclrtBinHandle loaded = nullptr;
  if (aclrtBinaryLoadFromFile(resolved, &options, &loaded) != ACL_SUCCESS ||
      aclrtBinaryGetFunction(loaded, "KvafAicpuGatherRead", &gather_function) != ACL_SUCCESS ||
      aclrtBinaryGetFunction(loaded, "KvafAicpuGatherScatter", &scatter_function) != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] failed to load AICPU kernel path=%s\n", resolved);
    if (loaded != nullptr) (void)aclrtBinaryUnLoad(loaded);
    gather_function = nullptr;
    scatter_function = nullptr;
    return false;
  }
  binary = loaded;
  return true;
}
int32_t LaunchUmdkControl(const UmdkControlArgs* arguments, aclrtStream stream) {
  if (arguments == nullptr) return -1;
  {
    std::lock_guard<std::mutex> lock(kernel_mutex);
    if (binary == nullptr) return -1;
    if (control_function == nullptr &&
        aclrtBinaryGetFunction(binary, "KvafAicpuUmdkControl", &control_function) != ACL_SUCCESS) {
      std::fprintf(stderr, "[KVAF_AICPU_URMA] UMDK control missing; rebuild and install kernel\n");
      return -1;
    }
  }
  return Launch(control_function, const_cast<UmdkControlArgs*>(arguments), sizeof(*arguments),
                stream);
}
int32_t LaunchUmdkGet(const UmdkBatchGetArgs* args, aclrtStream stream) {
  if (!args) return -1;
  if (Launch(gather_function, const_cast<UmdkBatchGetArgs*>(args), sizeof(*args), stream))
    return -1;
  return args->gather.worker_count <= 1U
             ? 0
             : Launch(scatter_function, const_cast<UmdkBatchGetArgs*>(args), sizeof(*args), stream,
                      args->gather.worker_count);
}
}  // namespace kvaf::aicpu_urma

#include "host.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <mutex>

namespace kvaf::aicpu_urma {
namespace {

std::mutex kernel_mutex;
aclrtBinHandle binary = nullptr;
aclrtFuncHandle put_function = nullptr;
aclrtFuncHandle get_function = nullptr;
aclrtFuncHandle gather_function = nullptr;
aclrtFuncHandle scatter_function = nullptr;
aclrtFuncHandle sq_copy_function = nullptr;
aclrtFuncHandle sq_verify_function = nullptr;

int32_t Launch(aclrtFuncHandle function, void* arguments, std::size_t bytes,
               aclrtStream stream, std::uint32_t blocks = 1U) {
  if (function == nullptr || arguments == nullptr || stream == nullptr) return -1;
  aclrtLaunchKernelAttr attribute{};
  attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  attribute.value.timeout = 180U;
  aclrtLaunchKernelCfg config{};
  config.numAttrs = 1U;
  config.attrs = &attribute;
  const aclError result = aclrtLaunchKernelWithHostArgs(function, blocks, stream,
                                                       &config, arguments, bytes, nullptr, 0U);
  if (result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] kernel launch failed acl=%d\n", static_cast<int>(result));
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
      aclrtBinaryGetFunction(loaded, "KvafAicpuUrmaBatchPut", &put_function) != ACL_SUCCESS ||
      aclrtBinaryGetFunction(loaded, "KvafAicpuUrmaBatchGet", &get_function) != ACL_SUCCESS ||
      aclrtBinaryGetFunction(loaded, "KvafAicpuUrmaGatherRead", &gather_function) != ACL_SUCCESS ||
      aclrtBinaryGetFunction(loaded, "KvafAicpuUrmaGatherScatter", &scatter_function) != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] failed to load AICPU kernel path=%s\n", resolved);
    if (loaded != nullptr) (void)aclrtBinaryUnLoad(loaded);
    put_function = nullptr;
    get_function = nullptr;
    gather_function = nullptr;
    scatter_function = nullptr;
    return false;
  }
  binary = loaded;
  return true;
}

int32_t LaunchPut(const KvafExecuteBatchPutLaunchArgs* arguments, aclrtStream stream) {
  return Launch(put_function, const_cast<KvafExecuteBatchPutLaunchArgs*>(arguments),
                sizeof(*arguments), stream);
}

int32_t LaunchGet(const KvafExecuteBatchGetKernelArgs* arguments, aclrtStream stream) {
  return Launch(get_function, const_cast<KvafExecuteBatchGetKernelArgs*>(arguments),
                sizeof(*arguments), stream);
}

int32_t LaunchGatherRead(const GatherReadArgs* arguments, aclrtStream stream) {
  if (Launch(gather_function, const_cast<GatherReadArgs*>(arguments),
             sizeof(*arguments), stream) != 0) return -1;
  if (arguments->worker_count <= 1U) return 0;
  return Launch(scatter_function, const_cast<GatherReadArgs*>(arguments),
                sizeof(*arguments), stream, arguments->worker_count);
}

int32_t LaunchSqCopy(const SqCopyBenchmarkArgs* arguments, aclrtStream stream) {
  if (arguments == nullptr) return -1;
  {
    std::lock_guard<std::mutex> lock(kernel_mutex);
    if (binary == nullptr) return -1;
    if (sq_copy_function == nullptr &&
        aclrtBinaryGetFunction(binary, "KvafAicpuUrmaSqCopyBenchmark", &sq_copy_function) != ACL_SUCCESS) {
      std::fprintf(stderr, "[KVAF_AICPU_URMA] SQ copy kernel missing; rebuild and install kernel\n");
      return -1;
    }
  }
  return Launch(sq_copy_function, const_cast<SqCopyBenchmarkArgs*>(arguments),
                sizeof(*arguments), stream, arguments->workers);
}

int32_t LaunchSqCopyBenchmark(const SqCopyBenchmarkArgs* arguments, aclrtStream stream) {
  if (LaunchSqCopy(arguments, stream) != 0) return -1;
  const auto result = LaunchSqCopyVerify(arguments, stream);
  if (result != 0) (void)aclrtSynchronizeStream(stream);
  return result;
}

int32_t LaunchSqCopyVerify(const SqCopyBenchmarkArgs* arguments, aclrtStream stream) {
  if (arguments == nullptr) return -1;
  {
    std::lock_guard<std::mutex> lock(kernel_mutex);
    if (binary == nullptr) return -1;
    if (sq_verify_function == nullptr &&
        aclrtBinaryGetFunction(binary, "KvafAicpuUrmaSqCopyVerify", &sq_verify_function) != ACL_SUCCESS) {
      std::fprintf(stderr, "[KVAF_AICPU_URMA] SQ verify kernel missing; rebuild and install kernel\n");
      return -1;
    }
  }
  return Launch(sq_verify_function, const_cast<SqCopyBenchmarkArgs*>(arguments),
                sizeof(*arguments), stream, arguments->workers);
}

}  // namespace kvaf::aicpu_urma

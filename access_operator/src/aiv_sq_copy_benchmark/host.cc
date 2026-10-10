#include "host.h"

#include <fstream>
#include <mutex>
#include <vector>

namespace kvaf::aiv_sq_copy {
namespace {

std::mutex loader_mutex;
aclrtBinHandle binary_handle = nullptr;
aclrtFuncHandle copy_function = nullptr;
aclrtFuncHandle clock_function = nullptr;
std::vector<char> binary_bytes;

int32_t Launch(aclrtFuncHandle function, const aicpu_urma::SqCopyBenchmarkArgs* args,
               aclrtStream stream, std::uint32_t workers) {
  if (function == nullptr || args == nullptr || stream == nullptr || workers == 0U ||
      workers > aicpu_urma::kSqCopyBenchmarkMaxWorkers) return -1;
  aclrtLaunchKernelAttr attributes[4]{};
  attributes[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attributes[0].value.schemMode = 1;
  attributes[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attributes[1].value.dynUBufSize = kLocalMemoryBytes;
  attributes[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
  attributes[2].value.timeoutUs.timeoutLow = 180000000U;
  attributes[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attributes[3].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
  aclrtLaunchKernelCfg configuration{};
  configuration.numAttrs = 4U;
  configuration.attrs = attributes;
  return aclrtLaunchKernelWithHostArgs(function, workers, stream, &configuration,
      const_cast<aicpu_urma::SqCopyBenchmarkArgs*>(args), sizeof(*args), nullptr, 0U) == ACL_SUCCESS
      ? 0 : -1;
}

}

bool Load(const char* object_path) {
  if (object_path == nullptr || object_path[0] == '\0') return false;
  std::lock_guard<std::mutex> lock(loader_mutex);
  if (binary_handle != nullptr) return true;
  std::ifstream file(object_path, std::ios::binary | std::ios::ate);
  if (!file.is_open() || file.tellg() <= 0) return false;
  binary_bytes.resize(static_cast<std::size_t>(file.tellg()));
  file.seekg(0, std::ios::beg);
  if (!file.read(binary_bytes.data(), binary_bytes.size()).good()) return false;
  const auto binary = aclrtCreateBinary(binary_bytes.data(), binary_bytes.size());
  if (binary == nullptr) return false;
  aclrtBinHandle loaded = nullptr;
  const auto result = aclrtBinaryLoad(binary, &loaded);
  aclrtDestroyBinary(binary);
  if (result != ACL_SUCCESS) return false;
  if (aclrtBinaryGetFunction(loaded, "KvafAivSqCopyBenchmark", &copy_function) != ACL_SUCCESS ||
      aclrtBinaryGetFunction(loaded, "KvafAivSqCopyClock", &clock_function) != ACL_SUCCESS) {
    (void)aclrtBinaryUnLoad(loaded);
    copy_function = nullptr;
    clock_function = nullptr;
    return false;
  }
  binary_handle = loaded;
  return true;
}

int32_t LaunchCopy(const aicpu_urma::SqCopyBenchmarkArgs* args, aclrtStream stream) {
  return Launch(copy_function, args, stream, args == nullptr ? 0U : args->workers);
}

int32_t LaunchClock(const aicpu_urma::SqCopyBenchmarkArgs* args, aclrtStream stream) {
  return Launch(clock_function, args, stream, 1U);
}

}

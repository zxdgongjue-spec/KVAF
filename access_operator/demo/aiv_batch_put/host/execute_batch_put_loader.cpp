#include "kvaf_aiv_batch_put.h"
#include "batch_put_internal.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

#include "acl/acl.h"

namespace {
constexpr char kKernelName[] = "KvafExecuteBatchPutAivKernel";
constexpr uint32_t kKernelTimeoutSeconds = 180U;
constexpr uint32_t kKernelLocalMemoryBytes = 128U * 1024U;
std::mutex g_mutex;
bool g_loaded = false;
aclrtBinHandle g_binary = nullptr;
aclrtFuncHandle g_function = nullptr;
std::vector<char> g_binary_bytes;

bool ReadBinary(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) return false;
  const std::streamsize size = file.tellg();
  if (size <= 0) return false;
  file.seekg(0, std::ios::beg);
  g_binary_bytes.resize(static_cast<size_t>(size));
  return file.read(g_binary_bytes.data(), size).good();
}
}  // namespace

extern "C" int32_t KvafExecuteBatchPutLoadKernel(const char *kernel_object_path) {
  if (kernel_object_path == nullptr || kernel_object_path[0] == '\0') return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_loaded) return KVAF_BATCH_PUT_SUCCESS;
  if (!ReadBinary(kernel_object_path)) return KVAF_BATCH_PUT_RUNTIME_ERROR;
  aclrtBinary binary = aclrtCreateBinary(g_binary_bytes.data(), g_binary_bytes.size());
  if (binary == nullptr) return KVAF_BATCH_PUT_RUNTIME_ERROR;
  const aclError load = aclrtBinaryLoad(binary, &g_binary);
  aclrtDestroyBinary(binary);
  if (load != ACL_SUCCESS || aclrtBinaryGetFunction(g_binary, kKernelName, &g_function) != ACL_SUCCESS) {
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  g_loaded = true;
  return KVAF_BATCH_PUT_SUCCESS;
}

extern "C" int32_t KvafExecuteBatchPutLaunchInternal(const KvafExecuteBatchPutLaunchArgs *arguments,
                                                       aclrtStream stream) {
  if (!g_loaded || g_function == nullptr || arguments == nullptr || stream == nullptr ||
      arguments->request.aiv_core_count == 0U ||
      arguments->request.aiv_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
      arguments->request.row_count == 0U || arguments->request.input_bytes == 0U ||
      arguments->channel_entity == 0U || arguments->queue_state == 0U ||
      arguments->jettys_per_core == 0U ||
      arguments->jettys_per_core > KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  aclrtLaunchKernelAttr attrs[4]{};
  attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attrs[0].value.schemMode = 1;
  attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attrs[1].value.dynUBufSize = kKernelLocalMemoryBytes;
  attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
  attrs[2].value.timeoutUs.timeoutLow = kKernelTimeoutSeconds * 1000000U;
  attrs[2].value.timeoutUs.timeoutHigh = 0U;
  attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attrs[3].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
  aclrtLaunchKernelCfg config{};
  config.numAttrs = 4U;
  config.attrs = attrs;
  const aclError ret = aclrtLaunchKernelWithHostArgs(g_function, arguments->request.aiv_core_count,
                                                     stream, &config,
                                                     const_cast<KvafExecuteBatchPutLaunchArgs *>(arguments),
                                                     sizeof(*arguments), nullptr, 0U);
  return ret == ACL_SUCCESS ? KVAF_BATCH_PUT_SUCCESS : KVAF_BATCH_PUT_RUNTIME_ERROR;
}

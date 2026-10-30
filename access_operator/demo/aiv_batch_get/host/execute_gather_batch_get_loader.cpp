#include "kvaf_aiv_batch_get.h"
#include "gather_batch_get.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace {
constexpr char kKernelName[] = "KvafGatherBatchGetAivKernelV2";
constexpr uint32_t kKernelTimeoutSeconds = 180U;
constexpr uint32_t kKernelLocalMemoryBytes =
    KVAF_AIV_BATCH_GET_LOCAL_MEMORY_BYTES;
std::mutex g_mutex;
bool g_loaded = false;
aclrtBinHandle g_binary = nullptr;
aclrtFuncHandle g_function = nullptr;
std::vector<char> g_binary_bytes;

std::string SequentialBinaryPath(const char* configured) {
  std::string path(configured);
  const auto separator = path.find_last_of("/\\");
  const auto filename = separator == std::string::npos ? 0 : separator + 1;
  // Standard and legacy selector paths resolve to this mode's own binary.
  if (path.compare(filename, std::string::npos, "kvaf_execute_batch_get_kernels.o") == 0 ||
      path.compare(filename, std::string::npos, "kvaf_sequential_batch_get_kernels.o") == 0)
    path.replace(filename, std::string::npos, "kvaf_gather_batch_get_kernels.o");
  return path;  // Explicit sequential/custom binary paths are used as-is.
}

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

extern "C" int32_t KvafGatherBatchGetLoadKernel(const char *kernel_object_path) {
  if (kernel_object_path == nullptr || kernel_object_path[0] == '\0') return KVAF_BATCH_GET_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_loaded) return KVAF_BATCH_GET_SUCCESS;
  const auto path = SequentialBinaryPath(kernel_object_path);
  if (!ReadBinary(path.c_str())) {
    std::fprintf(stderr, "[KVAF_AIV] cannot read gather BatchGet kernel: %s; "
        "rebuild and deploy kvaf_gather_batch_get_kernels.o alongside the direct-read binary\n",
        path.c_str());
    return KVAF_BATCH_GET_RUNTIME_ERROR;
  }
  aclrtBinary binary = aclrtCreateBinary(g_binary_bytes.data(), g_binary_bytes.size());
  if (binary == nullptr) return KVAF_BATCH_GET_RUNTIME_ERROR;
  const aclError load = aclrtBinaryLoad(binary, &g_binary);
  aclrtDestroyBinary(binary);
  if (load != ACL_SUCCESS || aclrtBinaryGetFunction(g_binary, kKernelName, &g_function) != ACL_SUCCESS) {
    return KVAF_BATCH_GET_RUNTIME_ERROR;
  }
  g_loaded = true;
  return KVAF_BATCH_GET_SUCCESS;
}

extern "C" int32_t KvafGatherBatchGetLaunch(const KvafGatherBatchGetArgs *arguments,
                                                void* stream) {
  if (!g_loaded || !g_function || !arguments || !stream ||
      !arguments->request.block_count || arguments->request.block_count > KVAF_SEQ_MAX_WORKERS)
    return KVAF_BATCH_GET_INVALID_ARGUMENT;
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
  const aclError ret = aclrtLaunchKernelWithHostArgs(g_function, arguments->request.block_count, stream, &config,
                                                     const_cast<KvafGatherBatchGetArgs *>(arguments),
                                                     sizeof(*arguments), nullptr, 0U);
  return ret == ACL_SUCCESS ? KVAF_BATCH_GET_SUCCESS : KVAF_BATCH_GET_RUNTIME_ERROR;
}

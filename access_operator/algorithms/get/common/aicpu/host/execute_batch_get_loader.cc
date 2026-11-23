#include "get/common/aicpu/include/kvaf_aicpu_batch_get.h"
#include "aicpu_urma/kernel_loader.h"
namespace kvaf::aicpu_urma {
int32_t LaunchBatchGet(const UmdkBatchGetArgs* args, aclrtStream stream) {
  if (!args) return -1;
  return LaunchUmdkKernel("KvafAicpuGatherReadExecutionV3", args, sizeof(*args), stream,
                          args->gather.worker_count);
}
}  // namespace kvaf::aicpu_urma
namespace kvaf::cpu_push {
int32_t LaunchCpuPushScatter(const KernelArgs* args, aclrtStream stream) {
  if (!args) return -1;
  return aicpu_urma::LaunchUmdkKernel("KvafAicpuCpuPushScatterExecutionV3", args, sizeof(*args), stream,
                                    args->workers);
}
int32_t LaunchCpuSglWrite(const CpuSglWriteArgs* args, aclrtStream stream) {
  if (!args) return -1;
  return aicpu_urma::LaunchUmdkKernel("KvafAicpuCpuSglWriteExecutionV1", args, sizeof(*args), stream,
                                    args->execution.workers);
}
}  // namespace kvaf::cpu_push

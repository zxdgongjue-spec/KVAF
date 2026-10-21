#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "umdk/kernel_loader.h"
namespace kvaf::aicpu_urma {
int32_t LaunchBatchPut(const BatchPutKernelArgs* args, aclrtStream stream) {
  return args ? LaunchUmdkKernel("KvafAicpuBatchPut", args, sizeof(*args), stream) : -1;
}
}  // namespace kvaf::aicpu_urma

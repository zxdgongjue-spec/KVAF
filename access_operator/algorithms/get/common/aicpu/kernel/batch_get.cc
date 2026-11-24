#include "get/gather/urma_aicpu/kernel/gather_read.h"
#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
// V4 separates launch/window sequences and consumes dynamic workspace capacity.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherReadExecutionV4(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return kvaf::aicpu_urma::ExecuteGatherRead(args);
}
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuCpuPushScatterExecutionV4(
    kvaf::cpu_push::KernelArgs* args) {
  return kvaf::cpu_push::ExecuteCpuPushScatter(args);
}
#endif

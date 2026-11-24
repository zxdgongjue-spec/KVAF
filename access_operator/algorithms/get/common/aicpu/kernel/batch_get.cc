#include "get/gather/urma_aicpu/kernel/gather_read.h"
#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
// Gather V5 carries the per-worker READ wave limit; Scatter retains its V4 ABI.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherReadExecutionV5(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return kvaf::aicpu_urma::ExecuteGatherRead(args);
}
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuCpuPushScatterExecutionV4(
    kvaf::cpu_push::KernelArgs* args) {
  return kvaf::cpu_push::ExecuteCpuPushScatter(args);
}
#endif

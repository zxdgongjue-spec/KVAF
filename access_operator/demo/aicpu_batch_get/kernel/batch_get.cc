#include "gather_read.h"
#include "cpu_push_scatter.h"
// Keep the existing gather_read entry for callers using the default path.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherRead(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return kvaf::aicpu_urma::ExecuteGatherRead(args);
}
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuCpuPushScatter(
    kvaf::cpu_push::KernelArgs* args) {
  return kvaf::cpu_push::ExecuteCpuPushScatter(args);
}
#endif

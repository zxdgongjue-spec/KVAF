#include "gather_read.h"
#include "cpu_push_scatter.h"
// Versioned entries reject packages with the previous Host-sequence ABI.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherReadV2(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return kvaf::aicpu_urma::ExecuteGatherRead(args);
}
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherReadRangesV1(
    kvaf::aicpu_urma::UmdkBatchGetRangesArgs* args) {
  return kvaf::aicpu_urma::ExecuteGatherReadRanges(args);
}
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuCpuPushScatterV2(
    kvaf::cpu_push::KernelArgs* args) {
  return kvaf::cpu_push::ExecuteCpuPushScatter(args);
}
#endif

#include "get/gather/urma_aicpu/kernel/gather_read.h"
#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
namespace {
template <class Call>
std::uint32_t ProfileEntry(bool enabled, Call call) {
  const auto entry = enabled ? kvaf::BatchGetProfileNowNs() : 0;
  kvaf::BatchGetWorkerProfile measured{};
  measured.entry_ns = entry;
  kvaf::BatchGetWorkerProfile* destination = nullptr;
  const auto result = call(enabled ? &measured : nullptr, &destination);
  // Execute has returned, including its local destructors and session unlock.
  if (destination) {
    *destination = measured;
    destination->return_ns = kvaf::BatchGetProfileNowNs();
  }
  return result;
}
}  // namespace

// Reject packages lacking the matching entry/return profiling ABI.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherReadExecutionV4(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return ProfileEntry(args && args->gather.profile, [&](auto* profile, auto** destination) {
    return kvaf::aicpu_urma::ExecuteGatherRead(args, profile, destination);
  });
}
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuCpuPushScatterExecutionV4(
    kvaf::cpu_push::KernelArgs* args) {
  return ProfileEntry(args && args->profile, [&](auto* profile, auto** destination) {
    return kvaf::cpu_push::ExecuteCpuPushScatter(args, profile, destination);
  });
}
#endif

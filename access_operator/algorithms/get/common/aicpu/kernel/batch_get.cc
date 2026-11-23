#include "get/gather/urma_aicpu/kernel/gather_read.h"
#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
namespace {
std::uint64_t ProfileCounter() {
#if defined(__aarch64__)
  std::uint64_t value;
  asm volatile("isb; mrs %0, cntvct_el0" : "=r"(value) :: "memory");
  return value;
#else
  return 0;
#endif
}
std::uint64_t ProfileCounterFrequency() {
#if defined(__aarch64__)
  std::uint64_t value;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
  return value;
#else
  return 0;
#endif
}
template <class Call>
std::uint32_t ProfileEntry(bool enabled, Call call) {
  const auto counter = enabled ? ProfileCounter() : 0;
  const auto entry = enabled ? kvaf::BatchGetProfileNowNs() : 0;
  kvaf::BatchGetWorkerProfile measured{};
  measured.entry_ns = entry;
  measured.entry_syscnt = counter;
  measured.syscnt_frequency = enabled ? ProfileCounterFrequency() : 0;
  kvaf::BatchGetWorkerProfile* destination = nullptr;
  const auto result = call(enabled ? &measured : nullptr, &destination);
  // Execute has returned, including its local destructors and session unlock.
  if (destination) {
    *destination = measured;
    destination->return_ns = kvaf::BatchGetProfileNowNs();
    destination->return_syscnt = ProfileCounter();
  }
  return result;
}
}  // namespace

// Reject packages lacking the matching counter profiling ABI.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherReadExecutionV5(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return ProfileEntry(args && args->gather.profile, [&](auto* profile, auto** destination) {
    return kvaf::aicpu_urma::ExecuteGatherRead(args, profile, destination);
  });
}
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuCpuPushScatterExecutionV5(
    kvaf::cpu_push::KernelArgs* args) {
  return ProfileEntry(args && args->profile, [&](auto* profile, auto** destination) {
    return kvaf::cpu_push::ExecuteCpuPushScatter(args, profile, destination);
  });
}
#endif

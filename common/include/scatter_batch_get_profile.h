#pragma once
#include <cstdint>
#include "cpu_push_get_abi.h"

namespace kvaf {
constexpr std::uint64_t kScatterProfileMagic = 0x4b56415343505232ULL;
// A versioned tag avoids interpreting an old diagnostic probe ID as profiling.
constexpr std::uint32_t kScatterProfileFlag = 0x53504631U;
// Stored after the unchanged array of 64-byte business statuses. Each core
// owns one record. CPU durations are copied by core 0 after observing done.
struct alignas(64) ScatterBatchGetCoreProfile {
  std::uint64_t magic{}, sequence{};
  std::uint64_t start_cycle{}, inputs_ready_cycle{}, notify_cycle{}, done_cycle{}, finish_cycle{};
  std::uint64_t requests{}, bytes{};
  std::uint64_t cpu_magic{}, cpu_header_ns{}, cpu_descriptor_ns{}, cpu_write_ns{};
  std::uint64_t cpu_comm_ns{}, cpu_post_ns{}, cpu_cq_wait_ns{};
  // CopyRows intervals include their real MTE completion waits, but exclude
  // waits for CPU progress. They can overlap notify->done in pipeline mode.
  std::uint64_t scatter_cycles{}, ready_wait_cycles{};
  std::uint64_t scatter_start_cycle{}, scatter_finish_cycle{}, scatter_chunks{}, pipeline{};
  std::uint64_t reserved[2]{};
};
static_assert(sizeof(ScatterBatchGetCoreProfile) == 192);
}  // namespace kvaf

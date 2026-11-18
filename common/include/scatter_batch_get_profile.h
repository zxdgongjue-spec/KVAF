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
  // Enqueue/service intervals plus the final MTE drain, excluding Ready/done
  // polling. Async MTE3 may also run during that polling; this is NOT a hardware
  // DMA-busy counter. The start/finish span includes inter-shard waits.
  std::uint64_t scatter_cycles{}, ready_wait_cycles{};
  std::uint64_t scatter_start_cycle{}, scatter_finish_cycle{}, scatter_chunks{};
  // Blocking wait for the final Suffix CQE on core 0.
  std::uint32_t pipeline{}, suffix_cq_blocked_cycles{};
  // Shards submitted while an earlier chunk still has unconsumed shards.
  std::uint32_t cross_chunk_shards{};
  // Keep the profile record and last-enqueue timestamp at their ABI offsets.
  std::uint32_t reserved{};
  // Last payload enqueue/copy, before the final MTE drain in pipeline mode.
  std::uint64_t scatter_enqueue_finish_cycle{};
};
static_assert(sizeof(ScatterBatchGetCoreProfile) == 192);
}  // namespace kvaf

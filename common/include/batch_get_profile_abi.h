#pragma once
#include <cstddef>
#include <cstdint>

namespace kvaf {
// CPU-clock window for one prepared DDR -> HBM SGL WRITE. Only worker 0 reports it.
struct CpuSglWriteSample {
  std::uint64_t elapsed_ns{};
  std::uint32_t data_wrs{};
  std::uint16_t waves{}, tail_cqes{};
};

// Device timestamps share a clock across workers, not with the Host clock.
// AIV SGL reuses these fields for raw cycles; its Host adapter exports cycle
// fields rather than presenting them as nanoseconds. CPU samples stay in ns.
struct alignas(64) BatchGetWorkerProfile {
  std::uint64_t enter_ns{}, exit_ns{};
  std::uint64_t request_ns{};
  // Device-push SGL owns these otherwise unused slots. The tag distinguishes
  // nested Head timings from legacy cache/write stages without changing stride.
  union { std::uint64_t input_cache_ns{}; std::uint64_t head_build_ns; };
  union { std::uint64_t descriptor_cache_ns{}; std::uint64_t head_doorbell_ns; };
  union { std::uint64_t source_write_ns{}; std::uint64_t head_profile_tag; };
  std::uint64_t notify_ns{}, read_ns{}, ack_ns{};
  std::uint64_t read_setup_ns{}, wr_build_ns{}, ready_wait_ns{}, post_ns{}, cq_wait_ns{};
  std::uint64_t bytes{}, spans{}, read_wqes{}, post_calls{}, ready_queries{};
  std::uint64_t scatter_ns{}, output_cache_ns{};
  std::uint64_t scatter_copies{};
  CpuSglWriteSample cpu_sgl_write{};
};
static_assert(sizeof(BatchGetWorkerProfile) == 192);
static_assert(offsetof(BatchGetWorkerProfile, head_build_ns) == 24 &&
              offsetof(BatchGetWorkerProfile, head_doorbell_ns) == 32 &&
              offsetof(BatchGetWorkerProfile, head_profile_tag) == 40);
constexpr std::uint64_t kCpuSglPushHeadProfileV1 = 0x53474c4845414431ULL;  // SGLHEAD1
inline bool HasCpuSglPushHeadProfile(const BatchGetWorkerProfile& p) {
  return p.head_profile_tag == kCpuSglPushHeadProfileV1;
}

}  // namespace kvaf

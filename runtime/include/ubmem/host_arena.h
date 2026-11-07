#pragma once

#include <cstdint>
#include <memory>

#include "runtime.h"

namespace kvaf {

enum class UbmemPageMode { kBase = 1, kThp = 2 };

// Borrowed views of one worker-owned allocation. The CPU and NPU addresses are
// separate fields deliberately: callers must not infer one from the other.
// Only valid_bytes may be addressed by slots; mapped_bytes includes padding.
// granularity is backing-size alignment, not observed physical/NPU page size.
// This descriptor does not extend the allocation lifetime or acquire a lease.
struct UbmemMappedRegion {
  Address host_base{0};
  Address device_base{0};
  std::uint64_t valid_bytes{0};
  std::uint64_t mapped_bytes{0};
  std::uint64_t granularity{0};
  std::uint32_t device_id{0};
};

// CANN >= 9.2 local Host DDR allocator. Base uses normal VMM pages.
// Linux thp uses ordinary aligned Host memory with MADV_HUGEPAGE, then registers
// that memory for AIV access; actual THP coverage is reported, not guaranteed.
// The calling thread must have
// selected device_id through ACL before Allocate, Release, or destruction.
// One allocator owns one arena. Its owner serializes calls and prevents new
// submissions before Release; the final device synchronization then drains
// outstanding accesses before unmapping. It is not a fused-context lease manager.
class UbmemDdrMemoryAllocator final : public DdrMemoryAllocator {
 public:
  explicit UbmemDdrMemoryAllocator(
      std::uint32_t device_id, UbmemPageMode page_mode = UbmemPageMode::kThp, int numa_node = -1);
  ~UbmemDdrMemoryAllocator() override;

  UbmemDdrMemoryAllocator(const UbmemDdrMemoryAllocator&) = delete;
  UbmemDdrMemoryAllocator& operator=(const UbmemDdrMemoryAllocator&) = delete;

  Status Allocate(const RuntimeInitRequest& request,
                  RemoteRegionDesc* remote_region) override;
  Status Release(const RemoteRegionDesc& remote_region) override;
  Status GetMappedRegion(const RemoteRegionDesc& remote_region,
                         UbmemMappedRegion* mapped_region) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<UbmemDdrMemoryAllocator> CreateUbmemDdrMemoryAllocator(
    std::uint32_t device_id, UbmemPageMode page_mode = UbmemPageMode::kThp, int numa_node = -1);

}  // namespace kvaf

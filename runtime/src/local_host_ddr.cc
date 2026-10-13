// SPDX-License-Identifier: Apache-2.0
// Runtime-owned host DDR allocation adapted from the validated local MTP path.
#include "local_host_ddr.h"
#include "acl/acl.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(__linux__)
#include <sys/mman.h>
#endif
namespace kvaf {
namespace {
Status InvalidArgument(std::string message) { return Status::InvalidArgument(std::move(message)); }
Status Internal(std::string message) { return Status::Internal(std::move(message)); }
[[maybe_unused]] constexpr std::size_t kDdrHugePageBytes = 2U * 1024U * 1024U;

enum class DdrPageMode { kAcl, kBase, kThp };

// Sum every VMA intersecting the DDR range: HCCP registration is permitted
// to split it into several VMAs. The mmap allocator applies madvise to the
// exact aligned DDR range, so its VMA accounting excludes untouched padding.
LocalHostDdrStats PageStatsInRange(const void* address, std::size_t bytes) {
  LocalHostDdrStats result;
#if defined(__linux__)
  std::ifstream smaps("/proc/self/smaps");
  if (!smaps || address == nullptr || bytes == 0) return result;
  std::string line;
  bool selected = false;
  const auto target = reinterpret_cast<std::uintptr_t>(address);
  const auto limit = target + bytes;
  while (std::getline(smaps, line)) {
    unsigned long long start = 0;
    unsigned long long end = 0;
    if (std::sscanf(line.c_str(), "%llx-%llx", &start, &end) == 2) {
      selected = start < limit && end > target;
      if (selected) {
        const auto intersection_begin = std::max<std::uint64_t>(start, target);
        const auto intersection_end = std::min<std::uint64_t>(end, limit);
        result.smaps_covered_bytes += intersection_end - intersection_begin;
      }
    } else if (selected && line.rfind("AnonHugePages:", 0) == 0) {
      unsigned long long kib = 0;
      if (std::sscanf(line.c_str(), "AnonHugePages: %llu kB", &kib) == 1) {
        result.anon_huge_bytes += static_cast<std::uint64_t>(kib) * 1024U;
      }
    }
  }
#else
  (void)address;
  (void)bytes;
#endif
  return result;
}

class RuntimeHostDdrAllocator final : public DdrMemoryAllocator {
 public:
  explicit RuntimeHostDdrAllocator(DdrPageMode mode) : mode_(mode) {}

  const char* PageMode() const {
    if (mode_ == DdrPageMode::kThp) return "thp";
    return mode_ == DdrPageMode::kBase ? "base" : "acl";
  }
  LocalHostDdrStats PageStats() const {
    return address_ == nullptr ? LocalHostDdrStats{}
                               : PageStatsInRange(address_, advised_bytes_);
  }

  Status Allocate(const RuntimeInitRequest& request,
                  RemoteRegionDesc* region) override {
    if (region == nullptr || request.ddr_device != DeviceKind::kCpu ||
        request.ddr_bytes == 0 || request.slot_size == 0 ||
        request.ddr_bytes % request.slot_size != 0 ||
        request.ddr_bytes > std::numeric_limits<std::size_t>::max()) {
      return InvalidArgument("invalid ACL Host DDR allocation request");
    }
    if (address_ != nullptr) {
      return Status::Error(StatusCode::kInvalidState,
                           "ACL Host DDR is already allocated");
    }
    if (mode_ == DdrPageMode::kAcl) {
      if (aclrtMallocHost(&address_, request.ddr_bytes) != ACL_SUCCESS) {
        return Status::Error(StatusCode::kOutOfMemory,
                             "aclrtMallocHost failed for Runtime DDR");
      }
      advised_bytes_ = static_cast<std::size_t>(request.ddr_bytes);
    } else {
#if defined(__linux__)
      if (request.ddr_bytes >
          std::numeric_limits<std::size_t>::max() - 2U * kDdrHugePageBytes) {
        return InvalidArgument("THP Runtime DDR size overflows size_t");
      }
      const std::size_t rounded =
          (static_cast<std::size_t>(request.ddr_bytes) +
           kDdrHugePageBytes - 1U) & ~(kDdrHugePageBytes - 1U);
      mapped_bytes_ = rounded + kDdrHugePageBytes;
      mapping_ = mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        mapped_bytes_ = 0;
        return Status::Error(StatusCode::kOutOfMemory,
                             std::string("mmap failed for THP Runtime DDR: ") +
                                 std::strerror(errno));
      }
      const auto base = reinterpret_cast<std::uintptr_t>(mapping_);
      const auto aligned = (base + kDdrHugePageBytes - 1U) &
                           ~(static_cast<std::uintptr_t>(kDdrHugePageBytes) - 1U);
      address_ = reinterpret_cast<void*>(aligned);
      advised_bytes_ = rounded;
      const int advice =
          mode_ == DdrPageMode::kThp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE;
      if (madvise(address_, rounded, advice) != 0) {
        const std::string error = std::strerror(errno);
        (void)munmap(mapping_, mapped_bytes_);
        address_ = nullptr;
        mapping_ = nullptr;
        mapped_bytes_ = 0;
        advised_bytes_ = 0;
        return Internal(std::string("madvise failed for ") + PageMode() +
                        " Runtime DDR: " + error);
      }
      // Touch the entire 2 MiB-aligned range before HCCP registration, so
      // the measurement never silently benchmarks an untouched VMA.
      std::memset(address_, 0, rounded);
      const LocalHostDdrStats page_stats = PageStats();
      std::fprintf(stderr,
                   "[KVAF_AIV] DDR_PAGE_MODE mode=%s requested_bytes=%llu "
                   "rounded_bytes=%zu anon_huge_bytes=%llu "
                   "smaps_covered_bytes=%llu\n",
                   PageMode(), static_cast<unsigned long long>(request.ddr_bytes), rounded,
                   static_cast<unsigned long long>(page_stats.anon_huge_bytes),
                   static_cast<unsigned long long>(page_stats.smaps_covered_bytes));
      if (page_stats.smaps_covered_bytes != rounded ||
          (mode_ == DdrPageMode::kThp && page_stats.anon_huge_bytes == 0) ||
          (mode_ == DdrPageMode::kBase && page_stats.anon_huge_bytes != 0)) {
        (void)munmap(mapping_, mapped_bytes_);
        address_ = nullptr;
        mapping_ = nullptr;
        mapped_bytes_ = 0;
        advised_bytes_ = 0;
        return Internal(std::string(PageMode()) +
                        " Runtime DDR has unexpected smaps page coverage after first touch");
      }
#else
      return InvalidArgument("THP Runtime DDR requires Linux");
#endif
    }
    if (mode_ == DdrPageMode::kAcl) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DDR_PAGE_MODE mode=acl requested_bytes=%llu "
                   "anon_huge_bytes=%llu\n",
                   static_cast<unsigned long long>(request.ddr_bytes),
                   static_cast<unsigned long long>(
                       PageStatsInRange(address_, request.ddr_bytes).anon_huge_bytes));
    }
    *region = RemoteRegionDesc{reinterpret_cast<Address>(address_),
                               request.ddr_bytes, request.slot_size,
                               DeviceKind::kCpu};
    return Status::OK();
  }

  Status Release(const RemoteRegionDesc& region) override {
    if (address_ == nullptr || reinterpret_cast<Address>(address_) != region.base) {
      return Status::Error(StatusCode::kNotFound,
                           "Runtime DDR allocation was not found");
    }
    Status status = Status::OK();
    if (mode_ == DdrPageMode::kAcl) {
      if (aclrtFreeHost(address_) != ACL_SUCCESS) {
        status = Internal("aclrtFreeHost failed for Runtime DDR");
      }
    } else {
#if defined(__linux__)
      if (munmap(mapping_, mapped_bytes_) != 0) {
        status = Internal(std::string("munmap failed for THP Runtime DDR: ") +
                          std::strerror(errno));
      }
#endif
      mapping_ = nullptr;
      mapped_bytes_ = 0;
    }
    address_ = nullptr;
    advised_bytes_ = 0;
    return status;
  }

 private:
  DdrPageMode mode_;
  void* address_{nullptr};
  void* mapping_{nullptr};
  std::size_t mapped_bytes_{0};
  std::size_t advised_bytes_{0};
};


}  // namespace
std::unique_ptr<DdrMemoryAllocator> CreateLocalHostDdrAllocator(const std::string& mode) {
  if (mode != "acl" && mode != "base" && mode != "thp") {
    throw std::invalid_argument("local DDR page mode must be acl, base or thp");
  }
  return std::make_unique<RuntimeHostDdrAllocator>(
      mode == "acl" ? DdrPageMode::kAcl : mode == "base" ? DdrPageMode::kBase : DdrPageMode::kThp);
}
LocalHostDdrStats LocalHostDdrPageStats(const DdrMemoryAllocator& allocator) {
  const auto* local = dynamic_cast<const RuntimeHostDdrAllocator*>(&allocator);
  return local == nullptr ? LocalHostDdrStats{} : local->PageStats();
}
}  // namespace kvaf

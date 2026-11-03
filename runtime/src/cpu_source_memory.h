#pragma once

// CPU source allocation only; registration still uses the requested logical
// length. Release explicitly after URMA teardown, retaining memory on DMA errors.
#include "cpu_runtime.h"
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace kvaf::cpu_memory {
inline const char* PageModeName(CpuMemoryPageMode mode) {
  switch (mode) {
    case CpuMemoryPageMode::kDefault: return "default";
    case CpuMemoryPageMode::kTransparentHugePage: return "thp";
    case CpuMemoryPageMode::kHugeTlb: return "hugetlb";
  }
  return "unknown";
}

inline Status ValidateOptions(CpuMemoryPageMode mode, std::uint32_t page_kb) {
  if (mode != CpuMemoryPageMode::kDefault && mode != CpuMemoryPageMode::kTransparentHugePage &&
      mode != CpuMemoryPageMode::kHugeTlb)
    return Status::InvalidArgument("unknown CPU memory page mode");
  if (page_kb && (mode != CpuMemoryPageMode::kHugeTlb || (page_kb & (page_kb - 1))))
    return Status::InvalidArgument("hugetlb_page_kb requires hugetlb mode and a power-of-two KiB size");
  return Status::OK();
}

inline std::uint64_t DefaultHugePageBytes() {
  std::ifstream input("/proc/meminfo");
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("Hugepagesize:", 0) != 0) continue;
    std::istringstream value(line.substr(13));
    std::uint64_t kb = 0;
    std::string unit;
    if (value >> kb >> unit && unit == "kB" && kb <= UINT64_MAX / 1024)
      return kb * 1024;
  }
  return 0;
}

struct SourceMemory {
  void* data{};
  std::size_t allocation_bytes{};
  // Nonzero only for explicit HugeTLB; THP advice is not proof of backing size.
  std::size_t hugetlb_page_bytes{};

  SourceMemory() = default;
  SourceMemory(const SourceMemory&) = delete;
  SourceMemory& operator=(const SourceMemory&) = delete;

  Status Allocate(std::uint64_t bytes, CpuMemoryPageMode mode, std::uint32_t page_kb) {
    if (data) return Status::InvalidState("CPU source memory already allocated");
    auto valid = ValidateOptions(mode, page_kb);
    if (!valid.ok()) return valid;
    if (!bytes || bytes > std::numeric_limits<std::size_t>::max())
      return Status::InvalidArgument("invalid CPU source allocation length");
    if (mode == CpuMemoryPageMode::kHugeTlb) {
#if defined(MAP_HUGETLB) && defined(MAP_HUGE_SHIFT)
      const auto page = page_kb ? std::uint64_t(page_kb) * 1024 : DefaultHugePageBytes();
      const auto base_page = sysconf(_SC_PAGESIZE);
      if (!page || (page & (page - 1)) || base_page <= 0 || page < std::uint64_t(base_page) ||
          page > std::numeric_limits<std::size_t>::max() ||
          bytes > std::numeric_limits<std::size_t>::max() - (page - 1))
        return Status::InvalidArgument("invalid/unknown HugeTLB page size or rounded length overflow");
      const auto rounded = static_cast<std::size_t>((bytes + page - 1) & ~(page - 1));
      unsigned flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
      if (page_kb) {
        unsigned shift = 0;
        for (auto p = page; p > 1; p >>= 1) ++shift;
        flags |= shift << MAP_HUGE_SHIFT;
      }
      // Keep reservations enabled. No ordinary-page fallback or MAP_NORESERVE.
      void* result = mmap(nullptr, rounded, PROT_READ | PROT_WRITE, static_cast<int>(flags), -1, 0);
      if (result == MAP_FAILED) {
        const int error = errno;
        return Status::OutOfMemory("mmap MAP_HUGETLB failed: errno=" + std::to_string(error) +
            " (" + std::strerror(error) + "), requested_bytes=" + std::to_string(bytes) +
            " mapped_bytes=" + std::to_string(rounded) + " page_bytes=" + std::to_string(page) +
            "; check the HugeTLB pool on the selected NUMA node, permissions and cgroup limits; no fallback");
      }
      data = result;
      allocation_bytes = rounded;
      hugetlb_page_bytes = static_cast<std::size_t>(page);
#else
      return Status::NotImplemented("MAP_HUGETLB/MAP_HUGE_SHIFT unavailable on this build");
#endif
    } else {
      const auto alignment = mode == CpuMemoryPageMode::kTransparentHugePage ? 2U * 1024U * 1024U : 4096U;
      const int error = posix_memalign(&data, alignment, static_cast<std::size_t>(bytes));
      if (error) {
        data = nullptr;
        return Status::OutOfMemory("posix_memalign failed: status=" + std::to_string(error));
      }
      allocation_bytes = static_cast<std::size_t>(bytes);
      if (mode == CpuMemoryPageMode::kTransparentHugePage &&
          madvise(data, allocation_bytes, MADV_HUGEPAGE)) {
        const int error = errno;
        Release();
        return Status::Internal("madvise MADV_HUGEPAGE failed: errno=" + std::to_string(error));
      }
    }
    // Fault before URMA registration under the service's NUMA memory policy.
    // Preserve the existing lazy allocation behavior of the default mode.
    if (mode != CpuMemoryPageMode::kDefault) std::memset(data, 0, static_cast<std::size_t>(bytes));
    return Status::OK();
  }

  Status Release() {
    if (!data) return Status::OK();
    if (hugetlb_page_bytes) {
      if (munmap(data, allocation_bytes))
        return Status::Internal("munmap HugeTLB failed: errno=" + std::to_string(errno));
    } else {
      std::free(data);
    }
    data = nullptr;
    allocation_bytes = hugetlb_page_bytes = 0;
    return Status::OK();
  }
};
}  // namespace kvaf::cpu_memory

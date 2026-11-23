#include "memory/cpu_ddr_allocation.h"

#include <sys/mman.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace kvaf {
namespace {
constexpr std::size_t kBasePageBytes = 4096U;
constexpr int kHugePage2MShift = 21;
constexpr int kHugePage1GShift = 30;
constexpr std::size_t kHugePage2MBytes = std::size_t{1} << kHugePage2MShift;

Status AllocationError(const char* operation, CpuMemoryPageMode mode, std::size_t bytes,
                       int error, std::size_t backing_bytes) {
  return Status::Internal(std::string(operation) + " failed; page_mode=" +
                          CpuMemoryPageModeName(mode) + " bytes=" + std::to_string(bytes) +
                          " backing_bytes=" + std::to_string(backing_bytes) +
                          " errno=" + std::to_string(error) + " (" + std::strerror(error) + ")");
}
}  // namespace

const char* CpuMemoryPageModeName(CpuMemoryPageMode mode) {
  switch (mode) {
    case CpuMemoryPageMode::kDefault:
      return "default";
    case CpuMemoryPageMode::kTransparentHugePage:
      return "thp";
    case CpuMemoryPageMode::kHugeTlb2M:
      return "hugetlb-2m";
    case CpuMemoryPageMode::kHugeTlb1G:
      return "hugetlb-1g";
  }
  return "unknown";
}

Status AllocateCpuDdr(CpuMemoryPageMode mode, std::size_t bytes, void*& memory,
                      std::size_t& mapped_bytes) {
  if (memory != nullptr || bytes == 0U)
    return Status::InvalidArgument("invalid CPU DDR allocation size or state");
  if (mode == CpuMemoryPageMode::kHugeTlb2M || mode == CpuMemoryPageMode::kHugeTlb1G) {
    const int page_shift = mode == CpuMemoryPageMode::kHugeTlb2M ? kHugePage2MShift : kHugePage1GShift;
    const std::size_t page_bytes = std::size_t{1} << page_shift;
    const std::size_t padding = (page_bytes - bytes % page_bytes) % page_bytes;
    if (bytes > std::numeric_limits<std::size_t>::max() - padding)
      return Status::InvalidArgument("HugeTLB DDR mapping size overflow");
    const std::size_t backing_bytes = bytes + padding;
    const int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (page_shift << MAP_HUGE_SHIFT);
    void* address = mmap(nullptr, backing_bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (address == MAP_FAILED)
      return AllocationError("mmap_hugetlb", mode, bytes, errno, backing_bytes);
    memory = address;
    mapped_bytes = backing_bytes;
#ifdef MADV_POPULATE_WRITE
    // Older kernels do not support the advice. Touch the requested range on
    // those kernels while the CPU service's NUMA policy is active.
    if (madvise(nullptr, 0U, MADV_POPULATE_WRITE) == 0) {
      if (madvise(memory, mapped_bytes, MADV_POPULATE_WRITE) != 0)
        return AllocationError("prefault_hugetlb", mode, bytes, errno, mapped_bytes);
      return Status::OK();
    }
#endif
    std::memset(memory, 0, bytes);
    return Status::OK();
  }

  const bool thp = mode == CpuMemoryPageMode::kTransparentHugePage;
  const int error = posix_memalign(&memory, thp ? kHugePage2MBytes : kBasePageBytes, bytes);
  if (error != 0) return AllocationError("posix_memalign", mode, bytes, error, bytes);
  if (thp) {
    if (madvise(memory, bytes, MADV_HUGEPAGE) != 0)
      return AllocationError("madvise_hugepage", mode, bytes, errno, bytes);
    std::memset(memory, 0, bytes);
  }
  return Status::OK();
}

Status ReleaseCpuDdr(void*& memory, std::size_t& mapped_bytes) {
  if (memory == nullptr) return Status::OK();
  if (mapped_bytes != 0U) {
    if (munmap(memory, mapped_bytes) != 0) {
      const int error = errno;
      return Status::Internal("munmap_hugetlb failed; backing_bytes=" + std::to_string(mapped_bytes) +
                              " errno=" + std::to_string(error) + " (" + std::strerror(error) + ")");
    }
  } else {
    std::free(memory);
  }
  memory = nullptr;
  mapped_bytes = 0U;
  return Status::OK();
}

}  // namespace kvaf

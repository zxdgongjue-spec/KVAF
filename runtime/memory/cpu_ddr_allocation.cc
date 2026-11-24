#include "memory/cpu_ddr_allocation.h"

#include <sys/mman.h>

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace kvaf {
namespace {
constexpr int kBasePageShift = 12;
constexpr int kBinaryUnitShift = 10;
constexpr std::size_t kBasePageBytes = std::size_t{1} << kBasePageShift;
constexpr std::size_t kHugePage2MBytes = std::size_t{1} << static_cast<int>(CpuMemoryPageMode::kHugeTlb2M);
constexpr std::string_view kHugeTlbPrefix = "hugetlb-";
constexpr std::string_view kPageSizeUnits = "kmgtpe";

bool IsHugeTlbShift(int shift) {
  return shift >= kBasePageShift && shift < std::numeric_limits<std::size_t>::digits &&
         shift <= MAP_HUGE_MASK;
}

Status AllocationError(const char* operation, CpuMemoryPageMode mode, std::size_t bytes,
                       int error, std::size_t backing_bytes) {
  return Status::Internal(std::string(operation) + " failed; page_mode=" +
                          CpuMemoryPageModeName(mode) + " bytes=" + std::to_string(bytes) +
                          " backing_bytes=" + std::to_string(backing_bytes) +
                          " errno=" + std::to_string(error) + " (" + std::strerror(error) + ")");
}
}  // namespace

bool ParseCpuMemoryPageMode(std::string_view text, CpuMemoryPageMode& mode) {
  if (text == "default") {
    mode = CpuMemoryPageMode::kDefault;
    return true;
  }
  if (text == "thp") {
    mode = CpuMemoryPageMode::kTransparentHugePage;
    return true;
  }
  if (text.size() < kHugeTlbPrefix.size() + 2 || text.substr(0, kHugeTlbPrefix.size()) != kHugeTlbPrefix)
    return false;
  const auto unit = kPageSizeUnits.find(text.back());
  if (unit == std::string_view::npos) return false;

  const char* first = text.data() + kHugeTlbPrefix.size();
  const char* last = text.data() + text.size() - 1;
  std::size_t count = 0;
  const auto parsed = std::from_chars(first, last, count);
  if (parsed.ec != std::errc{} || parsed.ptr != last || count == 0 || (count & (count - 1)) != 0)
    return false;

  int page_shift = static_cast<int>(unit + 1) * kBinaryUnitShift;
  for (; count > 1; count >>= 1) ++page_shift;
  if (!IsHugeTlbShift(page_shift)) return false;
  mode = static_cast<CpuMemoryPageMode>(page_shift);
  return true;
}

std::string CpuMemoryPageModeName(CpuMemoryPageMode mode) {
  if (mode == CpuMemoryPageMode::kDefault) return "default";
  if (mode == CpuMemoryPageMode::kTransparentHugePage) return "thp";
  const int page_shift = static_cast<int>(mode);
  if (!IsHugeTlbShift(page_shift)) return "unknown";
  const auto count = 1U << (page_shift % kBinaryUnitShift);
  const auto unit = kPageSizeUnits[page_shift / kBinaryUnitShift - 1];
  return std::string(kHugeTlbPrefix) + std::to_string(count) + unit;
}

std::size_t CpuDdrBackingAlignment(CpuMemoryPageMode mode) {
  if (mode == CpuMemoryPageMode::kDefault || mode == CpuMemoryPageMode::kTransparentHugePage)
    return 1U;
  const int page_shift = static_cast<int>(mode);
  return IsHugeTlbShift(page_shift) ? std::size_t{1} << page_shift : 0U;
}

Status AllocateCpuDdr(CpuMemoryPageMode mode, std::size_t bytes, void*& memory,
                      std::size_t& mapped_bytes) {
  if (memory != nullptr || bytes == 0U)
    return Status::InvalidArgument("invalid CPU DDR allocation size or state");
  const bool thp = mode == CpuMemoryPageMode::kTransparentHugePage;
  if (mode != CpuMemoryPageMode::kDefault && !thp) {
    const int page_shift = static_cast<int>(mode);
    const auto page_bytes = CpuDdrBackingAlignment(mode);
    if (!page_bytes) return Status::InvalidArgument("invalid CPU DDR page mode");
    const std::size_t padding = (page_bytes - bytes % page_bytes) % page_bytes;
    if (bytes > std::numeric_limits<std::size_t>::max() - padding)
      return Status::InvalidArgument("HugeTLB DDR mapping size overflow");
    const std::size_t backing_bytes = bytes + padding;
    const auto huge_flags = static_cast<unsigned int>(page_shift) << MAP_HUGE_SHIFT;
    const int flags = static_cast<int>(huge_flags | MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB);
    void* address = mmap(nullptr, backing_bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (address == MAP_FAILED) {
      const int error = errno;
      const auto failure = AllocationError("mmap_hugetlb", mode, bytes, error, backing_bytes);
      if (error == EINVAL)
        return Status::Internal(failure.message() + "; check supported sizes in /sys/kernel/mm/hugepages");
      return failure;
    }
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

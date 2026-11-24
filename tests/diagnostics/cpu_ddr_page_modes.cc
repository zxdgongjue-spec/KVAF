#include "memory/cpu_ddr_allocation.h"

#include <sys/mman.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

using kvaf::CpuMemoryPageMode;

constexpr std::size_t kTwoMiB = 2ULL << 20;
constexpr std::size_t kRequestBytes = kTwoMiB + 4096;

void Check(bool passed, const char* detail) {
  if (passed) return;
  std::fprintf(stderr, "%s\n", detail);
  std::abort();
}

struct MmapCalls {
  char memory[kRequestBytes]{};
  std::size_t allocated_bytes{};
  std::size_t released_bytes{};
  int flags{};
  int allocation_error{};
  int release_error{};
  int prefault_error{};
} page_calls;

void CheckLinuxPageModes() {
  void* memory = nullptr;
  std::size_t mapped_bytes = 0U;
  const struct {
    const char* name;
    unsigned int shift;
    std::size_t page_count;
  } cases[] = {{"hugetlb-64k", 16, 33}, {"hugetlb-2m", 21, 2}, {"hugetlb-16m", 24, 1},
               {"hugetlb-32m", 25, 1}, {"hugetlb-512m", 29, 1}, {"hugetlb-1g", 30, 1},
               {"hugetlb-16g", 34, 1}};
  for (const auto& test : cases) {
    CpuMemoryPageMode mode{};
    Check(kvaf::ParseCpuMemoryPageMode(test.name, mode), "valid HugeTLB size rejected");
    Check(kvaf::CpuMemoryPageModeName(mode) == test.name, "incorrect HugeTLB mode name");
    Check(kvaf::CpuDdrBackingAlignment(mode) == (std::size_t{1} << test.shift),
          "workspace budgets must use the allocator's HugeTLB alignment");
    const auto expected = test.page_count * (std::size_t{1} << test.shift);
    Check(kvaf::AllocateCpuDdr(mode, kRequestBytes, memory, mapped_bytes).ok(), "HugeTLB allocation failed");
    Check(memory == page_calls.memory && mapped_bytes == expected && page_calls.allocated_bytes == expected,
          "HugeTLB backing length was not rounded");
    Check(static_cast<unsigned int>(page_calls.flags) ==
              (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (test.shift << MAP_HUGE_SHIFT)),
          "wrong mmap flags for explicit HugeTLB mode");
    page_calls.release_error = EBUSY;
    Check(!kvaf::ReleaseCpuDdr(memory, mapped_bytes).ok() && memory && mapped_bytes == expected,
          "failed munmap must preserve ownership for retry");
    page_calls.release_error = 0;
    Check(kvaf::ReleaseCpuDdr(memory, mapped_bytes).ok() && !memory && !mapped_bytes &&
          page_calls.released_bytes == expected, "munmap must release the full backing length");
  }
  for (const char* invalid : {"hugetlb-", "hugetlb-0m", "hugetlb-3m", "hugetlb-16mb", "hugetlb--2m",
                              "hugetlb-2k", "hugetlb-16e", "hugetlb-18446744073709551616k"}) {
    CpuMemoryPageMode mode = CpuMemoryPageMode::kDefault;
    Check(!kvaf::ParseCpuMemoryPageMode(invalid, mode) && mode == CpuMemoryPageMode::kDefault,
          "invalid HugeTLB size must be rejected");
  }
  CpuMemoryPageMode mode{};
  Check(kvaf::ParseCpuMemoryPageMode("hugetlb-1024m", mode) && mode == CpuMemoryPageMode::kHugeTlb1G,
        "equivalent HugeTLB sizes should produce the same page mode");

  Check(!kvaf::AllocateCpuDdr(CpuMemoryPageMode::kHugeTlb2M, std::numeric_limits<std::size_t>::max(),
                             memory, mapped_bytes).ok(), "HugeTLB rounding overflow must fail");
  page_calls.allocation_error = ENOMEM;
  const auto failure = kvaf::AllocateCpuDdr(CpuMemoryPageMode::kHugeTlb2M, kRequestBytes, memory, mapped_bytes);
  Check(!failure.ok() && failure.message().find("mmap_hugetlb") != std::string::npos && !memory,
        "failed HugeTLB allocation must report the error without retrying with ordinary pages");
  page_calls.allocation_error = EINVAL;
  const auto unsupported = kvaf::AllocateCpuDdr(mode, kRequestBytes, memory, mapped_bytes);
  Check(!unsupported.ok() && unsupported.message().find("/sys/kernel/mm/hugepages") != std::string::npos &&
        !memory && !mapped_bytes, "invalid HugeTLB mmap must report where to check supported sizes");
  page_calls.allocation_error = 0;
#ifdef MADV_POPULATE_WRITE
  page_calls.prefault_error = EFAULT;
  const auto prefault = kvaf::AllocateCpuDdr(CpuMemoryPageMode::kHugeTlb2M, kRequestBytes, memory, mapped_bytes);
  Check(!prefault.ok() && prefault.message().find("prefault_hugetlb") != std::string::npos &&
        memory && mapped_bytes == 2 * kTwoMiB, "failed prefault must retain ownership for cleanup");
  Check(kvaf::ReleaseCpuDdr(memory, mapped_bytes).ok(), "failed prefault cleanup failed");
#endif
  for (auto mode : {CpuMemoryPageMode::kDefault, CpuMemoryPageMode::kTransparentHugePage}) {
    Check(kvaf::CpuDdrBackingAlignment(mode) == 1U, "heap/THP budgets use logical bytes");
    Check(kvaf::AllocateCpuDdr(mode, 4096, memory, mapped_bytes).ok() && memory && !mapped_bytes,
          "ordinary or transparent huge-page allocation failed");
    Check(kvaf::ReleaseCpuDdr(memory, mapped_bytes).ok() && !memory, "heap allocation release failed");
  }
  Check(kvaf::CpuDdrBackingAlignment(static_cast<CpuMemoryPageMode>(2)) == 0U,
        "invalid page modes must not produce an alignment");
}

}  // namespace

extern "C" void* __wrap_mmap(void*, std::size_t bytes, int, int flags, int, off_t) {
  page_calls.allocated_bytes = bytes;
  page_calls.flags = flags;
  if (page_calls.allocation_error) {
    errno = page_calls.allocation_error;
    return MAP_FAILED;
  }
  return page_calls.memory;
}

extern "C" int __wrap_madvise(void* address, std::size_t, int advice) {
#ifdef MADV_POPULATE_WRITE
  if (advice == MADV_POPULATE_WRITE && address && page_calls.prefault_error) {
    errno = page_calls.prefault_error;
    return -1;
  }
#else
  (void)address;
  (void)advice;
#endif
  return 0;
}

extern "C" int __wrap_munmap(void* address, std::size_t bytes) {
  Check(address == page_calls.memory, "munmap was called for the wrong address");
  if (page_calls.release_error) {
    errno = page_calls.release_error;
    return -1;
  }
  page_calls.released_bytes = bytes;
  return 0;
}

int main() {
  CheckLinuxPageModes();
}

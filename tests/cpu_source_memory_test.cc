// Fault injection exercises the production allocator without a HugeTLB pool or
// URMA hardware. mmap/munmap interception checks size, flags and strict failure.
#include "cpu_source_memory.h"
#include "cpu_write_diagnostics.h"
#include <cassert>
#include <iostream>

namespace {
std::size_t mapped_bytes{}, unmapped_bytes{};
unsigned mapped_flags{}, mmap_calls{}, normal_calls{}, advise_calls{};
bool fail_mmap{}, fail_munmap{}, fail_advice{};
constexpr std::size_t kPage = 2U * 1024U * 1024U;
}

extern "C" int __real_posix_memalign(void**, std::size_t, std::size_t);
extern "C" int __wrap_posix_memalign(void** result, std::size_t alignment, std::size_t bytes) {
  ++normal_calls;
  return __real_posix_memalign(result, alignment, bytes);
}
extern "C" void* __wrap_mmap(void* address, std::size_t bytes, int protection, int flags, int fd, off_t offset) {
  ++mmap_calls;
  assert(!address && fd == -1 && offset == 0 && protection == (PROT_READ | PROT_WRITE));
  mapped_bytes = bytes;
  mapped_flags = static_cast<unsigned>(flags);
  if (fail_mmap) { errno = ENOMEM; return MAP_FAILED; }
  void* data = nullptr;
  assert(!__real_posix_memalign(&data, kPage, bytes));
  std::memset(data, 0xa5, bytes);
  return data;
}
extern "C" int __wrap_munmap(void* address, std::size_t bytes) {
  unmapped_bytes = bytes;
  if (fail_munmap) { errno = EINVAL; return -1; }
  std::free(address);
  return 0;
}
extern "C" int __wrap_madvise(void*, std::size_t, int advice) {
  ++advise_calls;
  assert(advice == MADV_HUGEPAGE);
  if (fail_advice) { errno = EINVAL; return -1; }
  return 0;
}

int main() {
  using kvaf::CpuMemoryPageMode;
  using kvaf::cpu_memory::SourceMemory;
  SourceMemory memory;
  const auto logical = kPage + 656;
  assert(memory.Allocate(logical, CpuMemoryPageMode::kHugeTlb, 2048).ok());
  assert(memory.data && memory.allocation_bytes == 2 * kPage && memory.hugetlb_page_bytes == kPage);
  assert(mapped_bytes == 2 * kPage && normal_calls == 0 && advise_calls == 0);
  assert(mapped_flags & MAP_HUGETLB);
  assert((mapped_flags >> MAP_HUGE_SHIFT & MAP_HUGE_MASK) == 21);
  assert(!(mapped_flags & MAP_NORESERVE));
  auto* data = static_cast<unsigned char*>(memory.data);
  for (std::size_t i = 0; i < logical; ++i) assert(data[i] == 0);
  assert(data[logical] == 0xa5);  // Logical source length is not inflated.
  assert(!memory.Allocate(64, CpuMemoryPageMode::kDefault, 0).ok());
  fail_munmap = true;
  assert(!memory.Release().ok() && memory.data);  // Retain ownership for retry.
  fail_munmap = false;
  assert(memory.Release().ok() && !memory.data && unmapped_bytes == 2 * kPage);
  assert(memory.Release().ok());

  const auto calls = mmap_calls;
  assert(!memory.Allocate(UINT64_MAX, CpuMemoryPageMode::kHugeTlb, 2048).ok());
  assert(!memory.Allocate(64, CpuMemoryPageMode::kHugeTlb, 3).ok());
  assert(!memory.Allocate(64, CpuMemoryPageMode::kTransparentHugePage, 2048).ok());
  assert(!memory.Allocate(0, CpuMemoryPageMode::kHugeTlb, 2048).ok());
  assert(mmap_calls == calls);
  fail_mmap = true;
  const auto failed = memory.Allocate(656, CpuMemoryPageMode::kHugeTlb, 2048);
  assert(!failed.ok() && failed.message().find("no fallback") != std::string::npos);
  assert(!memory.data && !memory.allocation_bytes && normal_calls == 0);
  fail_mmap = false;

  // Default HugeTLB size is discovered, not assumed to be 2 MiB on ARM.
  const auto default_page = kvaf::cpu_memory::DefaultHugePageBytes();
  if (default_page) {
    fail_mmap = true;  // Inspect flags without allocating a potentially gigantic page.
    assert(!memory.Allocate(656, CpuMemoryPageMode::kHugeTlb, 0).ok());
    assert(mapped_bytes == default_page && !(mapped_flags >> MAP_HUGE_SHIFT & MAP_HUGE_MASK));
    fail_mmap = false;
  }
  assert(memory.Allocate(656, CpuMemoryPageMode::kDefault, 0).ok());
  assert(!memory.hugetlb_page_bytes && memory.allocation_bytes == 656);
  assert(memory.Release().ok());
  fail_advice = true;
  assert(!memory.Allocate(656, CpuMemoryPageMode::kTransparentHugePage, 0).ok() && !memory.data);
  fail_advice = false;
  assert(memory.Allocate(656, CpuMemoryPageMode::kTransparentHugePage, 0).ok());
  data = static_cast<unsigned char*>(memory.data);
  assert(data[0] == 0 && data[655] == 0 && !memory.hugetlb_page_bytes);
  assert(memory.Release().ok());

  std::istringstream smaps(
      "100000-200000 rw-p 00000000 00:00 0\nAnonHugePages: 1024 kB\nVmFlags: rd wr hg\n"
      "200000-600000 rw-p 00000000 00:0f 1 /anon_hugepage (deleted)\n"
      "KernelPageSize: 2048 kB\nMMUPageSize: 2048 kB\nAnonHugePages: 0 kB\n"
      "Private_Hugetlb: 4096 kB\nShared_Hugetlb: 0 kB\nVmFlags: rd wr ht\n"
      "600000-800000 rw-p 00000000 00:00 0\nAnonHugePages: 2048 kB\n");
  const auto info = kvaf::cpu_push::diagnostic::ReadPageInfo(smaps, 0x200100);
  assert(info.hugetlb_vma && info.private_hugetlb == 2 * kPage && !info.anon_huge);
  assert(info.vma_bytes == 2 * kPage && info.kernel_page == kPage && info.mmu_page == kPage);
  const auto json = kvaf::cpu_push::diagnostic::Environment(info, -1, {}, {});
  assert(json.find("\"vma_private_hugetlb_bytes\":4194304") != std::string::npos);
  assert(json.find("\"vma_hugetlb\":1") != std::string::npos);
  std::cout << "PASS: CPU source page modes, strict HugeTLB allocation/release and smaps evidence\n";
}

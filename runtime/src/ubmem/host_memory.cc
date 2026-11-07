#include "host_memory.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace kvaf::ubmem_host {

ThpCoverage ParseThpCoverage(std::istream& input, std::uintptr_t address,
                            std::size_t bytes) {
  ThpCoverage result;
  if (!input.good() || bytes == 0 ||
      address > std::numeric_limits<std::uintptr_t>::max() - bytes) return result;
  result.available = true;
  const auto end = address + bytes;
  unsigned long long begin = 0, limit = 0, huge_kib = 0;
  bool have_vma = false, have_huge = false;
  auto record = [&] {
    if (!have_vma || !have_huge || begin < address || limit > end || limit <= begin) return;
    const auto length = limit - begin;
    if (huge_kib > length / 1024 || result.measured_bytes > bytes - length) return;
    result.measured_bytes += length;
    result.anon_huge_bytes += huge_kib * 1024;
  };
  std::string line;
  while (std::getline(input, line)) {
    unsigned long long next_begin = 0, next_limit = 0;
    char permissions[5]{};
    if (std::sscanf(line.c_str(), "%llx-%llx %4s", &next_begin, &next_limit, permissions) == 3) {
      record();
      begin = next_begin;
      limit = next_limit;
      have_vma = true;
      have_huge = false;
      huge_kib = 0;
    } else if (line.compare(0, 14, "AnonHugePages:") == 0) {
      have_huge = std::sscanf(line.c_str(), "AnonHugePages: %llu kB", &huge_kib) == 1;
    }
  }
  record();
  result.exact = !input.bad() && result.measured_bytes == bytes;
  return result;
}

// Tests replace only these platform operations, leaving the parser above real.
#if !defined(KVAF_TEST_UBMEM_HOST_MEMORY)
bool ThpSupported() {
#if defined(__linux__)
  return true;
#else
  return false;
#endif
}

Status AllocateThp(void** address, std::size_t bytes) {
  *address = nullptr;
  if (!ThpSupported()) return Status::NotImplemented("UBmem thp requires Linux");
  const int code = posix_memalign(address, kThpAlignment, bytes);
  if (code != 0) {
    return Status::OutOfMemory("UBmem THP posix_memalign failed, rc=" + std::to_string(code));
  }
  return Status::OK();
}

Status AdviseThp(void* address, std::size_t bytes) {
#if defined(__linux__)
  if (madvise(address, bytes, MADV_HUGEPAGE) != 0) {
    return Status::Internal("UBmem MADV_HUGEPAGE failed, errno=" + std::to_string(errno));
  }
  return Status::OK();
#else
  (void)address;
  (void)bytes;
  return Status::NotImplemented("UBmem thp requires Linux");
#endif
}

Status BindNuma(void* address, std::size_t bytes, int node) {
  if (node < -1 || node > 4095) return Status::InvalidArgument("UBmem NUMA node must be -1 or 0..4095");
#if defined(__linux__)
  constexpr std::size_t bits = sizeof(unsigned long) * 8;
  // Linux get_nodes() consumes maxnode - 1 bits. Reserve a trailing zero
  // bit, including at word boundaries, and pass the allocated bit capacity.
  std::vector<unsigned long> mask(node < 0 ? 0 : (static_cast<std::size_t>(node) + 1) / bits + 1, 0);
  if (node >= 0) mask[node / bits] |= 1UL << (node % bits);
  // Range-local policy before first touch. No process-wide policy changes or
  // migration of previously registered arenas. Bind only this Host DDR range.
  if (syscall(SYS_mbind, address, bytes, node < 0 ? MPOL_DEFAULT : MPOL_BIND,
              node < 0 ? nullptr : mask.data(),
              static_cast<unsigned long>(mask.size() * bits), 0UL) != 0) {
    return Status::Internal("UBmem mbind node=" + std::to_string(node) +
                            " failed, errno=" + std::to_string(errno));
  }
  return Status::OK();
#else
  (void)address;
  (void)bytes;
  if (node == -1) return Status::OK();
  return Status::NotImplemented("UBmem NUMA binding requires Linux");
#endif
}

void Prefault(void* address, std::size_t bytes) { std::memset(address, 0, bytes); }
void Free(void* address) { std::free(address); }

ThpCoverage ReadThpCoverage(void* address, std::size_t bytes) {
  try {
    std::ifstream input("/proc/self/smaps");
    return ParseThpCoverage(input, reinterpret_cast<std::uintptr_t>(address), bytes);
  } catch (...) {
    // Optional evidence must neither fail allocation nor leak an owned mapping.
    return {};
  }
}
#endif

}  // namespace kvaf::ubmem_host

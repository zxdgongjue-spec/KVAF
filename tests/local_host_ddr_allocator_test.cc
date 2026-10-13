// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "acl/acl.h"
#include "local_host_ddr.h"

namespace {
#if defined(__linux__)
constexpr std::size_t kHugePage = 2U * 1024U * 1024U;
#endif
struct FakeAcl {
  int malloc_calls{0};
  int free_calls{0};
  bool fail_malloc{false};
  std::unordered_set<void*> live;
} driver;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void RequireEmptyStats(const kvaf::DdrMemoryAllocator& allocator) {
  const auto stats = kvaf::LocalHostDdrPageStats(allocator);
  Require(stats.anon_huge_bytes == 0 && stats.smaps_covered_bytes == 0,
          "unallocated allocator must have empty page evidence");
}

void TestRequestValidation() {
  for (const std::string mode : {"", "THP", "auto", "remote"}) {
    bool rejected = false;
    try {
      (void)kvaf::CreateLocalHostDdrAllocator(mode);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "invalid page mode must be explicit: " + mode);
  }
  for (const std::string mode : {"acl", "base", "thp"}) {
    auto allocator = kvaf::CreateLocalHostDdrAllocator(mode);
    RequireEmptyStats(*allocator);
    kvaf::RemoteRegionDesc region{123, 456, 789, kvaf::DeviceKind::kNpu};
    const kvaf::RuntimeInitRequest invalid[] = {
        {0, 256, kvaf::DeviceKind::kCpu},
        {4096, 0, kvaf::DeviceKind::kCpu},
        {4097, 256, kvaf::DeviceKind::kCpu},
        {4096, 256, kvaf::DeviceKind::kNpu},
    };
    const int calls_before = driver.malloc_calls;
    for (const auto& request : invalid) {
      Require(allocator->Allocate(request, &region).code() ==
                  kvaf::StatusCode::kInvalidArgument,
              "reject invalid DDR request before allocation");
      Require(region.base == 123 && region.length == 456 && region.slot_size == 789 &&
                  region.device == kvaf::DeviceKind::kNpu,
              "rejected request must not publish a partial region");
    }
    Require(allocator->Allocate({4096, 256, kvaf::DeviceKind::kCpu}, nullptr).code() ==
                kvaf::StatusCode::kInvalidArgument,
            "reject null region output");
    Require(driver.malloc_calls == calls_before, "invalid requests must not call ACL");
    Require(allocator->Release(region).code() == kvaf::StatusCode::kNotFound,
            "reject release before allocation");
    RequireEmptyStats(*allocator);
  }
}

void TestAclAllocationAndRelease() {
  auto allocator = kvaf::CreateLocalHostDdrAllocator("acl");
  kvaf::RemoteRegionDesc region;
  const int malloc_before = driver.malloc_calls;
  const int free_before = driver.free_calls;
  driver.fail_malloc = true;
  Require(allocator->Allocate({4096, 256, kvaf::DeviceKind::kCpu}, &region).code() ==
              kvaf::StatusCode::kOutOfMemory,
          "surface ACL allocation failure");
  Require(region.base == 0 && driver.live.empty(), "failed allocation must own no span");
  RequireEmptyStats(*allocator);
  driver.fail_malloc = false;
  Require(allocator->Allocate({4096, 256, kvaf::DeviceKind::kCpu}, &region).ok(),
          "retry after ACL allocation failure");
  Require(region.base != 0 && region.length == 4096 && region.slot_size == 256 &&
              region.device == kvaf::DeviceKind::kCpu,
          "publish exact requested host DDR span");
  auto* bytes = reinterpret_cast<unsigned char*>(region.base);
  bytes[0] = 0x12;
  bytes[region.length - 1] = 0x34;
  kvaf::RemoteRegionDesc second;
  Require(allocator->Allocate({4096, 256, kvaf::DeviceKind::kCpu}, &second).code() ==
              kvaf::StatusCode::kInvalidState && second.base == 0,
          "duplicate allocation must preserve the original region");
  auto wrong = region;
  ++wrong.base;
  Require(allocator->Release(wrong).code() == kvaf::StatusCode::kNotFound &&
              driver.free_calls == free_before,
          "wrong address must not release owned DDR");
#if defined(__linux__)
  Require(kvaf::LocalHostDdrPageStats(*allocator).smaps_covered_bytes == region.length,
          "ACL evidence describes the requested range");
#endif
  Require(bytes[0] == 0x12 && bytes[region.length - 1] == 0x34,
          "rejected operations must preserve live payload");
  Require(allocator->Release(region).ok() && driver.live.empty(), "release ACL span");
  Require(driver.malloc_calls == malloc_before + 2 && driver.free_calls == free_before + 1,
          "exactly one successful ACL allocation and release");
  RequireEmptyStats(*allocator);
  Require(allocator->Release(region).code() == kvaf::StatusCode::kNotFound,
          "duplicate release must not double free");
  Require(allocator->Allocate({8192, 512, kvaf::DeviceKind::kCpu}, &second).ok(),
          "allocator can own a new span after release");
  Require(allocator->Release(second).ok() && driver.live.empty(), "release replacement span");
}

void TestPageBackedAllocation() {
  auto allocator = kvaf::CreateLocalHostDdrAllocator("base");
  kvaf::RemoteRegionDesc region;
  const int malloc_before = driver.malloc_calls;
#if defined(__linux__)
  Require(allocator->Allocate({std::numeric_limits<std::uint64_t>::max(), 1,
                               kvaf::DeviceKind::kCpu}, &region).code() ==
              kvaf::StatusCode::kInvalidArgument,
          "reject rounded mmap size overflow before allocation");
  RequireEmptyStats(*allocator);
  constexpr std::size_t requested = kHugePage + 4096;
  const auto status = allocator->Allocate({requested, 4096, kvaf::DeviceKind::kCpu}, &region);
  Require(status.ok(), "base mmap allocation: " + status.message());
  Require(region.base % kHugePage == 0 && region.length == requested && region.slot_size == 4096,
          "aligned mapping must retain unrounded public length");
  auto* bytes = reinterpret_cast<unsigned char*>(region.base);
  Require(std::all_of(bytes, bytes + 2 * kHugePage,
                      [](unsigned char value) { return value == 0; }),
          "entire rounded span must be touched and zero initialized");
  // HCCP may split a registered VMA. Reproduce that with distinct protection
  // on its second half and require evidence to cover both pieces.
  Require(mprotect(bytes + kHugePage, kHugePage, PROT_READ) == 0, "split DDR VMA");
  const auto stats = kvaf::LocalHostDdrPageStats(*allocator);
  Require(stats.smaps_covered_bytes == 2 * kHugePage && stats.anon_huge_bytes == 0,
          "base mode evidence must cover every split VMA and contain no THP");
  const auto released_address = region.base;
  Require(allocator->Release(region).ok(), "munmap entire aligned DDR allocation");
  unsigned char residency = 0;
  errno = 0;
  Require(mincore(reinterpret_cast<void*>(released_address),
                  static_cast<std::size_t>(sysconf(_SC_PAGESIZE)), &residency) == -1 &&
              errno == ENOMEM,
          "released DDR must no longer be mapped");
#else
  Require(allocator->Allocate({4096, 256, kvaf::DeviceKind::kCpu}, &region).code() ==
              kvaf::StatusCode::kInvalidArgument,
          "base mmap mode must explicitly reject non-Linux hosts");
#endif
  RequireEmptyStats(*allocator);
  Require(driver.malloc_calls == malloc_before, "mmap modes must not allocate through ACL");
}

int TestThpAllocation() {
#if defined(__linux__)
  auto allocator = kvaf::CreateLocalHostDdrAllocator("thp");
  kvaf::RemoteRegionDesc region;
  const auto status = allocator->Allocate({2 * kHugePage, 4096, kvaf::DeviceKind::kCpu}, &region);
  if (!status.ok()) {
    Require(region.base == 0, "THP failure must not publish a region");
    RequireEmptyStats(*allocator);
    std::ifstream enabled("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string policy;
    std::getline(enabled, policy);
    if (policy.find("[never]") != std::string::npos &&
        status.code() == kvaf::StatusCode::kInternal &&
        status.message() ==
            "thp Runtime DDR has unexpected smaps page coverage after first touch") {
      std::cout << "SKIP: kernel did not provide verified THP; allocator rejected it: "
                << status.message() << '\n';
      return 77;
    }
    throw std::runtime_error("THP allocation failed: " + status.message());
  }
  const auto stats = kvaf::LocalHostDdrPageStats(*allocator);
  Require(region.base % kHugePage == 0 && stats.smaps_covered_bytes == 2 * kHugePage &&
              stats.anon_huge_bytes > 0,
          "successful THP must have real smaps evidence");
  Require(allocator->Release(region).ok(), "release THP span");
  RequireEmptyStats(*allocator);
  Require(driver.malloc_calls == 0 && driver.free_calls == 0,
          "THP uses mmap rather than the ACL stub");
  std::cout << "PASS: THP bytes=" << stats.anon_huge_bytes << '\n';
  return 0;
#else
  std::cout << "SKIP: verified THP requires Linux /proc/self/smaps\n";
  return 77;
#endif
}
}  // namespace

extern "C" aclError aclrtMallocHost(void** address, std::size_t bytes) {
  ++driver.malloc_calls;
  if (driver.fail_malloc) return 1;
  *address = std::malloc(bytes);
  if (*address == nullptr) return 1;
  driver.live.insert(*address);
  return ACL_SUCCESS;
}

extern "C" aclError aclrtFreeHost(void* address) {
  ++driver.free_calls;
  if (driver.live.erase(address) != 1) return 1;
  std::free(address);
  return ACL_SUCCESS;
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--thp") return TestThpAllocation();
    Require(argc == 1, "usage: local_host_ddr_allocator_test [--thp]");
    TestRequestValidation();
    TestAclAllocationAndRelease();
    TestPageBackedAllocation();
    Require(driver.live.empty(), "all fake ACL spans must be released");
    std::cout << "PASS: local host DDR validation, allocation, release and page evidence\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

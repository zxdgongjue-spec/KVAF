#include "ubmem/hal.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

using kvaf::ubmem::FabricHandle;
using kvaf::ubmem::Hal;
using kvaf::ubmem::Mapping;
using kvaf::ubmem::MemoryProperty;

constexpr std::size_t kTwoMiB = 2ULL << 20;
constexpr std::size_t kOneGiB = 1ULL << 30;
constexpr std::size_t kRequestBytes = kTwoMiB + 4096;
constexpr int kMockError = 25;

void Check(bool passed, const char* detail) {
  if (passed) return;
  std::fprintf(stderr, "%s\n", detail);
  std::abort();
}

struct Allocation {
  std::size_t bytes;
  bool mapped{};
};

std::shared_ptr<Hal> FakeHal() {
  auto hal = std::make_shared<Hal>();
  hal->create = [](void** handle, std::size_t bytes, const MemoryProperty* prop, std::uint64_t flags) {
    Check(prop->side == 0 && prop->device == 0 && prop->memory_type == 3 && flags == 0,
          "expected a Host P2P DDR allocation");
    Check((bytes == 2 * kTwoMiB && prop->page_type == 1) ||
          (bytes == kOneGiB && prop->page_type == 2), "wrong allocation length or HAL page type");
    *handle = new Allocation{bytes};
    return 0;
  };
  hal->reserve = [](void** address, std::size_t bytes, std::size_t alignment, void* hint,
                    std::uint64_t flags) {
    const std::uint64_t page_type = bytes == kOneGiB ? 2 : 1;
    Check(alignment == 0 && hint == nullptr && flags == ((1ULL << 10) | page_type),
          "wrong Host UVA reservation flags");
    *address = reinterpret_cast<void*>(kOneGiB);
    return 0;
  };
  hal->map = [](void*, std::size_t bytes, std::size_t, void* handle, std::uint64_t) {
    auto* allocation = static_cast<Allocation*>(handle);
    Check(bytes == allocation->bytes, "mapping must cover the rounded allocation");
    allocation->mapped = true;
    return 0;
  };
  hal->export_memory = [](void* handle, int type, std::uint64_t, FabricHandle* descriptor) {
    Check(type == 1 && static_cast<Allocation*>(handle)->mapped, "expected a mapped Fabric handle");
    descriptor->front() = 7;
    return 0;
  };
  hal->translate_handle = [](int, FabricHandle*, std::uint32_t*, std::uint64_t*) { return 0; };
  hal->share_attribute = [](std::uint64_t, int, Hal::ShareAttribute) { return 0; };
  hal->unmap = [](void*) { return 0; };
  hal->free_address = [](void*) { return 0; };
  hal->release = [](void* handle) { delete static_cast<Allocation*>(handle); return 0; };
  return hal;
}

}  // namespace

int main() {
  auto hal = FakeHal();
  for (const bool giant_pages : {false, true}) {
    Mapping mapping(hal);
    FabricHandle descriptor{};
    Check(mapping.AllocateHost(kRequestBytes, &descriptor, giant_pages).ok(), "allocation failed");
    Check(mapping.bytes() == (giant_pages ? kOneGiB : 2 * kTwoMiB) && descriptor.front() == 7,
          "wrong rounded length or descriptor");
    Check(mapping.Close().ok() && mapping.bytes() == 0, "mapping release failed");
  }

  hal->create = [](void**, std::size_t, const MemoryProperty* prop, std::uint64_t) {
    Check(prop->page_type == 2, "1 GiB failure must not retry with 2 MiB pages");
    return kMockError;
  };
  Mapping failed(hal);
  FabricHandle descriptor{};
  const auto status = failed.AllocateHost(kRequestBytes, &descriptor, true);
  Check(!status.ok() && status.message().find("1 GiB") != std::string::npos,
        "1 GiB driver error must reach the caller");
  Check(!failed.AllocateHost(std::numeric_limits<std::uint64_t>::max(), &descriptor, true).ok(),
        "rounding overflow must fail before allocation");
}

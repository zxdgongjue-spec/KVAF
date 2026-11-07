#include "ubmem/host_arena.h"
#include "host_memory.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace kvaf {
namespace {

// VA flags select normal pages (0), not access permission. Keep
// them consistent with the physical allocation; MemSetAccess grants NPU access.
constexpr std::uint64_t kReserveBasePage = 0;
constexpr std::int32_t kRuntime92 = 90200000;

Status AclStatus(aclError code, const char* operation) {
  if (code == ACL_SUCCESS) return Status::OK();
  return Status::Internal(std::string(operation) + " failed, rc=" +
                          std::to_string(code));
}

bool SameRegion(const RemoteRegionDesc& lhs, const RemoteRegionDesc& rhs) {
  return lhs.base == rhs.base && lhs.length == rhs.length &&
         lhs.slot_size == rhs.slot_size && lhs.device == rhs.device;
}

void LogThpCoverage(const char* stage, std::uint32_t device, void* host,
                    std::size_t bytes) {
  const auto coverage = ubmem_host::ReadThpCoverage(host, bytes);
  std::fprintf(stderr,
      "[KVAF_UBMEM] UBMEM_THP_COVERAGE stage=%s device=%u host_base=%p "
      "arena_bytes=%zu status=%s measured_bytes=%llu anon_huge_bytes=%llu\n",
      stage, device, host, bytes,
      !coverage.available ? "unavailable" : coverage.exact ? "exact" : "partial",
      static_cast<unsigned long long>(coverage.measured_bytes),
      static_cast<unsigned long long>(coverage.anon_huge_bytes));
}

}  // namespace

struct UbmemDdrMemoryAllocator::Impl {
  Impl(std::uint32_t device, UbmemPageMode mode, int node)
      : device_id(device), page_mode(mode), numa_node(node) {}

  struct Arena {
    aclrtDrvMemHandle physical{nullptr};
    void* address{nullptr};
    void* host_allocation{nullptr};
    RemoteRegionDesc region;
    UbmemMappedRegion view;
    bool physical_allocated{false};
    bool virtual_reserved{false};
    bool mapped{false};
    bool host_registered{false};
    bool numa_bound{false};
    bool published{false};
  };

  Status CheckDevice() const {
    if (device_id > static_cast<std::uint32_t>(
                        std::numeric_limits<std::int32_t>::max())) {
      return Status::InvalidArgument("UBmem device_id exceeds the ACL device ID range");
    }
    std::int32_t current = -1;
    Status status = AclStatus(aclrtGetDevice(&current), "aclrtGetDevice");
    if (!status.ok()) return status;
    if (current < 0 || static_cast<std::uint32_t>(current) != device_id) {
      return Status::InvalidState("current ACL device differs from UBmem arena device");
    }
    return Status::OK();
  }

  // Stop at the first failure. In particular, failed synchronization must not
  // unmap live DMA/kernel memory, and failed unmap must not free its backing.
  Status Cleanup(bool synchronize) {
    if (!arena) return Status::OK();
    Status status = CheckDevice();
    if (!status.ok()) return status;
    if (synchronize && (arena->mapped || arena->host_registered)) {
      status = AclStatus(aclrtSynchronizeDevice(), "UBmem release synchronize");
      if (!status.ok()) return status;
    }
    if (page_mode == UbmemPageMode::kThp) {
      if (arena->host_registered) {
        status = AclStatus(aclrtHostUnregister(arena->host_allocation), "aclrtHostUnregister(THP)");
        if (!status.ok()) return status;
        arena->host_registered = false;
        arena->mapped = false;
      }
      if (arena->numa_bound) {
        status = ubmem_host::BindNuma(arena->host_allocation, arena->view.mapped_bytes, -1);
        if (!status.ok()) return status;
        arena->numa_bound = false;
      }
      if (arena->host_allocation != nullptr) ubmem_host::Free(arena->host_allocation);
      arena.reset();
      return Status::OK();
    }
    if (arena->mapped) {
      status = AclStatus(aclrtUnmapMem(arena->address), "aclrtUnmapMem");
      if (!status.ok()) return status;
      arena->mapped = false;
    }
    if (arena->physical_allocated) {
      status = AclStatus(aclrtFreePhysical(arena->physical), "aclrtFreePhysical");
      if (!status.ok()) return status;
      arena->physical_allocated = false;
      arena->physical = nullptr;
    }
    if (arena->virtual_reserved) {
      status = AclStatus(aclrtReleaseMemAddress(arena->address),
                         "aclrtReleaseMemAddress");
      if (!status.ok()) return status;
      arena->virtual_reserved = false;
      arena->address = nullptr;
    }
    arena.reset();
    return Status::OK();
  }

  Status AllocationFailed(Status failure) {
    // No work can reference an arena that has not been returned to the caller.
    const Status cleanup = Cleanup(false);
    if (!cleanup.ok()) {
      return Status::Error(failure.code(), failure.message() +
          "; UBmem cleanup retained resources: " + cleanup.message());
    }
    return failure;
  }

  Status AllocateThp(const RuntimeInitRequest& request, RemoteRegionDesc* region) {
    if (!ubmem_host::ThpSupported()) return Status::NotImplemented("UBmem thp requires Linux");
    aclrtHostMemMapCapability capability = ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
    Status status = AclStatus(aclrtHostMemMapCapabilities(
        device_id, ACL_RT_HAC_TYPE_AIV, &capability), "aclrtHostMemMapCapabilities(AIV)");
    if (!status.ok()) return status;
    if (capability != ACL_RT_HOST_MEM_MAP_SUPPORTED) {
      return Status::NotImplemented("UBmem THP requires mapped Host memory access from AIV");
    }
    const auto bytes = static_cast<std::size_t>(request.ddr_bytes);
    constexpr auto alignment = ubmem_host::kThpAlignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
      return Status::InvalidArgument("UBmem THP allocation size overflow");
    }
    const auto rounded = ((bytes + alignment - 1) / alignment) * alignment;
    arena.reset(new (std::nothrow) Arena());
    if (!arena) return Status::OutOfMemory("failed to allocate UBmem arena metadata");
    arena->view = {0, 0, request.ddr_bytes, rounded, alignment, device_id};
    std::fprintf(stderr,
        "[KVAF_UBMEM] UBMEM_ARENA_ALLOC device=%u page_mode=thp valid_bytes=%llu "
        "mapped_bytes=%zu backing_alignment=%zu numa_policy=%s numa_node=%d "
        "physical_page_size=unmeasured\n", device_id,
        static_cast<unsigned long long>(request.ddr_bytes), rounded, alignment,
        numa_node >= 0 ? "bind" : "process", numa_node);
    status = ubmem_host::AllocateThp(&arena->host_allocation, rounded);
    if (!status.ok()) return AllocationFailed(std::move(status));
    const auto host = reinterpret_cast<std::uintptr_t>(arena->host_allocation);
    if (host == 0 || host % alignment != 0 ||
        host > std::numeric_limits<std::uintptr_t>::max() - rounded) {
      return AllocationFailed(Status::Internal("invalid UBmem THP Host allocation address"));
    }
    status = ubmem_host::AdviseThp(arena->host_allocation, rounded);
    if (!status.ok()) return AllocationFailed(std::move(status));
    if (numa_node >= 0) {
      status = ubmem_host::BindNuma(arena->host_allocation, rounded, numa_node);
      if (!status.ok()) return AllocationFailed(std::move(status));
      arena->numa_bound = true;
      std::fprintf(stderr, "[KVAF_UBMEM] UBMEM_NUMA_BIND device=%u node=%d bytes=%zu\n",
                   device_id, numa_node, rounded);
    }
    ubmem_host::Prefault(arena->host_allocation, rounded);
    LogThpCoverage("before_register", device_id, arena->host_allocation, rounded);
    // Match the A5 host_register_flags() contract used by vLLM: MAPPED also
    // accepts ordinary Host mappings; adding PINNED is rejected with 107000.
    status = AclStatus(aclrtHostRegisterV2(arena->host_allocation, rounded,
                                         ACL_HOST_REG_MAPPED),
                       "aclrtHostRegisterV2(THP,MAPPED)");
    if (!status.ok()) return AllocationFailed(std::move(status));
    arena->host_registered = true;
    LogThpCoverage("after_register", device_id, arena->host_allocation, rounded);
    status = AclStatus(aclrtHostGetDevicePointer(arena->host_allocation, &arena->address, 0),
                       "aclrtHostGetDevicePointer(THP)");
    if (!status.ok()) return AllocationFailed(std::move(status));
    const auto device = reinterpret_cast<std::uintptr_t>(arena->address);
    if (device == 0 || device % 32 != 0 ||
        device > std::numeric_limits<std::uintptr_t>::max() - rounded) {
      return AllocationFailed(Status::Internal("invalid UBmem THP mapped device address"));
    }
    arena->mapped = true;
    arena->view.host_base = host;
    arena->view.device_base = device;
    arena->region = {device, request.ddr_bytes, request.slot_size, DeviceKind::kCpu};
    arena->published = true;
    *region = arena->region;
    return Status::OK();
  }

  std::uint32_t device_id;
  UbmemPageMode page_mode;
  int numa_node;
  std::unique_ptr<Arena> arena;
};

UbmemDdrMemoryAllocator::UbmemDdrMemoryAllocator(std::uint32_t device_id,
                                               UbmemPageMode page_mode, int numa_node)
    : impl_(std::make_unique<Impl>(device_id, page_mode, numa_node)) {}

UbmemDdrMemoryAllocator::~UbmemDdrMemoryAllocator() {
  if (!impl_->arena) return;
  const Status status = impl_->Cleanup(impl_->arena->published);
  if (!status.ok()) {
    // Intentionally leave driver-owned mappings/handles alive. Destroying the
    // bookkeeping object does not free them. A runtime with uncertain work
    // completion must retain its whole Impl and never destroy this allocator.
    std::fprintf(stderr, "KVAF UBmem arena retained until process exit: %s\n",
                 status.message().c_str());
  }
}

Status UbmemDdrMemoryAllocator::Allocate(const RuntimeInitRequest& request,
                                        RemoteRegionDesc* remote_region) {
  if (remote_region == nullptr) {
    return Status::InvalidArgument("remote_region must not be null");
  }
  *remote_region = {};
  if (impl_->page_mode != UbmemPageMode::kBase && impl_->page_mode != UbmemPageMode::kThp) {
    return Status::InvalidArgument("UBmem page mode must be thp or base");
  }
  if (impl_->numa_node < -1 || impl_->numa_node > 4095 ||
      (impl_->numa_node >= 0 && impl_->page_mode != UbmemPageMode::kThp)) {
    return Status::InvalidArgument("UBmem NUMA node requires THP mode and must be -1 or 0..4095");
  }
  if (impl_->arena) return Status::Busy("UBmem allocator already owns an arena");
  if (request.ddr_device != DeviceKind::kCpu || request.ddr_bytes == 0 ||
      request.slot_size == 0 || request.ddr_bytes % request.slot_size != 0) {
    return Status::InvalidArgument(
        "UBmem requires CPU DDR bytes that are a non-zero multiple of slot_size");
  }
  if (request.ddr_bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("UBmem DDR bytes exceed the host address range");
  }
  Status status = impl_->CheckDevice();
  if (!status.ok()) return status;
  char package[] = "runtime";
  std::int32_t version = 0;
  status = AclStatus(aclsysGetVersionNum(package, &version),
                     "aclsysGetVersionNum(runtime)");
  if (!status.ok()) return status;
  if (version < kRuntime92) {
    return Status::NotImplemented("UBmem Host DDR allocator requires CANN runtime >= 9.2");
  }
  if (impl_->page_mode == UbmemPageMode::kThp) return impl_->AllocateThp(request, remote_region);

  aclrtPhysicalMemProp prop{};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_DDR_MEM_P2P_NORMAL;
  prop.location.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
  // -1 asks the CANN driver to choose the Host NUMA node.
  prop.location.id = static_cast<std::uint32_t>(-1);
  std::size_t granularity = 0;
  status = AclStatus(aclrtMemGetAllocationGranularity(
                        &prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity),
                     "aclrtMemGetAllocationGranularity");
  if (!status.ok()) return status;
  const auto bytes = static_cast<std::size_t>(request.ddr_bytes);
  if (granularity == 0 ||
      bytes > std::numeric_limits<std::size_t>::max() - (granularity - 1)) {
    return Status::InvalidArgument("invalid UBmem granularity or allocation size overflow");
  }
  const std::size_t rounded = ((bytes + granularity - 1) / granularity) * granularity;
  impl_->arena.reset(new (std::nothrow) Impl::Arena());
  if (!impl_->arena) return Status::OutOfMemory("failed to allocate UBmem arena metadata");
  auto& arena = *impl_->arena;
  arena.view = {0, 0, request.ddr_bytes, rounded, granularity, impl_->device_id};
  std::fprintf(stderr,
      "[KVAF_UBMEM] UBMEM_ARENA_ALLOC device=%u page_mode=%s valid_bytes=%llu "
      "mapped_bytes=%zu granularity=%zu numa_id=-1\n",
      impl_->device_id, "base",
      static_cast<unsigned long long>(request.ddr_bytes), rounded, granularity);

  status = AclStatus(aclrtMallocPhysical(&arena.physical, rounded, &prop, 0),
                     "aclrtMallocPhysical(HOST_NUMA,DDR_P2P_NORMAL)");
  if (!status.ok()) return impl_->AllocationFailed(std::move(status));
  if (arena.physical == nullptr) {
    return impl_->AllocationFailed(Status::Internal("UBmem physical allocator returned null"));
  }
  arena.physical_allocated = true;
  status = AclStatus(aclrtReserveMemAddress(&arena.address, rounded, 0, nullptr,
                        kReserveBasePage),
                     "aclrtReserveMemAddress(base)");
  if (!status.ok()) return impl_->AllocationFailed(std::move(status));
  if (arena.address == nullptr) {
    return impl_->AllocationFailed(Status::Internal("UBmem VA reservation returned null"));
  }
  arena.virtual_reserved = true;
  const auto address = reinterpret_cast<std::uintptr_t>(arena.address);
  if (address > std::numeric_limits<std::uintptr_t>::max() - rounded) {
    return impl_->AllocationFailed(Status::Internal("UBmem mapping address range overflows"));
  }
  status = AclStatus(aclrtMapMem(arena.address, rounded, 0, arena.physical, 0),
                     "aclrtMapMem(HOST_NUMA)");
  if (!status.ok()) return impl_->AllocationFailed(std::move(status));
  arena.mapped = true;
  aclrtMemAccessDesc access{};
  access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = impl_->device_id;
  access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
  status = AclStatus(aclrtMemSetAccess(arena.address, rounded, &access, 1),
                     "aclrtMemSetAccess(DEVICE,READWRITE)");
  if (!status.ok()) return impl_->AllocationFailed(std::move(status));

  // HOST_NUMA maps CPU-accessible VA; SetAccess grants this worker's NPU access
  // to that VA. Consumers still use distinct host/device address fields.
  arena.view.host_base = address;
  arena.view.device_base = address;
  arena.region = {arena.view.device_base, request.ddr_bytes, request.slot_size,
                  DeviceKind::kCpu};
  arena.published = true;
  *remote_region = arena.region;
  return Status::OK();
}

Status UbmemDdrMemoryAllocator::Release(const RemoteRegionDesc& remote_region) {
  if (!impl_->arena || !impl_->arena->published) {
    return Status::NotFound("UBmem arena was not allocated");
  }
  if (!SameRegion(remote_region, impl_->arena->region)) {
    return Status::InvalidArgument("remote_region does not identify this UBmem arena");
  }
  return impl_->Cleanup(true);
}

Status UbmemDdrMemoryAllocator::GetMappedRegion(
    const RemoteRegionDesc& remote_region, UbmemMappedRegion* mapped_region) const {
  if (mapped_region == nullptr) {
    return Status::InvalidArgument("mapped_region must not be null");
  }
  *mapped_region = {};
  if (!impl_->arena || !impl_->arena->published || !impl_->arena->mapped) {
    return Status::NotFound("UBmem arena has no live mapping");
  }
  if (!SameRegion(remote_region, impl_->arena->region)) {
    return Status::InvalidArgument("remote_region does not identify this UBmem arena");
  }
  *mapped_region = impl_->arena->view;
  return Status::OK();
}

std::unique_ptr<UbmemDdrMemoryAllocator> CreateUbmemDdrMemoryAllocator(
    std::uint32_t device_id, UbmemPageMode page_mode, int numa_node) {
  return std::make_unique<UbmemDdrMemoryAllocator>(device_id, page_mode, numa_node);
}

}  // namespace kvaf

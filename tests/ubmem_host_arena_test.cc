#include <acl/acl.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ubmem/host_arena.h"
#include "../runtime/src/ubmem/host_memory.h"
#if defined(KVAF_TEST_UBMEM_FUSED_CONTEXT)
#include "npu_runtime.h"
#endif

namespace {

struct Driver {
  std::int32_t device{1};
  std::int32_t version{90200000};
  std::size_t granularity{2 * 1024 * 1024};
  kvaf::UbmemPageMode page_mode{kvaf::UbmemPageMode::kBase};
  std::size_t allocated{0};
  std::string failures;
  std::string calls;
  bool physical{false};
  bool reserved{false};
  bool mapped{false};
  bool thp_supported{true};
  bool host_map_supported{true};
  bool host_allocated{false};
  bool host_registered{false};
  bool null_device_address{false};
  unsigned coverage_reads{0};
} driver;

void ResetDriver(kvaf::UbmemPageMode mode) {
  driver = {};
  driver.page_mode = mode;
  if (mode == kvaf::UbmemPageMode::kBase) driver.granularity = 4096;
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

aclError Call(char name) {
  driver.calls += name;
  return driver.failures.find(name) == std::string::npos ? ACL_SUCCESS : 999;
}

void* Address() { return reinterpret_cast<void*>(0x10000000); }
void* Handle() { return reinterpret_cast<void*>(0x31); }
void* HostDeviceAddress() { return reinterpret_cast<void*>(0x20000000); }

void Properties(const aclrtPhysicalMemProp* prop) {
  const auto expected_attr = driver.page_mode == kvaf::UbmemPageMode::kBase
                                 ? ACL_DDR_MEM_P2P_NORMAL : ACL_DDR_MEM_P2P_HUGE;
  Require(prop != nullptr && prop->handleType == ACL_MEM_HANDLE_TYPE_NONE &&
              prop->allocationType == ACL_MEM_ALLOCATION_TYPE_PINNED &&
              prop->memAttr == expected_attr &&
              prop->location.type == ACL_MEM_LOCATION_TYPE_HOST_NUMA &&
              prop->location.id == static_cast<std::uint32_t>(-1),
          "HOST_NUMA DDR_P2P allocation properties match selected page mode");
}

kvaf::RuntimeInitRequest Request() { return {8448, 256, kvaf::DeviceKind::kCpu}; }

void TestExactCapacityAndRelease(kvaf::UbmemPageMode mode) {
  ResetDriver(mode);
  auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, mode);
  kvaf::RemoteRegionDesc region;
  Require(allocator->Allocate(Request(), &region).ok(), "allocate");
  Require(region.base == reinterpret_cast<std::uintptr_t>(Address()) &&
              region.length == 8448 && region.slot_size == 256 &&
              region.device == kvaf::DeviceKind::kCpu,
          "publish exact valid capacity and NPU address");
  kvaf::UbmemMappedRegion mapping;
  Require(allocator->GetMappedRegion(region, &mapping).ok(), "describe mapping");
  Require(mapping.host_base == region.base && mapping.device_base == region.base &&
              mapping.valid_bytes == 8448 &&
              mapping.mapped_bytes == (mode == kvaf::UbmemPageMode::kBase ? 12288 : 2 * 1024 * 1024) &&
              mapping.granularity == driver.granularity && mapping.device_id == 1,
          "keep padding out of slot capacity");
  kvaf::RemoteRegionDesc other;
  Require(allocator->Allocate(Request(), &other).code() == kvaf::StatusCode::kBusy,
          "one active arena");
  other = region;
  ++other.length;
  Require(!allocator->Release(other).ok(), "reject foreign descriptor");
  driver.calls.clear();
  Require(allocator->Release(region).ok(), "release");
  Require(driver.calls == "dsufx" && !driver.physical && !driver.reserved && !driver.mapped,
          "sync before unmap before physical free before VA release");
  Require(!allocator->GetMappedRegion(region, &mapping).ok() && mapping.device_base == 0,
          "released mapping is not exposed");
  Require(!allocator->Release(region).ok(), "reject duplicate release");
}

void TestInvalidRequests(kvaf::UbmemPageMode mode) {
  ResetDriver(mode);
  auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, mode);
  kvaf::RemoteRegionDesc region;
  Require(!allocator->Allocate(Request(), nullptr).ok(), "reject null output");
  for (const auto request : {kvaf::RuntimeInitRequest{0, 1, kvaf::DeviceKind::kCpu},
                             kvaf::RuntimeInitRequest{8, 0, kvaf::DeviceKind::kCpu},
                             kvaf::RuntimeInitRequest{8, 3, kvaf::DeviceKind::kCpu},
                             kvaf::RuntimeInitRequest{8, 1, kvaf::DeviceKind::kNpu}}) {
    Require(!allocator->Allocate(request, &region).ok(), "reject invalid geometry");
  }
  Require(driver.calls.empty(), "invalid geometry performs no ACL operations");
  auto invalid_device = kvaf::CreateUbmemDdrMemoryAllocator(
      std::numeric_limits<std::uint32_t>::max(), mode);
  Require(!invalid_device->Allocate(Request(), &region).ok() && driver.calls.empty(),
          "reject device ID before narrowing");
  auto invalid_mode = kvaf::CreateUbmemDdrMemoryAllocator(
      1, static_cast<kvaf::UbmemPageMode>(42));
  Require(invalid_mode->Allocate(Request(), &region).code() ==
              kvaf::StatusCode::kInvalidArgument && driver.calls.empty(),
          "reject invalid page mode before calling ACL");
  driver.device = 0;
  Require(!allocator->Allocate(Request(), &region).ok(), "reject wrong current device");
  driver.device = 1;
  driver.version = 90100000;
  Require(allocator->Allocate(Request(), &region).code() == kvaf::StatusCode::kNotImplemented,
          "require runtime 9.2 user device ID semantics");
  driver.version = 90200000;
  driver.granularity = 0;
  Require(!allocator->Allocate(Request(), &region).ok(), "reject zero granularity");
  driver.granularity = 4096;
  Require(!allocator->Allocate({std::numeric_limits<std::uint64_t>::max(), 1,
                                kvaf::DeviceKind::kCpu}, &region).ok(),
          "reject rounding overflow");
  Require(!driver.physical && !driver.reserved, "invalid requests allocate nothing");
  driver.granularity = 3;
  Require(allocator->Allocate({10, 2, kvaf::DeviceKind::kCpu}, &region).ok() &&
              driver.allocated == 12 && region.length == 10,
          "round without assuming a power-of-two granularity");
  Require(allocator->Release(region).ok(), "release non-power-of-two test allocation");
}

void TestConstructionFailures(kvaf::UbmemPageMode mode) {
  for (char failure : std::string("dvgprma")) {
    ResetDriver(mode);
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, mode);
    driver.failures = failure;
    kvaf::RemoteRegionDesc region{123, 8, 1, kvaf::DeviceKind::kCpu};
    Require(!allocator->Allocate(Request(), &region).ok() && region.base == 0,
            "construction failure does not publish a region");
    Require(!driver.physical && !driver.reserved && !driver.mapped,
            "construction failure unwinds acquired resources");
    Require(driver.calls.find('s') == std::string::npos,
            "unpublished allocation has no submitted work");
    driver.failures.clear();
    Require(allocator->Allocate(Request(), &region).ok(), "retry after cleaned construction");
    Require(allocator->Release(region).ok(), "release retry");
  }
}

void TestReleaseFailuresRetainDependencies(kvaf::UbmemPageMode mode) {
  for (char failure : std::string("sufx")) {
    ResetDriver(mode);
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, mode);
    kvaf::RemoteRegionDesc region;
    Require(allocator->Allocate(Request(), &region).ok(), "allocate for release failure");
    driver.failures = failure;
    driver.calls.clear();
    Require(!allocator->Release(region).ok(), "report release failure");
    Require(driver.calls.back() == failure, "stop cleanup at first failure");
    Require(driver.reserved, "retain VA reservation until its own release succeeds");
    Require(driver.physical == (failure != 'x'), "retain physical pages while unmap/free pending");
    Require(driver.mapped == (failure == 's' || failure == 'u'), "retain mapping on sync/unmap failure");
    driver.failures.clear();
    Require(allocator->Release(region).ok(), "retry remaining teardown steps");
    Require(!driver.mapped && !driver.physical && !driver.reserved, "retry completes teardown");
  }
  ResetDriver(mode);
  {
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, mode);
    kvaf::RemoteRegionDesc region;
    Require(allocator->Allocate(Request(), &region).ok(), "allocate for destructor");
    driver.calls.clear();
    driver.failures = "s";
  }
  Require(driver.calls == "ds" && driver.mapped && driver.physical && driver.reserved,
          "destructor never frees memory after unknown completion");
}

void TestFailedConstructionCleanupIsQuarantined(kvaf::UbmemPageMode mode) {
  ResetDriver(mode);
  {
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, mode);
    driver.failures = "au";
    kvaf::RemoteRegionDesc region;
    Require(!allocator->Allocate(Request(), &region).ok(), "access grant failure");
    Require(driver.mapped && driver.physical && driver.reserved,
            "failed unmap retains dependent resources");
    Require(allocator->Allocate(Request(), &region).code() == kvaf::StatusCode::kBusy,
            "cannot overwrite quarantined construction state");
    driver.failures.clear();
    driver.calls.clear();
  }
  Require(driver.calls == "dufx" && !driver.mapped && !driver.physical && !driver.reserved,
          "destructor can retry unpublished cleanup without a device synchronization");
}

void TestThpNumaBinding() {
  ResetDriver(kvaf::UbmemPageMode::kThp);
  auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp, 2);
  kvaf::RemoteRegionDesc region;
  Require(allocator->Allocate(Request(), &region).ok(), "THP with explicit NUMA node");
  Require(driver.calls == "dvcHPNTBRAG", "bind before prefault and register");
  driver.calls.clear();
  driver.failures = {'n'};
  Require(!allocator->Release(region).ok(), "failed policy cleanup retains allocation");
  Require(driver.host_allocated, "do not free while range policy cleanup is incomplete");
  Require(driver.calls == "dsUn", "unregister before policy cleanup, stop on failure");
  driver.calls.clear();
  driver.failures.clear();
  Require(allocator->Release(region).ok(), "release bound arena");
  Require(driver.calls == "dnF", "retry cleanup resets policy before freeing");
  driver.calls.clear();
  driver.failures = {'N'};
  Require(!allocator->Allocate(Request(), &region).ok(), "binding failure stops startup");
  Require(driver.calls == "dvcHPNdF", "binding failure frees before any touch or registration");
  Require(!driver.host_allocated, "binding failure releases Host storage");
}

void TestThpAllocationAndRelease() {
  ResetDriver(kvaf::UbmemPageMode::kThp);
  auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
  kvaf::RemoteRegionDesc region;
  Require(allocator->Allocate(Request(), &region).ok(), "allocate THP backing");
  Require(driver.calls == "dvcHPTBRAG", "capability before allocation, THP prefault before registration");
  kvaf::UbmemMappedRegion mapping;
  Require(allocator->GetMappedRegion(region, &mapping).ok(), "describe THP mapping");
  Require(region.base == reinterpret_cast<std::uintptr_t>(HostDeviceAddress()) &&
              mapping.host_base == reinterpret_cast<std::uintptr_t>(Address()) &&
              mapping.device_base == region.base && mapping.host_base != mapping.device_base &&
              mapping.mapped_bytes == 2 * 1024 * 1024 && mapping.valid_bytes == 8448 &&
              region.length == 8448 && region.slot_size == 256,
          "THP rounds only backing and preserves independent Host/device addresses");
  Require(driver.coverage_reads == 2, "report coverage before and after registration, including zero after");
  driver.calls.clear();
  Require(allocator->Release(region).ok(), "release THP");
  Require(driver.calls == "dsUF" && !driver.host_allocated && !driver.host_registered,
          "THP device fence before unregister before host free, no VMM cleanup");
}

void TestThpConstructionFailures() {
  for (const char failure : std::string("dvcHPRG")) {
    ResetDriver(kvaf::UbmemPageMode::kThp);
    driver.failures = failure;
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
    kvaf::RemoteRegionDesc region{123, 8, 1, kvaf::DeviceKind::kCpu};
    Require(!allocator->Allocate(Request(), &region).ok() && region.base == 0,
            "THP construction failure never publishes an address");
    Require(!driver.host_allocated && !driver.host_registered,
            "THP construction failure unwinds in dependency order");
    Require(driver.calls.find('s') == std::string::npos, "unpublished THP needs no execution fence");
    if (failure == 'G') Require(driver.calls.find("GdUF") != std::string::npos,
                                "failed device-pointer lookup unregisters before free");
    driver.failures.clear();
    Require(allocator->Allocate(Request(), &region).ok(), "retry after THP construction failure");
    Require(allocator->Release(region).ok(), "release THP retry");
  }
  for (bool platform_failure : {false, true}) {
    ResetDriver(kvaf::UbmemPageMode::kThp);
    driver.thp_supported = !platform_failure;
    driver.host_map_supported = platform_failure;
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
    kvaf::RemoteRegionDesc region;
    Require(allocator->Allocate(Request(), &region).code() == kvaf::StatusCode::kNotImplemented &&
                driver.calls.find('H') == std::string::npos,
            "reject unavailable Linux/AIV Host mapping before allocation");
  }
  ResetDriver(kvaf::UbmemPageMode::kThp);
  auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
  kvaf::RemoteRegionDesc region;
  Require(!allocator->Allocate({std::numeric_limits<std::uint64_t>::max(), 1,
                                kvaf::DeviceKind::kCpu}, &region).ok() &&
              driver.calls.find('H') == std::string::npos,
          "THP rounding overflow is rejected before allocation");
  driver.null_device_address = true;
  Require(!allocator->Allocate(Request(), &region).ok() && !driver.host_allocated &&
              !driver.host_registered, "null mapped device address is unregistered and freed");
}

void TestThpCleanupRetention() {
  for (const char failure : std::string("sU")) {
    ResetDriver(kvaf::UbmemPageMode::kThp);
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
    kvaf::RemoteRegionDesc region;
    Require(allocator->Allocate(Request(), &region).ok(), "allocate THP for cleanup failure");
    driver.failures = failure;
    driver.calls.clear();
    Require(!allocator->Release(region).ok() && driver.calls.back() == failure &&
                driver.host_registered && driver.host_allocated,
            "failed fence/unregister retains registered Host backing");
    driver.failures.clear();
    Require(allocator->Release(region).ok() && !driver.host_allocated && !driver.host_registered,
            "retry THP cleanup safely");
  }
  ResetDriver(kvaf::UbmemPageMode::kThp);
  {
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
    kvaf::RemoteRegionDesc region;
    driver.failures = "GU";
    Require(!allocator->Allocate(Request(), &region).ok() && driver.host_registered &&
                driver.host_allocated, "failed partial THP construction is retained on unregister failure");
    Require(allocator->Allocate(Request(), &region).code() == kvaf::StatusCode::kBusy,
            "cannot overwrite retained THP construction state");
    driver.failures.clear();
    driver.calls.clear();
  }
  Require(driver.calls == "dUF" && !driver.host_allocated,
          "destructor retries unpublished THP unregister without a device fence");
  ResetDriver(kvaf::UbmemPageMode::kThp);
  {
    auto allocator = kvaf::CreateUbmemDdrMemoryAllocator(1, kvaf::UbmemPageMode::kThp);
    kvaf::RemoteRegionDesc region;
    Require(allocator->Allocate(Request(), &region).ok(), "allocate published THP before destructor");
    driver.failures = "s";
    driver.calls.clear();
  }
  Require(driver.calls == "ds" && driver.host_allocated && driver.host_registered,
          "THP destructor retains storage after unknown completion");
}

void TestThpCoverageParser() {
  auto parse = [](const std::string& text) {
    std::istringstream input(text);
    return kvaf::ubmem_host::ParseThpCoverage(input, 0x200000, 0x400000);
  };
  auto coverage = parse("00200000-00600000 rw-p 0 0:0 0\nAnonHugePages: 4096 kB\n");
  Require(coverage.available && coverage.exact && coverage.measured_bytes == 0x400000 &&
              coverage.anon_huge_bytes == 0x400000, "exact arena THP coverage");
  coverage = parse("00200000-00600000 rw-p 0 0:0 0\nAnonHugePages: 0 kB\n");
  Require(coverage.exact && coverage.anon_huge_bytes == 0,
          "zero THP coverage is reported as zero, not absent or guaranteed huge");
  coverage = parse("00100000-00400000 rw-p 0 0:0 0\nAnonHugePages: 2048 kB\n"
                   "00400000-00600000 rw-p 0 0:0 0\nAnonHugePages: 0 kB\n");
  Require(coverage.available && !coverage.exact && coverage.measured_bytes == 0x200000 &&
              coverage.anon_huge_bytes == 0, "never attribute partial VMA huge pages to the arena");
  coverage = parse("00200000-00400000 rw-p 0 0:0 0\nAnonHugePages: 2048 kB\n"
                   "00400000-00600000 rw-p 0 0:0 0\nAnonHugePages: 0 kB\n");
  Require(coverage.exact && coverage.anon_huge_bytes == 0x200000,
          "aggregate complete VMAs without claiming full huge-page coverage");
  Require(!parse("00200000-00600000 rw-p 0 0:0 0\nRss: 4096 kB\n").exact,
          "missing AnonHugePages is unknown rather than zero");
}

#if defined(KVAF_TEST_UBMEM_FUSED_CONTEXT)
kvaf::NpuRuntimeOptions FusedOptions(kvaf::UbmemPageMode mode) {
  kvaf::NpuRuntimeOptions options("fake-get", 656, 1);
  options.batch_put_kernel_object_path = "fake-put";
  options.transport = kvaf::Transport::kUbmem;
  options.ubmem_page_mode = mode;
  options.device_id = 1;
  return options;
}

void TestNativeFusedContext(kvaf::UbmemPageMode mode) {
  ResetDriver(mode);
  kvaf::NpuRuntime runtime(FusedOptions(mode));
  Require(runtime.Initialize(656 * 8, 656).ok(), "initialize native fused Runtime");
  alignas(32) std::uint8_t output[656 * 2]{}, other_output[656 * 2]{};
  int stream = 1, other_stream = 2;
  kvaf::UbmemFusedContext owner, view;
  Require(runtime.AcquireUbmemFusedContext(output, sizeof(output), &stream, &owner).ok(),
          "acquire native fused context");
  Require(owner.lease_id != 0 && owner.abi_version == 1 && owner.device_id == 1 &&
              owner.region_base == runtime.core().remote_region().base &&
              owner.region_bytes == 656 * 8 && owner.slot_bytes == 656 &&
              owner.output_address == reinterpret_cast<std::uintptr_t>(output) &&
              owner.output_bytes == sizeof(output) && owner.token_bytes == 656 &&
              owner.tokens_per_block == 1, "publish independent mapping and geometry contract");
  Require(runtime.AcquireUbmemFusedContext(output, sizeof(output), &stream, &view).code() ==
              kvaf::StatusCode::kBusy, "reject duplicate native owner");
  Require(runtime.RegisterUbmemFusedOutput(owner, other_output, sizeof(other_output), &view).ok(),
          "register another output view");
  Require(view.lease_id == owner.lease_id && view.region_base == owner.region_base &&
              view.output_address != owner.output_address, "views share only lease and arena");
  Require(!runtime.ReleaseUbmemFusedContext(view, &stream).ok(), "output view cannot release owner");
  Require(!runtime.ReleaseUbmemFusedContext(owner, &other_stream).ok(), "reject different release stream");
  driver.device = 0;
  Require(!runtime.RegisterUbmemFusedOutput(owner, output, sizeof(output), &view).ok(),
          "reject wrong current device");
  driver.device = 1;
  Require(runtime.core().BatchPut({{0}, {output, 656, kvaf::DeviceKind::kNpu}, {0}, {656}}).ok(),
          "Put can append while arena leased");
  kvaf::AccessTableHandle table;
  Require(runtime.core().PrepareAccessTable({{0}}, &table).ok(), "prepare native fused table");
  Require(runtime.RegisterUbmemFusedAccessTable(owner, table).ok(), "pin optional native table");
  Require(runtime.RegisterUbmemFusedAccessTable(owner, table).ok(), "duplicate registration is idempotent");
  Require(runtime.core().ReleaseAccessTable(table).code() == kvaf::StatusCode::kBusy,
          "registered table cannot lose metadata pin");
  Require(runtime.Shutdown().code() == kvaf::StatusCode::kBusy, "live graph arena cannot shut down");
  driver.calls.clear();
  Require(runtime.ReleaseUbmemFusedContext(owner, &stream).ok() && driver.calls == "dds",
          "native release fences device before removing table readers");
  Require(runtime.core().ReleaseAccessTable(table).ok(), "release original table after extra pin drops");
  Require(runtime.core().BatchFree({{0}}).ok(), "key becomes reusable after table release");
  Require(!runtime.ReleaseUbmemFusedContext(owner, &stream).ok(), "reject stale owner");
  Require(runtime.AcquireUbmemFusedContext(output, sizeof(output), &stream, &view).ok() &&
              view.lease_id != owner.lease_id, "new acquisition receives a fresh lease ID");
  Require(!runtime.ReleaseUbmemFusedContext(owner, &stream).ok(), "old owner cannot release a new lease");
  Require(runtime.ReleaseUbmemFusedContext(view, &stream).ok(), "release new lease");
  Require(runtime.Shutdown().ok(), "safe native shutdown");
}

void TestNativeFusedUnknownCompletion(kvaf::UbmemPageMode mode) {
  ResetDriver(mode);
  {
    kvaf::NpuRuntime runtime(FusedOptions(mode));
    Require(runtime.Initialize(656 * 8, 656).ok(), "initialize poisoned fused Runtime");
    alignas(32) std::uint8_t output[656 * 2]{};
    int stream = 1;
    kvaf::UbmemFusedContext owner;
    Require(runtime.AcquireUbmemFusedContext(output, sizeof(output), &stream, &owner).ok(),
            "acquire lease before failed fence");
    driver.failures = "s";
    Require(runtime.ReleaseUbmemFusedContext(owner, &stream).code() ==
                kvaf::StatusCode::kCompletionUnknown, "failed release fence poisons Runtime");
    Require(runtime.Shutdown().code() == kvaf::StatusCode::kCompletionUnknown,
            "cannot shut down poisoned native lease");
  }
  Require(driver.physical && driver.mapped && driver.reserved,
          "native destructor retains the arena after unknown fused completion");
}

void TestNativePageModeValidation() {
  for (auto options : {FusedOptions(static_cast<kvaf::UbmemPageMode>(42)),
                       FusedOptions(static_cast<kvaf::UbmemPageMode>(0))}) {
    bool rejected = false;
    try {
      kvaf::NpuRuntime runtime(options);
    } catch (const std::invalid_argument& error) {
      rejected = std::string(error.what()).find("page") != std::string::npos;
    }
    Require(rejected, "native Runtime rejects invalid or non-UBmem page configuration");
  }
  Require(kvaf::NpuRuntimeOptions("get", 656, 128).ubmem_page_mode ==
              kvaf::UbmemPageMode::kThp, "default Runtime page mode is THP");
}
#endif

}  // namespace

extern "C" {
aclError aclrtGetDevice(std::int32_t* device) {
  const auto status = Call('d');
  if (status == ACL_SUCCESS) *device = driver.device;
  return status;
}
aclError aclrtSetDevice(std::int32_t device) {
  driver.device = device;
  return ACL_SUCCESS;
}
aclError aclsysGetVersionNum(char* package, std::int32_t* version) {
  Require(std::strcmp(package, "runtime") == 0, "query runtime package version");
  const auto status = Call('v');
  if (status == ACL_SUCCESS) *version = driver.version;
  return status;
}
aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp* prop,
    aclrtMemGranularityOptions option, std::size_t* size) {
  Properties(prop);
  Require(option == ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, "minimum granularity");
  const auto status = Call('g');
  if (status == ACL_SUCCESS) *size = driver.granularity;
  return status;
}
aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, std::size_t size,
    const aclrtPhysicalMemProp* prop, std::uint64_t flags) {
  Properties(prop);
  Require(flags == 0 && size % driver.granularity == 0, "physical allocation geometry");
  const auto status = Call('p');
  if (status == ACL_SUCCESS) {
    *handle = Handle();
    driver.physical = true;
    driver.allocated = size;
  }
  return status;
}
aclError aclrtReserveMemAddress(void** address, std::size_t size, std::size_t align,
    void* expected, std::uint64_t flags) {
  const auto expected_flags = driver.page_mode == kvaf::UbmemPageMode::kBase ? 0U : 1U;
  Require(driver.physical && size == driver.allocated && align == 0 &&
              expected == nullptr && flags == expected_flags,
          "VA reservation page mode matches physical pages");
  const auto status = Call('r');
  if (status == ACL_SUCCESS) {
    *address = Address();
    driver.reserved = true;
  }
  return status;
}
aclError aclrtMapMem(void* address, std::size_t size, std::size_t offset,
    aclrtDrvMemHandle handle, std::uint64_t flags) {
  Require(driver.reserved && address == Address() && size == driver.allocated &&
              offset == 0 && handle == Handle() && flags == 0, "map owned physical region");
  const auto status = Call('m');
  if (status == ACL_SUCCESS) driver.mapped = true;
  return status;
}
aclError aclrtMemSetAccess(void* address, std::size_t size,
    aclrtMemAccessDesc* access, std::size_t count) {
  Require(driver.mapped && address == Address() && size == driver.allocated &&
              count == 1 && access->location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
              access->location.id == static_cast<std::uint32_t>(driver.device) &&
              access->flags == ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
          "grant read/write to runtime user device ID");
  return Call('a');
}
aclError aclrtSynchronizeDevice() { return Call('s'); }
aclError aclrtUnmapMem(void* address) {
  Require(driver.mapped && address == Address(), "unmap live mapping");
  const auto status = Call('u');
  if (status == ACL_SUCCESS) driver.mapped = false;
  return status;
}
aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  Require(!driver.mapped && driver.physical && handle == Handle(), "free only unmapped physical memory");
  const auto status = Call('f');
  if (status == ACL_SUCCESS) driver.physical = false;
  return status;
}
aclError aclrtReleaseMemAddress(void* address) {
  Require(!driver.mapped && !driver.physical && driver.reserved && address == Address(),
          "release VA after physical memory");
  const auto status = Call('x');
  if (status == ACL_SUCCESS) driver.reserved = false;
  return status;
}
aclError aclrtHostMemMapCapabilities(std::uint32_t device, aclrtHacType type,
    aclrtHostMemMapCapability* capability) {
  Require(device == static_cast<std::uint32_t>(driver.device) && type == ACL_RT_HAC_TYPE_AIV,
          "query AIV Host mapping on the selected device");
  const auto status = Call('c');
  if (status == ACL_SUCCESS) *capability = driver.host_map_supported
      ? ACL_RT_HOST_MEM_MAP_SUPPORTED : ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
  return status;
}
aclError aclrtHostRegisterV2(void* host, std::uint64_t bytes, std::uint32_t flags) {
  Require(driver.host_allocated && host == Address() && bytes == driver.allocated &&
              flags == ACL_HOST_REG_MAPPED, "A5 THP registration uses MAPPED only, never PINNED");
  const auto status = Call('R');
  if (status == ACL_SUCCESS) driver.host_registered = true;
  return status;
}
aclError aclrtHostGetDevicePointer(void* host, void** device, std::uint32_t flags) {
  Require(driver.host_registered && host == Address() && flags == 0, "lookup registered Host mapping");
  const auto status = Call('G');
  if (status == ACL_SUCCESS) *device = driver.null_device_address ? nullptr : HostDeviceAddress();
  return status;
}
aclError aclrtHostUnregister(void* host) {
  Require(driver.host_registered && host == Address(), "unregister original Host address");
  const auto status = Call('U');
  if (status == ACL_SUCCESS) driver.host_registered = false;
  return status;
}
}

namespace kvaf::ubmem_host {
bool ThpSupported() { return driver.thp_supported; }
Status AllocateThp(void** address, std::size_t bytes) {
  Require(bytes != 0 && bytes % kThpAlignment == 0, "THP backing rounded to 2 MiB");
  *address = nullptr;
  if (Call('H') != ACL_SUCCESS) return Status::OutOfMemory("host allocation failure");
  *address = ::Address();
  driver.host_allocated = true;
  driver.allocated = bytes;
  return Status::OK();
}
Status AdviseThp(void* address, std::size_t bytes) {
  Require(driver.host_allocated && address == ::Address() && bytes == driver.allocated,
          "huge-page advice covers owned rounded backing");
  return Call('P') == ACL_SUCCESS ? Status::OK() : Status::Internal("madvise failure");
}
Status BindNuma(void* address, std::size_t bytes, int node) {
  Require(address == ::Address() && bytes == driver.allocated && (node == 2 || node == -1), "range-local NUMA node");
  return Call(node == -1 ? 'n' : 'N') == ACL_SUCCESS ? Status::OK() : Status::Internal("mbind failure");
}
void Prefault(void* address, std::size_t bytes) {
  Require(address == ::Address() && bytes == driver.allocated, "prefault whole THP backing");
  (void)Call('T');
}
void Free(void* address) {
  Require(address == ::Address() && driver.host_allocated && !driver.host_registered,
          "free only unregistered Host backing");
  (void)Call('F');
  driver.host_allocated = false;
}
ThpCoverage ReadThpCoverage(void* address, std::size_t bytes) {
  Require(address == ::Address() && bytes == driver.allocated, "measure only current arena");
  ++driver.coverage_reads;
  (void)Call(driver.host_registered ? 'A' : 'B');
  return {true, true, bytes, driver.host_registered ? 0U : bytes};
}
}

#if defined(KVAF_TEST_UBMEM_FUSED_CONTEXT)
namespace kvaf {
namespace {
class FusedTestOperator final : public AccessOperator {
 public:
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
};
}
std::unique_ptr<AccessOperator> CreateUbmemAccessOperator(const AccessOperatorConfig&) {
  return std::make_unique<FusedTestOperator>();
}
}
#endif

int main() {
  try {
    for (const auto mode : {kvaf::UbmemPageMode::kBase}) {
      TestExactCapacityAndRelease(mode);
      TestInvalidRequests(mode);
      TestConstructionFailures(mode);
      TestReleaseFailuresRetainDependencies(mode);
      TestFailedConstructionCleanupIsQuarantined(mode);
#if defined(KVAF_TEST_UBMEM_FUSED_CONTEXT)
      TestNativeFusedContext(mode);
      TestNativeFusedUnknownCompletion(mode);
#endif
    }
    TestThpNumaBinding();
    TestThpAllocationAndRelease();
    TestThpConstructionFailures();
    TestThpCleanupRetention();
    TestThpCoverageParser();
#if defined(KVAF_TEST_UBMEM_FUSED_CONTEXT)
    TestNativeFusedContext(kvaf::UbmemPageMode::kThp);
    TestNativePageModeValidation();
#endif
    std::cout << "PASS: UBmem host arena allocation and failure lifecycle (ACL stubs)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

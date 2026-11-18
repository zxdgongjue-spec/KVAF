#include "ubmem/hal.h"

#include <dlfcn.h>
#include <cstdio>
#include <limits>

namespace kvaf::ubmem {
namespace {
constexpr int kFabric = 1;
constexpr std::uint32_t kHostSide = 0;
constexpr std::uint32_t kHugePage = 1;
constexpr std::uint32_t kGiantPage = 2;
constexpr std::uint32_t kP2pDdr = 3;
constexpr std::uint64_t kHostUva = 1ULL << 10;
constexpr std::size_t kGiantPageBytes = 1ULL << 30;
constexpr int kHalNotSupported = 0xfffe;

template <typename T>
bool Symbol(void* library, const char* name, T* result) {
  *result = reinterpret_cast<T>(dlsym(library, name));
  if (*result == nullptr) std::fprintf(stderr, "[KVAF_UBMEM] missing HAL symbol %s\n", name);
  return *result != nullptr;
}
}  // namespace

Status HalError(const char* operation, int result) {
  const std::string message = std::string(operation) + " failed, ret=" + std::to_string(result);
  std::fprintf(stderr, "[KVAF_UBMEM] %s\n", message.c_str());
  return Status::Internal(message);
}

Hal::~Hal() { if (library_ != nullptr) dlclose(library_); }

Status Hal::Load(const std::string& path) {
  if (library_ != nullptr) return path == path_ ? Status::OK()
      : Status::InvalidState("HAL is already loaded from a different library");
  library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr) {
    const std::string message = "cannot load " + path + ": " + dlerror();
    std::fprintf(stderr, "[KVAF_UBMEM] %s\n", message.c_str());
    return Status::NotFound(message);
  }
  if (!Symbol(library_, "halMemCreate", &create) ||
      !Symbol(library_, "halMemRelease", &release) ||
      !Symbol(library_, "halMemAddressReserve", &reserve) ||
      !Symbol(library_, "halMemAddressFree", &free_address) ||
      !Symbol(library_, "halMemMap", &map) ||
      !Symbol(library_, "halMemUnmap", &unmap) ||
      !Symbol(library_, "halMemExportToShareableHandleV2", &export_memory) ||
      !Symbol(library_, "halMemTransShareableHandle", &translate_handle) ||
      !Symbol(library_, "halMemShareHandleSetAttribute", &share_attribute)) {
    dlclose(library_);
    library_ = nullptr;
    return Status::NotFound("HAL lacks required Fabric/VMM interfaces");
  }
  path_ = path;
  return Status::OK();
}

Mapping::~Mapping() { (void)Close(); }

Status Mapping::ReserveAndMap(std::uint64_t page_flag) {
  int ret = hal_->reserve(&address_, bytes_, 0, nullptr, page_flag | kHostUva);
  if (ret != 0) return HalError("reserve virtual address", ret);
  ret = hal_->map(address_, bytes_, 0, handle_, 0);
  if (ret != 0) return HalError("halMemMap", ret);
  mapped_ = true;
  return Status::OK();
}

Status Mapping::AllocateHost(std::uint64_t requested_bytes, FabricHandle* descriptor,
                             bool giant_pages) {
  const std::size_t page_bytes = giant_pages ? kGiantPageBytes : kPageBytes;
  if (handle_ != nullptr || descriptor == nullptr || requested_bytes == 0 ||
      requested_bytes > std::numeric_limits<std::size_t>::max() - (page_bytes - 1)) {
    return Status::InvalidArgument("invalid UBmem Host P2P DDR allocation size or mapping state");
  }
  const std::size_t mapping_bytes = (requested_bytes + page_bytes - 1) / page_bytes * page_bytes;
  const auto page_type = giant_pages ? kGiantPage : kHugePage;
  const MemoryProperty prop{kHostSide, 0, 0, page_type, kP2pDdr, 0};
  int ret = hal_->create(&handle_, mapping_bytes, &prop, 0);
  if (giant_pages && ret == kHalNotSupported) {
    const std::string message = "1 GiB UBmem Host P2P DDR is unsupported by this driver or platform "
        "(halMemCreate ret=" + std::to_string(ret) + "). Use --ubmem-page-size 2m.";
    std::fprintf(stderr, "[KVAF_UBMEM] %s\n", message.c_str());
    return Status::NotImplemented(message);
  }
  if (ret != 0) {
    return HalError(giant_pages ? "halMemCreate(Host P2P DDR, 1 GiB pages)"
                               : "halMemCreate(Host P2P DDR)", ret);
  }
  bytes_ = mapping_bytes;
  Status status = ReserveAndMap(page_type);
  if (!status.ok()) return status;
  ret = hal_->export_memory(handle_, kFabric, 0, descriptor);
  if (ret != 0) return HalError("halMemExportToShareableHandleV2(FABRIC)", ret);
  std::uint32_t server = 0;
  std::uint64_t shareable = 0;
  ret = hal_->translate_handle(kFabric, descriptor, &server, &shareable);
  if (ret != 0) return HalError("halMemTransShareableHandle", ret);
  // Same policy as SHMEM: allow same-server import without a PID allowlist.
  // Cross-server validation and the Fabric descriptor remain driver-owned.
  Hal::ShareAttribute attr{};
  attr.enable = 1;
  ret = hal_->share_attribute(shareable, 0, attr);
  return ret == 0 ? Status::OK() : HalError("halMemShareHandleSetAttribute", ret);
}

Status Mapping::Close() {
  int ret;
  if (mapped_) {
    ret = hal_->unmap(address_);
    if (ret != 0) return HalError("halMemUnmap", ret);
    mapped_ = false;
  }
  if (address_ != nullptr) {
    ret = hal_->free_address(address_);
    if (ret != 0) return HalError("release virtual address", ret);
    address_ = nullptr;
  }
  if (handle_ != nullptr) {
    ret = hal_->release(handle_);
    if (ret != 0) return HalError("halMemRelease", ret);
    handle_ = nullptr;
  }
  bytes_ = 0;
  return Status::OK();
}
}  // namespace kvaf::ubmem

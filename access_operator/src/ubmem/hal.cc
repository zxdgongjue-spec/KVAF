#include "ubmem/hal.h"

#include <dlfcn.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>

namespace kvaf::ubmem {
namespace {
constexpr int kFabric = 1;
constexpr std::uint64_t kHugePage = 1;
constexpr std::uint64_t kHostUva = 1ULL << 10;

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

Status Mapping::ReserveAndMap() {
  int ret = hal_->reserve(&address_, bytes_, 0, nullptr, kHugePage | kHostUva);
  if (ret != 0) return HalError("reserve virtual address", ret);
  ret = hal_->map(address_, bytes_, 0, handle_, 0);
  if (ret != 0) return HalError("halMemMap", ret);
  mapped_ = true;
  return Status::OK();
}

Status Mapping::AllocateHost(std::uint64_t bytes, FabricHandle* descriptor) {
  if (handle_ != nullptr || descriptor == nullptr || bytes == 0 || bytes % kPageBytes != 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("Host DDR requires a 2-MiB-aligned size");
  }
  // MEM_HOST_SIDE, MEM_HUGE_PAGE_TYPE, MEM_P2P_DDR_TYPE.
  MemoryProperty prop{0, 0, 0, 1, 3, 0};
  // HAL status codes are not errno values. Capture errno immediately, before
  // logging or cleanup can replace it; some HAL failures leave it at zero.
  errno = 0;
  int ret = hal_->create(&handle_, bytes, &prop, 0);
  const int create_errno = errno;
  if (ret != 0) {
    const std::string message = "halMemCreate(Host P2P DDR) failed, ret=" +
        std::to_string(ret) + "; bytes=" + std::to_string(bytes) +
        "; cpu_service_pid=" + std::to_string(getpid()) +
        "; errno=" + std::to_string(create_errno) + " (" +
        (create_errno ? std::strerror(create_errno) : "not supplied by HAL") + ")";
    std::fprintf(stderr, "[KVAF_UBMEM] %s\n", message.c_str());
    return Status::Internal(message);
  }
  bytes_ = bytes;
  Status status = ReserveAndMap();
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

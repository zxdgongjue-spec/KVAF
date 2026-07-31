#include "kvaf/access_operator/a5/host_ddr_mapping.h"

#include <dlfcn.h>

#include <string>

namespace kvaf::access::a5 {
namespace {

constexpr std::uint32_t kHostMemMapDevice = 3;
constexpr std::uint64_t kHalMemHost = 0x2ULL << 10;
constexpr std::uint64_t kHalMemTypeDdr = 0x0ULL << 14;
constexpr std::uint64_t kHalMemPageNormal = 0x0ULL << 17;
constexpr std::uint64_t kA5HostDdrAllocFlags =
    kHalMemHost | kHalMemTypeDdr | kHalMemPageNormal;
constexpr const char* kAscendHalLibrary = "libascend_hal.so";

}  // namespace

struct HostDdrDeviceMapping::Impl {
  using HostRegister = int (*)(void*, std::uint64_t, std::uint32_t,
                               std::uint32_t, void**);
  using HostUnregister = int (*)(void*, std::uint32_t, std::uint32_t);
  using MemAlloc = int (*)(void**, std::uint64_t, std::uint64_t);
  using MemFree = int (*)(void*);

  void* library{nullptr};
  HostRegister host_register{nullptr};
  HostUnregister host_unregister{nullptr};
  MemAlloc mem_alloc{nullptr};
  MemFree mem_free{nullptr};
  Address host_address{0};
  Address device_address{0};
  std::uint64_t bytes{0};
  std::uint32_t logical_device_id{0};
  bool registered{false};
  bool owns_allocation{false};
};

HostDdrDeviceMapping::HostDdrDeviceMapping()
    : impl_(std::make_unique<Impl>()) {}

HostDdrDeviceMapping::~HostDdrDeviceMapping() {
  (void)Close();
}

Status HostDdrDeviceMapping::LoadHal(bool require_allocator) {
  impl_->library = dlopen(kAscendHalLibrary, RTLD_NOW | RTLD_LOCAL);
  if (impl_->library == nullptr) {
    const char* error = dlerror();
    return Status::Error(
        StatusCode::kUnavailable,
        std::string("dlopen(libascend_hal.so) failed: ") +
            (error == nullptr ? "unknown dlopen error" : error));
  }

  impl_->host_register = reinterpret_cast<Impl::HostRegister>(
      dlsym(impl_->library, "halHostRegister"));
  impl_->host_unregister = reinterpret_cast<Impl::HostUnregister>(
      dlsym(impl_->library, "halHostUnregisterEx"));
  if (require_allocator) {
    impl_->mem_alloc = reinterpret_cast<Impl::MemAlloc>(
        dlsym(impl_->library, "halMemAlloc"));
    impl_->mem_free = reinterpret_cast<Impl::MemFree>(
        dlsym(impl_->library, "halMemFree"));
  }
  if (impl_->host_register == nullptr || impl_->host_unregister == nullptr ||
      (require_allocator &&
       (impl_->mem_alloc == nullptr || impl_->mem_free == nullptr))) {
    const char* error = dlerror();
    dlclose(impl_->library);
    impl_->library = nullptr;
    impl_->host_register = nullptr;
    impl_->host_unregister = nullptr;
    impl_->mem_alloc = nullptr;
    impl_->mem_free = nullptr;
    return Status::Error(
        StatusCode::kUnavailable,
        std::string("required A5 Host DDR symbol is unavailable: ") +
            (error == nullptr ? "unknown dlsym error" : error));
  }
  return Status::OK();
}

Status HostDdrDeviceMapping::AllocateAndRegister(
    std::uint64_t bytes, std::uint32_t logical_device_id) {
  if (impl_->registered || bytes == 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "Host DDR mapping is already active or invalid");
  }
  auto status = LoadHal(true);
  if (!status.ok()) {
    return status;
  }
  void* host_address = nullptr;
  const auto alloc_result =
      impl_->mem_alloc(&host_address, bytes, kA5HostDdrAllocFlags);
  if (host_address != nullptr) {
    impl_->host_address = reinterpret_cast<Address>(host_address);
    impl_->bytes = bytes;
    impl_->owns_allocation = true;
  }
  if (alloc_result != 0 || host_address == nullptr) {
    (void)Close();
    return Status::Error(StatusCode::kOutOfMemory,
                         "halMemAlloc(A5 Host DDR) failed, ret=" +
                             std::to_string(alloc_result));
  }

  void* device_address = nullptr;
  const auto result = impl_->host_register(
      host_address, bytes, kHostMemMapDevice,
      logical_device_id, &device_address);
  impl_->device_address = reinterpret_cast<Address>(device_address);
  impl_->logical_device_id = logical_device_id;
  impl_->registered = result == 0 || device_address != nullptr;
  if (result != 0 || device_address == nullptr) {
    (void)Close();
    return Status::Error(
        StatusCode::kTransportError,
        "halHostRegister(HOST_MEM_MAP_DEV) failed, ret=" +
            std::to_string(result));
  }

  return Status::OK();
}

Status HostDdrDeviceMapping::Register(
    Address host_address, std::uint64_t bytes,
    std::uint32_t logical_device_id) {
  if (impl_->registered || host_address == 0 || bytes == 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "Host DDR mapping is already active or invalid");
  }
  auto status = LoadHal(false);
  if (!status.ok()) {
    return status;
  }

  void* device_address = nullptr;
  const auto result = impl_->host_register(
      reinterpret_cast<void*>(host_address), bytes, kHostMemMapDevice,
      logical_device_id, &device_address);
  impl_->host_address = host_address;
  impl_->device_address = reinterpret_cast<Address>(device_address);
  impl_->bytes = bytes;
  impl_->logical_device_id = logical_device_id;
  impl_->registered = result == 0 || device_address != nullptr;
  if (result != 0 || device_address == nullptr) {
    (void)Close();
    return Status::Error(
        StatusCode::kTransportError,
        "halHostRegister(HOST_MEM_MAP_DEV) failed, ret=" +
            std::to_string(result));
  }

  return Status::OK();
}

Address HostDdrDeviceMapping::host_address() const {
  return impl_->host_address;
}

Address HostDdrDeviceMapping::device_address() const {
  return impl_->device_address;
}

std::uint64_t HostDdrDeviceMapping::bytes() const {
  return impl_->bytes;
}

Status HostDdrDeviceMapping::Close() {
  if (impl_->registered) {
    const auto result = impl_->host_unregister(
        reinterpret_cast<void*>(impl_->host_address),
        impl_->logical_device_id, kHostMemMapDevice);
    if (result != 0) {
      // Keep the allocation and the HAL handle alive. Freeing pages that the
      // device may still reference would turn a teardown error into DMA
      // use-after-free; the caller may retry Close.
      return Status::Error(
          StatusCode::kTransportError,
          "halHostUnregisterEx(HOST_MEM_MAP_DEV) failed, ret=" +
              std::to_string(result));
    }
    impl_->registered = false;
  }

  if (impl_->owns_allocation && impl_->host_address != 0) {
    const auto result =
        impl_->mem_free(reinterpret_cast<void*>(impl_->host_address));
    if (result != 0) {
      return Status::Error(StatusCode::kTransportError,
                           "halMemFree(A5 Host DDR) failed, ret=" +
                               std::to_string(result));
    }
  }
  impl_->owns_allocation = false;
  impl_->host_address = 0;
  impl_->device_address = 0;
  impl_->bytes = 0;
  if (impl_->library != nullptr) {
    dlclose(impl_->library);
    impl_->library = nullptr;
  }
  impl_->host_register = nullptr;
  impl_->host_unregister = nullptr;
  impl_->mem_alloc = nullptr;
  impl_->mem_free = nullptr;
  return Status::OK();
}

}  // namespace kvaf::access::a5

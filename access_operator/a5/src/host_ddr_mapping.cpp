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
constexpr const char* kAscendHalLibrary =
    "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so";

}  // namespace

struct HostDdrDeviceMapping::Impl {
  using HostRegister = int (*)(void*, std::uint64_t, std::uint32_t,
                               std::uint32_t, void**);
  using HostUnregister = int (*)(void*, std::uint32_t, std::uint32_t);
  using MemAlloc = int (*)(void**, std::uint64_t, std::uint64_t);
  using MemFree = int (*)(void*);

  enum class HostAddressOwnership {
    kNone,
    kBorrowed,
    kAllocated,
  };

  bool empty() const {
    return library == nullptr && host_register == nullptr &&
           host_unregister == nullptr && mem_alloc == nullptr &&
           mem_free == nullptr && host_address == 0 && device_address == 0 &&
           bytes == 0 && logical_device_id == 0 && !registration_may_exist &&
           ownership == HostAddressOwnership::kNone;
  }

  void* library{nullptr};
  HostRegister host_register{nullptr};
  HostUnregister host_unregister{nullptr};
  MemAlloc mem_alloc{nullptr};
  MemFree mem_free{nullptr};
  Address host_address{0};
  Address device_address{0};
  std::uint64_t bytes{0};
  std::uint32_t logical_device_id{0};
  // A non-null device address on an error return is treated as a possibly
  // live registration too.  Close must disprove that state before the host
  // backing can be released or reused.
  bool registration_may_exist{false};
  HostAddressOwnership ownership{HostAddressOwnership::kNone};
};

HostDdrDeviceMapping::HostDdrDeviceMapping()
    : impl_(std::make_unique<Impl>()) {}

HostDdrDeviceMapping::~HostDdrDeviceMapping() {
  (void)Close();
}

Status HostDdrDeviceMapping::LoadHal(bool require_allocator) {
  if (!impl_->empty()) {
    return Status::Error(
        StatusCode::kInternal,
        "cannot load Ascend HAL over a retained Host DDR mapping");
  }

  void* library = dlopen(kAscendHalLibrary, RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    const char* error = dlerror();
    return Status::Error(
        StatusCode::kUnavailable,
        std::string("dlopen(pinned libascend_hal.so) failed: ") +
            (error == nullptr ? "unknown dlopen error" : error));
  }

  auto host_register = reinterpret_cast<Impl::HostRegister>(
      dlsym(library, "halHostRegister"));
  auto host_unregister = reinterpret_cast<Impl::HostUnregister>(
      dlsym(library, "halHostUnregisterEx"));
  Impl::MemAlloc mem_alloc = nullptr;
  Impl::MemFree mem_free = nullptr;
  if (require_allocator) {
    mem_alloc = reinterpret_cast<Impl::MemAlloc>(
        dlsym(library, "halMemAlloc"));
    mem_free = reinterpret_cast<Impl::MemFree>(
        dlsym(library, "halMemFree"));
  }
  if (host_register == nullptr || host_unregister == nullptr ||
      (require_allocator && (mem_alloc == nullptr || mem_free == nullptr))) {
    const char* error = dlerror();
    const std::string detail =
        error == nullptr ? "unknown dlsym error" : error;
    if (dlclose(library) != 0) {
      // Preserve the handle so the normal retryable Close path owns it. A
      // failed dlclose must not disappear behind an apparently empty object.
      impl_->library = library;
      impl_->host_register = host_register;
      impl_->host_unregister = host_unregister;
      impl_->mem_alloc = mem_alloc;
      impl_->mem_free = mem_free;
      return Status::Error(
          StatusCode::kUnavailable,
          "required A5 Host DDR symbol is unavailable and dlclose failed: " +
              detail);
    }
    return Status::Error(
        StatusCode::kUnavailable,
        "required A5 Host DDR symbol is unavailable: " + detail);
  }

  // Commit the resolved table only after every requested symbol is present.
  // This keeps a failed LoadHal from creating a half-initialized reusable
  // object.
  impl_->library = library;
  impl_->host_register = host_register;
  impl_->host_unregister = host_unregister;
  impl_->mem_alloc = mem_alloc;
  impl_->mem_free = mem_free;
  return Status::OK();
}

Status HostDdrDeviceMapping::AllocateAndRegister(
    std::uint64_t bytes, std::uint32_t logical_device_id) {
  if (bytes == 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "Host DDR mapping size must be non-zero");
  }
  if (!impl_->empty()) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "Host DDR mapping still owns retained state; Close must succeed "
        "before AllocateAndRegister can be called again");
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
    impl_->ownership = Impl::HostAddressOwnership::kAllocated;
  }
  if (alloc_result != 0 || host_address == nullptr) {
    const auto cleanup_status = Close();
    return Status::Error(StatusCode::kOutOfMemory,
                         "halMemAlloc(A5 Host DDR) failed, ret=" +
                             std::to_string(alloc_result) +
                             (cleanup_status.ok()
                                  ? ""
                                  : "; cleanup incomplete and allocation "
                                    "retained for Close retry: " +
                                        cleanup_status.message()));
  }

  void* device_address = nullptr;
  const auto result = impl_->host_register(
      host_address, bytes, kHostMemMapDevice,
      logical_device_id, &device_address);
  impl_->device_address = reinterpret_cast<Address>(device_address);
  impl_->logical_device_id = logical_device_id;
  impl_->registration_may_exist = result == 0 || device_address != nullptr;
  if (result != 0 || device_address == nullptr) {
    const auto cleanup_status = Close();
    return Status::Error(
        StatusCode::kTransportError,
        "halHostRegister(HOST_MEM_MAP_DEV) failed, ret=" +
            std::to_string(result) +
            (cleanup_status.ok()
                 ? ""
                 : "; cleanup incomplete and mapping retained for Close "
                   "retry: " +
                       cleanup_status.message()));
  }

  return Status::OK();
}

Status HostDdrDeviceMapping::Register(
    Address host_address, std::uint64_t bytes,
    std::uint32_t logical_device_id) {
  if (host_address == 0 || bytes == 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "Host DDR mapping address and size must be non-zero");
  }
  if (!impl_->empty()) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "Host DDR mapping still owns retained state; Close must succeed "
        "before Register can be called again");
  }
  auto status = LoadHal(false);
  if (!status.ok()) {
    return status;
  }

  // Record borrowed ownership before registration.  A later cleanup failure
  // must never inherit kAllocated from an older operation and free memory
  // supplied by the caller.
  impl_->host_address = host_address;
  impl_->bytes = bytes;
  impl_->logical_device_id = logical_device_id;
  impl_->ownership = Impl::HostAddressOwnership::kBorrowed;

  void* device_address = nullptr;
  const auto result = impl_->host_register(
      reinterpret_cast<void*>(host_address), bytes, kHostMemMapDevice,
      logical_device_id, &device_address);
  impl_->device_address = reinterpret_cast<Address>(device_address);
  impl_->registration_may_exist = result == 0 || device_address != nullptr;
  if (result != 0 || device_address == nullptr) {
    const auto cleanup_status = Close();
    return Status::Error(
        StatusCode::kTransportError,
        "halHostRegister(HOST_MEM_MAP_DEV) failed, ret=" +
            std::to_string(result) +
            (cleanup_status.ok()
                 ? ""
                 : "; cleanup incomplete and mapping retained for Close "
                   "retry: " +
                       cleanup_status.message()));
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

bool HostDdrDeviceMapping::device_mapping_removed() const {
  return !impl_->registration_may_exist;
}

Status HostDdrDeviceMapping::Close() {
  if (impl_->registration_may_exist) {
    if (impl_->host_unregister == nullptr || impl_->host_address == 0) {
      return Status::Error(
          StatusCode::kInternal,
          "Host DDR registration state is retained without an unregister "
          "handle");
    }
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
    impl_->registration_may_exist = false;
    impl_->device_address = 0;
  }

  if (impl_->ownership == Impl::HostAddressOwnership::kAllocated) {
    if (impl_->mem_free == nullptr || impl_->host_address == 0) {
      return Status::Error(
          StatusCode::kInternal,
          "allocated Host DDR state is retained without a valid free "
          "handle");
    }
    const auto result =
        impl_->mem_free(reinterpret_cast<void*>(impl_->host_address));
    if (result != 0) {
      return Status::Error(StatusCode::kTransportError,
                           "halMemFree(A5 Host DDR) failed, ret=" +
                               std::to_string(result));
    }
    impl_->host_address = 0;
    impl_->ownership = Impl::HostAddressOwnership::kNone;
  } else if (impl_->ownership == Impl::HostAddressOwnership::kBorrowed) {
    // Registration is gone, so the caller-owned address is no longer part of
    // this object's lifetime.
    impl_->host_address = 0;
    impl_->ownership = Impl::HostAddressOwnership::kNone;
  }

  impl_->device_address = 0;
  impl_->bytes = 0;
  impl_->logical_device_id = 0;
  if (impl_->library != nullptr) {
    if (dlclose(impl_->library) != 0) {
      const char* error = dlerror();
      return Status::Error(
          StatusCode::kInternal,
          std::string("dlclose(libascend_hal.so) failed: ") +
              (error == nullptr ? "unknown dlclose error" : error));
    }
    impl_->library = nullptr;
  }
  impl_->host_register = nullptr;
  impl_->host_unregister = nullptr;
  impl_->mem_alloc = nullptr;
  impl_->mem_free = nullptr;
  return Status::OK();
}

}  // namespace kvaf::access::a5

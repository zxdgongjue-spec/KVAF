#include "kvaf/access_operator/a5/hcomm_urma_resources.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "hcomm/hcomm_res.h"
#if __has_include("hcomm/hcomm_res_defs.h")
#include "hcomm/hcomm_res_defs.h"
#elif __has_include("hcomm/hcomm_res_entity_defs.h")
#include "hcomm/hcomm_res_entity_defs.h"
#else
#error "KVAF requires hcomm_res_defs.h or hcomm_res_entity_defs.h"
#endif

namespace kvaf::access::a5 {
namespace {

constexpr std::int32_t kChannelReady = 0;
constexpr std::int32_t kChannelConnecting = 1;
constexpr std::uint32_t kPollIntervalMs = 10;
constexpr std::uint32_t kHostMemMapDevice = 3;
constexpr std::uint32_t kDefaultQos = 4;
constexpr std::uint64_t kHalMemHost = 0x2ULL << 10;
constexpr std::uint64_t kHalMemTypeDdr = 0x0ULL << 14;
constexpr std::uint64_t kHalMemPageNormal = 0x0ULL << 17;
constexpr std::uint64_t kA5HostDdrAllocFlags =
    kHalMemHost | kHalMemTypeDdr | kHalMemPageNormal;
constexpr const char* kAscendHalLibrary = "libascend_hal.so";

template <typename T, typename = void>
struct HasQos : std::false_type {};

template <typename T>
struct HasQos<T, std::void_t<decltype(std::declval<T&>().qos)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasHccsQos : std::false_type {};

template <typename T>
struct HasHccsQos<
    T, std::void_t<decltype(std::declval<T&>().hccsAttr.qos)>>
    : std::true_type {};

template <typename T>
void SetDefaultQos(T* desc) {
  if constexpr (HasQos<T>::value) {
    desc->qos = kDefaultQos;
  } else if constexpr (HasHccsQos<T>::value) {
    desc->hccsAttr.qos = kDefaultQos;
  }
}

Status HcommFailure(const char* operation, std::int32_t result) {
  return Status::Error(StatusCode::kTransportError,
                       std::string(operation) + " failed, HCOMM ret=" +
                           std::to_string(result));
}

Status HcommInvalidOutput(const char* operation) {
  return Status::Error(
      StatusCode::kTransportError,
      std::string(operation) + " returned success with an invalid output");
}

Status BuildEndpointDesc(const EndpointSpec& spec, EndpointDesc* desc) {
  if (desc == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "endpoint descriptor output is null");
  }
  bool has_nonzero_eid_byte = false;
  for (const auto byte : spec.eid) {
    has_nonzero_eid_byte = has_nonzero_eid_byte || byte != 0;
  }
  if (!has_nonzero_eid_byte) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "endpoint EID must not be all zero");
  }
  const auto init_result = EndpointDescInit(desc, 1);
  if (init_result != 0) {
    return HcommFailure("EndpointDescInit", init_result);
  }

  desc->protocol = COMM_PROTOCOL_UBC_CTP;
  desc->commAddr.type = COMM_ADDR_TYPE_EID;
  std::memcpy(desc->commAddr.eid, spec.eid.data(), spec.eid.size());
  if (spec.location == EndpointLocation::kHost) {
    desc->loc.locType = ENDPOINT_LOC_TYPE_HOST;
    desc->loc.host.id = spec.physical_device_id;
  } else {
    desc->loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc->loc.device.devPhyId = spec.physical_device_id;
    desc->loc.device.superDevId = spec.super_device_id;
    desc->loc.device.serverIdx = spec.server_index;
    desc->loc.device.superPodIdx = spec.super_pod_index;
  }
  return Status::OK();
}

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

Status HostDdrDeviceMapping::Register(Address host_address,
                                      std::uint64_t bytes,
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
  Status status = Status::OK();
  if (impl_->registered) {
    const auto result = impl_->host_unregister(
        reinterpret_cast<void*>(impl_->host_address),
        impl_->logical_device_id, kHostMemMapDevice);
    if (result != 0) {
      status = Status::Error(
          StatusCode::kTransportError,
          "halHostUnregisterEx(HOST_MEM_MAP_DEV) failed, ret=" +
              std::to_string(result));
    }
  }
  impl_->registered = false;
  if (impl_->owns_allocation && impl_->host_address != 0) {
    const auto result =
        impl_->mem_free(reinterpret_cast<void*>(impl_->host_address));
    if (result != 0 && status.ok()) {
      status = Status::Error(StatusCode::kTransportError,
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
  return status;
}

struct HcommUrmaResources::Impl {
  struct Registration {
    HcommMemHandle handle{nullptr};
  };

  EndpointHandle endpoint{nullptr};
  ChannelHandle channel{0};
  std::vector<Registration> registrations;
  std::vector<std::vector<std::uint8_t>> imported_descriptors;
  bool closed{false};
};

HcommUrmaResources::HcommUrmaResources() : impl_(std::make_unique<Impl>()) {}

HcommUrmaResources::~HcommUrmaResources() {
  (void)Close();
}

Status HcommUrmaResources::OpenEndpoint(const EndpointSpec& local) {
  if (impl_->closed || impl_->endpoint != nullptr) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "HCOMM endpoint is already opened or closed");
  }
  EndpointDesc desc{};
  auto status = BuildEndpointDesc(local, &desc);
  if (!status.ok()) {
    return status;
  }
  EndpointHandle endpoint = nullptr;
  const auto result = HcommEndpointCreate(&desc, &endpoint);
  if (result != 0) {
    if (endpoint != nullptr) {
      (void)HcommEndpointDestroy(endpoint);
    }
    return HcommFailure("HcommEndpointCreate", result);
  }
  if (endpoint == nullptr) {
    return HcommInvalidOutput("HcommEndpointCreate");
  }
  impl_->endpoint = endpoint;
  return Status::OK();
}

Status HcommUrmaResources::RegisterMemory(
    const std::string& tag, Address address, std::uint64_t bytes,
    MemoryLocation location, std::uint32_t* registration_id) {
  if (impl_->endpoint == nullptr || registration_id == nullptr || tag.empty() ||
      address == 0 || bytes == 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "invalid endpoint, tag, address, size, or output id");
  }

  CommMem memory{};
  memory.type = location == MemoryLocation::kDeviceHbm
                    ? COMM_MEM_TYPE_DEVICE
                    : COMM_MEM_TYPE_HOST;
  memory.addr = reinterpret_cast<void*>(address);
  memory.size = bytes;

  HcommMemHandle handle = nullptr;
  const auto result =
      HcommMemReg(impl_->endpoint, tag.c_str(), &memory, &handle);
  if (result != 0) {
    if (handle != nullptr) {
      (void)HcommMemUnreg(impl_->endpoint, handle);
    }
    return HcommFailure("HcommMemReg", result);
  }
  if (handle == nullptr) {
    return HcommInvalidOutput("HcommMemReg");
  }
  try {
    impl_->registrations.push_back({handle});
  } catch (const std::bad_alloc&) {
    (void)HcommMemUnreg(impl_->endpoint, handle);
    return Status::Error(StatusCode::kOutOfMemory,
                         "tracking the HCOMM registration failed");
  }
  *registration_id =
      static_cast<std::uint32_t>(impl_->registrations.size() - 1);
  return Status::OK();
}

Status HcommUrmaResources::ExportMemory(
    std::uint32_t registration_id,
    std::vector<std::uint8_t>* descriptor) const {
  if (impl_->endpoint == nullptr || descriptor == nullptr ||
      registration_id >= impl_->registrations.size()) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "invalid endpoint, registration id, or descriptor");
  }
  void* raw_descriptor = nullptr;
  std::uint32_t raw_bytes = 0;
  const auto result =
      HcommMemExport(impl_->endpoint,
                     impl_->registrations[registration_id].handle,
                     &raw_descriptor, &raw_bytes);
  if (result != 0) {
    return HcommFailure("HcommMemExport", result);
  }
  if (raw_descriptor == nullptr || raw_bytes == 0) {
    return HcommInvalidOutput("HcommMemExport");
  }
  const auto* begin = static_cast<const std::uint8_t*>(raw_descriptor);
  try {
    descriptor->assign(begin, begin + raw_bytes);
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kOutOfMemory,
                         "copying the HCOMM export descriptor failed");
  }
  return Status::OK();
}

Status HcommUrmaResources::ImportMemory(
    const std::vector<std::uint8_t>& descriptor,
    ImportedMemoryView* view) {
  if (impl_->endpoint == nullptr || descriptor.empty() || view == nullptr ||
      descriptor.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "invalid endpoint, descriptor, or imported view");
  }
  CommMem memory{};
  const auto result =
      HcommMemImport(impl_->endpoint, descriptor.data(),
                     static_cast<std::uint32_t>(descriptor.size()), &memory);
  if (result != 0) {
    if (memory.addr != nullptr || memory.size != 0 ||
        memory.type != COMM_MEM_TYPE_INVALID) {
      (void)HcommMemUnimport(
          impl_->endpoint, descriptor.data(),
          static_cast<std::uint32_t>(descriptor.size()));
    }
    return HcommFailure("HcommMemImport", result);
  }
  if (memory.addr == nullptr || memory.size == 0 ||
      (memory.type != COMM_MEM_TYPE_HOST &&
       memory.type != COMM_MEM_TYPE_DEVICE)) {
    const auto rollback_result =
        HcommMemUnimport(impl_->endpoint, descriptor.data(),
                         static_cast<std::uint32_t>(descriptor.size()));
    std::string message =
        "HcommMemImport returned a null, empty, or invalid memory view";
    if (rollback_result != 0) {
      message += "; HcommMemUnimport rollback failed, ret=" +
                 std::to_string(rollback_result);
    }
    return Status::Error(StatusCode::kTransportError, message);
  }

  ImportedMemoryView imported_view{};
  imported_view.address = reinterpret_cast<Address>(memory.addr);
  imported_view.bytes = memory.size;
  imported_view.location = memory.type == COMM_MEM_TYPE_DEVICE
                               ? MemoryLocation::kDeviceHbm
                               : MemoryLocation::kHostDdr;
  try {
    impl_->imported_descriptors.push_back(descriptor);
  } catch (const std::bad_alloc&) {
    const auto rollback_result =
        HcommMemUnimport(impl_->endpoint, descriptor.data(),
                         static_cast<std::uint32_t>(descriptor.size()));
    if (rollback_result != 0) {
      return HcommFailure("HcommMemUnimport rollback", rollback_result);
    }
    return Status::Error(StatusCode::kOutOfMemory,
                         "tracking the HCOMM import descriptor failed");
  }
  *view = imported_view;
  return Status::OK();
}

Status HcommUrmaResources::CreateChannel(const EndpointSpec& peer,
                                         ChannelEngine engine,
                                         ChannelRole role) {
  if (impl_->endpoint == nullptr || impl_->channel != 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "endpoint is absent or channel already exists");
  }

  HcommChannelDesc desc{};
  const auto init_result = HcommChannelDescInit(&desc, 1);
  if (init_result != 0) {
    return HcommFailure("HcommChannelDescInit", init_result);
  }
  auto status = BuildEndpointDesc(peer, &desc.remoteEndpoint);
  if (!status.ok()) {
    return status;
  }
  desc.notifyNum = 0;
  desc.exchangeAllMems = true;
  desc.memHandles = nullptr;
  desc.memHandleNum = 0;
  desc.role = role == ChannelRole::kClient ? HCOMM_SOCKET_ROLE_CLIENT
                                           : HCOMM_SOCKET_ROLE_SERVER;
  desc.port = 0;
  SetDefaultQos(&desc);

  const auto hcomm_engine =
      engine == ChannelEngine::kAiv ? COMM_ENGINE_AIV : COMM_ENGINE_CPU;
  ChannelHandle channel = 0;
  const auto result =
      HcommChannelCreate(impl_->endpoint, hcomm_engine, &desc, 1,
                         &channel);
  if (result != 0) {
    if (channel != 0) {
      (void)HcommChannelDestroy(&channel, 1);
    }
    return HcommFailure("HcommChannelCreate", result);
  }
  if (channel == 0) {
    return HcommInvalidOutput("HcommChannelCreate");
  }
  impl_->channel = channel;
  return Status::OK();
}

Status HcommUrmaResources::WaitChannelReady(std::uint32_t timeout_ms) {
  if (impl_->channel == 0) {
    return Status::Error(StatusCode::kNotInitialized,
                         "HCOMM channel has not been created");
  }
  std::uint32_t elapsed_ms = 0;
  while (elapsed_ms <= timeout_ms) {
    std::int32_t channel_status = kChannelConnecting;
    const auto result =
        HcommChannelGetStatus(&impl_->channel, 1, &channel_status);
    if (result != 0) {
      return HcommFailure("HcommChannelGetStatus", result);
    }
    if (channel_status == kChannelReady) {
      return Status::OK();
    }
    if (channel_status != kChannelConnecting) {
      return Status::Error(
          StatusCode::kTransportError,
          "HCOMM channel entered terminal status " +
              std::to_string(channel_status));
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kPollIntervalMs));
    elapsed_ms += kPollIntervalMs;
  }
  return Status::Error(StatusCode::kTimeout,
                       "timed out waiting for HCOMM channel readiness");
}

std::uint64_t HcommUrmaResources::channel() const {
  return impl_->channel;
}

Status HcommUrmaResources::Close() {
  if (impl_->closed) {
    return Status::OK();
  }
  Status first_error = Status::OK();

  if (impl_->channel != 0) {
    const auto result = HcommChannelDestroy(&impl_->channel, 1);
    if (result != 0 && first_error.ok()) {
      first_error = HcommFailure("HcommChannelDestroy", result);
    }
    impl_->channel = 0;
  }
  if (impl_->endpoint != nullptr) {
    for (auto it = impl_->imported_descriptors.rbegin();
         it != impl_->imported_descriptors.rend(); ++it) {
      const auto result =
          HcommMemUnimport(impl_->endpoint, it->data(),
                           static_cast<std::uint32_t>(it->size()));
      if (result != 0 && first_error.ok()) {
        first_error = HcommFailure("HcommMemUnimport", result);
      }
    }
    impl_->imported_descriptors.clear();

    for (auto it = impl_->registrations.rbegin();
         it != impl_->registrations.rend(); ++it) {
      const auto result = HcommMemUnreg(impl_->endpoint, it->handle);
      if (result != 0 && first_error.ok()) {
        first_error = HcommFailure("HcommMemUnreg", result);
      }
    }
    impl_->registrations.clear();

    const auto result = HcommEndpointDestroy(impl_->endpoint);
    if (result != 0 && first_error.ok()) {
      first_error = HcommFailure("HcommEndpointDestroy", result);
    }
    impl_->endpoint = nullptr;
  }
  impl_->closed = true;
  return first_error;
}

}  // namespace kvaf::access::a5

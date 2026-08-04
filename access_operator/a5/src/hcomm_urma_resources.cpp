#include "kvaf/access_operator/a5/hcomm_urma_resources.h"

#include <acl/acl.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

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
constexpr std::uint32_t kDefaultQos = 4;
constexpr std::uint32_t kMaximumChannelResources = 1024U;
// HCOMM v3 documents HCOMM_CHANNEL_NAME_MAX_LEN as 191.  Keep the numeric
// bound local because older public headers did not expose that macro, while a
// named channel is essential to prevent two concurrent probes on the same EID
// pair from anonymously pairing with each other.
constexpr std::size_t kMaximumChannelNameBytes = 191U;
constexpr std::int32_t kDirectRegedBufferRma = 1;
constexpr std::int32_t kDirectProtectionUb = 1;
constexpr std::int32_t kDirectSqUbJfs = 0;
constexpr std::int32_t kDirectCqUbJfc = 0;

// Minimal mirror of the device ChannelEntity ABI populated by HCOMM. These
// definitions intentionally stay in this Host-only translation unit: KVAF's
// stable ABI is AivDirectUrmaChannel, not the CANN-internal ChannelEntity.
struct DirectProtectionInfo {
  std::int32_t type;
  union {
    struct {
      std::uint32_t token_id;
      std::uint32_t token_value;
    } ub;
    std::uint8_t raw[24];
  } memory;
};

struct DirectRegedBufferEntity {
  std::int32_t type;
  union {
    struct {
      std::uint64_t address;
      std::uint64_t bytes;
    } ipc;
    struct {
      std::uint64_t address;
      std::uint64_t bytes;
      DirectProtectionInfo protection;
    } rma;
    std::uint8_t raw[56];
  } buffer;
};

struct DirectSqContext {
  std::int32_t type;
  union {
    struct {
      std::uint64_t sq_address;
      std::uint64_t head_address;
      std::uint64_t tail_address;
      std::uint64_t doorbell_address;
      std::uint32_t jfs_id;
      std::uint32_t wqe_size;
      std::uint32_t sq_depth;
      std::uint32_t tp_id;
      std::uint8_t remote_eid[16];
    } ub_jfs;
    std::uint8_t raw[120];
  } context;
};

struct DirectCqContext {
  std::int32_t type;
  union {
    struct {
      std::uint64_t cq_address;
      std::uint64_t head_address;
      std::uint64_t tail_address;
      std::uint64_t doorbell_address;
      std::uint32_t jfc_id;
      std::uint32_t cqe_size;
      std::uint32_t cq_depth;
    } ub_jfc;
    std::uint8_t raw[120];
  } context;
};

struct DirectChannelEntity {
  CommAbiHeader abi_header;
  std::int32_t engine;
  std::int32_t protocol;
  std::uint32_t local_notify_count;
  std::uint32_t remote_notify_count;
  std::uint32_t local_buffer_count;
  std::uint32_t remote_buffer_count;
  std::uint32_t sq_count;
  std::uint32_t cq_count;
  std::uint64_t local_notify_address;
  std::uint64_t remote_notify_address;
  std::uint64_t local_buffer_address;
  std::uint64_t remote_buffer_address;
  std::uint64_t sq_context_address;
  std::uint64_t cq_context_address;
  std::uint8_t reserved[160];
};

static_assert(sizeof(DirectProtectionInfo) == 28,
              "unexpected HCOMM protection ABI");
static_assert(sizeof(DirectRegedBufferEntity) == 64,
              "unexpected HCOMM registered-buffer ABI");
static_assert(sizeof(DirectSqContext) == 128,
              "unexpected HCOMM SQ ABI");
static_assert(sizeof(DirectCqContext) == 128,
              "unexpected HCOMM CQ ABI");
static_assert(sizeof(DirectChannelEntity) == 256,
              "unexpected HCOMM channel ABI");
bool IsPowerOfTwo(std::uint32_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

bool RangeContains(std::uint64_t base, std::uint64_t capacity,
                   std::uint64_t address, std::uint64_t bytes) {
  return address >= base && bytes <= capacity &&
         address - base <= capacity - bytes;
}

Status CopyFromDevice(void* destination, std::uint64_t source,
                      std::size_t bytes, const char* name) {
  if (destination == nullptr || source == 0 || bytes == 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         std::string("invalid ") + name +
                             " device snapshot input");
  }
  const auto result = aclrtMemcpy(
      destination, bytes, reinterpret_cast<const void*>(source), bytes,
      ACL_MEMCPY_DEVICE_TO_HOST);
  if (result != ACL_SUCCESS) {
    return Status::Error(StatusCode::kTransportError,
                         std::string("aclrtMemcpy(D2H ") + name +
                             ") failed, ACL ret=" +
                             std::to_string(result));
  }
  return Status::OK();
}

const DirectRegedBufferEntity* FindRmaBuffer(
    const std::vector<DirectRegedBufferEntity>& buffers,
    std::uint64_t address, std::uint64_t bytes) {
  for (const auto& buffer : buffers) {
    if (buffer.type == kDirectRegedBufferRma &&
        buffer.buffer.rma.protection.type == kDirectProtectionUb &&
        RangeContains(buffer.buffer.rma.address, buffer.buffer.rma.bytes,
                      address, bytes)) {
      return &buffer;
    }
  }
  return nullptr;
}

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
    // HCOMM channel descriptor v1 has no independent QoS field.  Its
    // hccsAttr and the UBC attributes used by newer runtimes share the same
    // union.  HcommChannelDescInit deliberately leaves that union as all
    // ones, which means "use the UBC defaults".  Writing HCCS QoS=4 for a
    // UBC-CTP descriptor aliases the first UBC word to sqDepth=4; the
    // runtime rejects that before it creates a channel because the legal
    // range starts at 16.  Only HCCS may use the v1 union member.
    if (desc->remoteEndpoint.protocol == COMM_PROTOCOL_HCCS) {
      desc->hccsAttr.qos = kDefaultQos;
    }
  }
}

template <typename T>
Status ValidateLegacyUbcQueueDefaults(const T& desc) {
  // This is intentionally limited to the v1 layout.  Newer headers expose a
  // separate top-level QoS field, so their UB attribute union must remain
  // entirely owned by the installed HCOMM ABI.
  if constexpr (!HasQos<T>::value && HasHccsQos<T>::value) {
    if (desc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP ||
        desc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
      std::uint32_t first_union_word = 0;
      std::memcpy(&first_union_word, &desc.hccsAttr.qos,
                  sizeof(first_union_word));
      if (first_union_word != std::numeric_limits<std::uint32_t>::max()) {
        return Status::Error(
            StatusCode::kInvalidArgument,
            "legacy HCOMM UBC descriptor has a non-default shared queue "
            "attribute; refusing HcommChannelCreate before an HCOMM "
            "resource operation");
      }
    }
  }
  return Status::OK();
}

template <typename T, typename = void>
struct HasChannelName : std::false_type {};

template <typename T>
struct HasChannelName<
    T, std::void_t<decltype(std::declval<T&>().channelName)>>
    : std::true_type {};

template <typename T>
Status SetChannelName(T* desc, const std::string& name) {
  if constexpr (HasChannelName<T>::value) {
    // HCOMM keeps this pointer while its asynchronous channel construction is
    // active.  The caller owns the backing string in HcommUrmaResources::Impl
    // until HcommChannelDestroy has succeeded.
    desc->channelName = name.empty() ? nullptr : name.c_str();
    return Status::OK();
  }
  if (!name.empty()) {
    return Status::Error(
        StatusCode::kNotImplemented,
        "installed public HCOMM headers do not expose ChannelDesc.channelName");
  }
  return Status::OK();
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

Status PrepareChannelDescriptor(const EndpointSpec& peer, ChannelRole role,
                                HcommChannelDesc* desc) {
  if (desc == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "HCOMM channel descriptor output is null");
  }
  const auto init_result = HcommChannelDescInit(desc, 1);
  if (init_result != 0) {
    return HcommFailure("HcommChannelDescInit", init_result);
  }
  auto status = BuildEndpointDesc(peer, &desc->remoteEndpoint);
  if (!status.ok()) {
    return status;
  }
  desc->notifyNum = 0;
  desc->exchangeAllMems = true;
  desc->memHandles = nullptr;
  desc->memHandleNum = 0;
  desc->role = role == ChannelRole::kClient ? HCOMM_SOCKET_ROLE_CLIENT
                                            : HCOMM_SOCKET_ROLE_SERVER;
  desc->port = 0;
  SetDefaultQos(desc);
  return ValidateLegacyUbcQueueDefaults(*desc);
}

}  // namespace

struct HcommUrmaResources::Impl {
  struct Registration {
    HcommMemHandle handle{nullptr};
  };

  EndpointHandle endpoint{nullptr};
  ChannelHandle channel{0};
  CommAbiHeader channel_abi_header{};
  std::string channel_name;
  std::vector<Registration> registrations;
  std::vector<std::vector<std::uint8_t>> imported_descriptors;
  // The public HCOMM reference specifies only that a nonzero return is a
  // failure; it does not state that every out-parameter is guaranteed empty
  // or that no internal object exists.  A null/zero output on a failed (or
  // malformed successful) create operation is therefore not proof that the
  // backing memory may be released.  Keep the owning process alive rather
  // than guessing until the product contract explicitly supplies that proof.
  const char* unproven_resource_operation{nullptr};
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
    // HCOMM may return a partially created endpoint together with an error.
    // Keep it owned so Close can retry destruction; dropping it would make a
    // subsequent MR/Host-DDR cleanup race an unknown endpoint lifetime.
    if (endpoint != nullptr) {
      impl_->endpoint = endpoint;
    } else {
      impl_->unproven_resource_operation = "HcommEndpointCreate";
    }
    return HcommFailure("HcommEndpointCreate", result);
  }
  if (endpoint == nullptr) {
    impl_->unproven_resource_operation = "HcommEndpointCreate";
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

  // Reserve before HCOMM can hand back a live handle.  No allocation may be
  // needed after the call while recording a partial/successful registration.
  try {
    impl_->registrations.reserve(impl_->registrations.size() + 1U);
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kOutOfMemory,
                         "reserving HCOMM registration tracking failed");
  }
  HcommMemHandle handle = nullptr;
  const auto result =
      HcommMemReg(impl_->endpoint, tag.c_str(), &memory, &handle);
  if (handle != nullptr) {
    impl_->registrations.push_back({handle});
  }
  if (result != 0) {
    // A non-null handle is retained above even on an error return.  The
    // retryable Close path, rather than a best-effort rollback, decides when
    // it is safe for the caller to release the registered backing memory.
    if (handle == nullptr) {
      impl_->unproven_resource_operation = "HcommMemReg";
    }
    return HcommFailure("HcommMemReg", result);
  }
  if (handle == nullptr) {
    impl_->unproven_resource_operation = "HcommMemReg";
    return HcommInvalidOutput("HcommMemReg");
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
    MemoryLocation expected_location,
    ImportedMemoryView* view) {
  if (impl_->endpoint == nullptr || descriptor.empty() || view == nullptr ||
      descriptor.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "invalid endpoint, descriptor, or imported view");
  }
  CommMem memory{};
  // Copy and reserve before the API call.  If HCOMM reports a partial or
  // successful import, moving this preallocated descriptor into ownership
  // cannot fail and makes HcommMemUnimport retryable through Close.
  std::vector<std::uint8_t> retained_descriptor;
  try {
    retained_descriptor = descriptor;
    impl_->imported_descriptors.reserve(
        impl_->imported_descriptors.size() + 1U);
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kOutOfMemory,
                         "reserving HCOMM import tracking failed");
  }
  const auto result =
      HcommMemImport(impl_->endpoint, descriptor.data(),
                     static_cast<std::uint32_t>(descriptor.size()), &memory);
  if (result != 0) {
    if (memory.addr != nullptr || memory.size != 0) {
      impl_->imported_descriptors.push_back(std::move(retained_descriptor));
    } else {
      impl_->unproven_resource_operation = "HcommMemImport";
    }
    return HcommFailure("HcommMemImport", result);
  }
  impl_->imported_descriptors.push_back(std::move(retained_descriptor));
  if (memory.addr == nullptr || memory.size == 0) {
    impl_->unproven_resource_operation = "HcommMemImport";
    return Status::Error(
        StatusCode::kTransportError,
        "HcommMemImport returned a null or empty memory view; the worker "
        "must retain process-owned backing until HCOMM documents the "
        "failure contract");
  }

  ImportedMemoryView imported_view{};
  imported_view.address = reinterpret_cast<Address>(memory.addr);
  imported_view.bytes = memory.size;
  // The UBC/URMA HCOMM implementation shipped with the A5 source tree fills
  // only addr/size in MemoryImport.  Its C adapter zero-initializes CommMem,
  // so reading `memory.type` would incorrectly classify every imported Host
  // DDR region as COMM_MEM_TYPE_DEVICE.  The descriptor producer is known by
  // this two-party exchange; preserve that explicit contract here.
  imported_view.location = expected_location;
  *view = imported_view;
  return Status::OK();
}

Status HcommUrmaResources::ValidateChannelDescriptor(
    const EndpointSpec& peer, ChannelRole role) {
  HcommChannelDesc desc{};
  return PrepareChannelDescriptor(peer, role, &desc);
}

Status HcommUrmaResources::CreateChannel(const EndpointSpec& peer,
                                         ChannelEngine engine,
                                         ChannelRole role,
                                         const std::string& channel_name) {
  if (impl_->endpoint == nullptr || impl_->channel != 0) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "endpoint is absent or channel already exists");
  }

  HcommChannelDesc desc{};
  auto status = PrepareChannelDescriptor(peer, role, &desc);
  if (!status.ok()) {
    return status;
  }

  if (channel_name.size() > kMaximumChannelNameBytes) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "HCOMM channel name exceeds the documented limit");
  }
  try {
    impl_->channel_name = channel_name;
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kOutOfMemory,
                         "tracking the HCOMM channel name failed");
  }
  const auto channel_name_status = SetChannelName(&desc, impl_->channel_name);
  if (!channel_name_status.ok()) {
    impl_->channel_name.clear();
    return channel_name_status;
  }

  const auto hcomm_engine =
      engine == ChannelEngine::kAiv ? COMM_ENGINE_AIV : COMM_ENGINE_CPU;
  ChannelHandle channel = 0;
  const auto result =
      HcommChannelCreate(impl_->endpoint, hcomm_engine, &desc, 1,
                         &channel);
  if (result != 0) {
    // A failed create can still hand back a partially initialized channel.
    // Retain both its handle and the owned channelName backing; silently
    // clearing either would make a later asynchronous destroy use a dangling
    // pointer.  The caller's normal retryable Close path owns the cleanup.
    if (channel != 0) {
      impl_->channel = channel;
      impl_->channel_abi_header = desc.header;
      return HcommFailure("HcommChannelCreate", result);
    }
    impl_->unproven_resource_operation = "HcommChannelCreate";
    return HcommFailure("HcommChannelCreate", result);
  }
  if (channel == 0) {
    impl_->unproven_resource_operation = "HcommChannelCreate";
    return HcommInvalidOutput("HcommChannelCreate");
  }
  impl_->channel = channel;
  impl_->channel_abi_header = desc.header;
  return Status::OK();
}

Status HcommUrmaResources::WaitChannelReady(std::uint32_t timeout_ms) {
  std::uint32_t elapsed_ms = 0;
  while (elapsed_ms <= timeout_ms) {
    bool ready = false;
    auto status = PollChannelReady(&ready);
    if (!status.ok()) {
      return status;
    }
    if (ready) {
      return Status::OK();
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kPollIntervalMs));
    elapsed_ms += kPollIntervalMs;
  }
  return Status::Error(StatusCode::kTimeout,
                       "timed out waiting for HCOMM channel readiness");
}

Status HcommUrmaResources::PollChannelReady(bool* ready) {
  if (impl_->channel == 0) {
    return Status::Error(StatusCode::kNotInitialized,
                         "HCOMM channel has not been created");
  }
  if (ready == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "HCOMM channel readiness output is null");
  }
  std::int32_t channel_status = kChannelConnecting;
  const auto result =
      HcommChannelGetStatus(&impl_->channel, 1, &channel_status);
  if (result != 0) {
    return HcommFailure("HcommChannelGetStatus", result);
  }
  if (channel_status != kChannelReady &&
      channel_status != kChannelConnecting) {
    return Status::Error(
        StatusCode::kTransportError,
        "HCOMM channel entered terminal status " +
            std::to_string(channel_status));
  }
  *ready = channel_status == kChannelReady;
  return Status::OK();
}

Status HcommUrmaResources::SnapshotDirectChannel(
    Address local_address, std::uint64_t local_bytes,
    Address remote_address, std::uint64_t remote_bytes,
    AivDirectUrmaChannel* direct_channel) const {
  if (impl_->channel == 0 || direct_channel == nullptr ||
      local_address == 0 || local_bytes == 0 || remote_address == 0 ||
      remote_bytes == 0) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "invalid HCOMM channel, local/remote range, or direct-channel output");
  }

  DirectChannelEntity channel{};
  auto status = CopyFromDevice(
      &channel, impl_->channel, sizeof(channel), "HCOMM ChannelEntity");
  if (!status.ok()) {
    return status;
  }
  if (channel.engine != COMM_ENGINE_AIV ||
      channel.protocol != COMM_PROTOCOL_UBC_CTP) {
    return Status::Error(
        StatusCode::kTransportError,
        "HCOMM channel is not an AIV UBC_CTP channel");
  }
  if (std::memcmp(&channel.abi_header, &impl_->channel_abi_header,
                  sizeof(channel.abi_header)) != 0 ||
      channel.sq_count == 0 || channel.cq_count == 0 ||
      channel.local_buffer_count == 0 ||
      channel.remote_buffer_count == 0 ||
      channel.sq_context_address == 0 ||
      channel.cq_context_address == 0 ||
      channel.local_buffer_address == 0 ||
      channel.remote_buffer_address == 0 ||
      channel.sq_count > kMaximumChannelResources ||
      channel.cq_count > kMaximumChannelResources ||
      channel.local_buffer_count > kMaximumChannelResources ||
      channel.remote_buffer_count > kMaximumChannelResources) {
    return Status::Error(
        StatusCode::kTransportError,
        "HCOMM ChannelEntity has an invalid ABI header or resource table");
  }

  DirectSqContext sq{};
  DirectCqContext cq{};
  status = CopyFromDevice(&sq, channel.sq_context_address, sizeof(sq),
                          "HCOMM SQ context");
  if (!status.ok()) {
    return status;
  }
  status = CopyFromDevice(&cq, channel.cq_context_address, sizeof(cq),
                          "HCOMM CQ context");
  if (!status.ok()) {
    return status;
  }
  if (sq.type != kDirectSqUbJfs || cq.type != kDirectCqUbJfc ||
      sq.context.ub_jfs.sq_address == 0 ||
      sq.context.ub_jfs.head_address == 0 ||
      sq.context.ub_jfs.tail_address == 0 ||
      sq.context.ub_jfs.doorbell_address == 0 ||
      cq.context.ub_jfc.cq_address == 0 ||
      cq.context.ub_jfc.tail_address == 0 ||
      cq.context.ub_jfc.doorbell_address == 0 ||
      sq.context.ub_jfs.wqe_size != 64 ||
      cq.context.ub_jfc.cqe_size != 64 ||
      !IsPowerOfTwo(sq.context.ub_jfs.sq_depth) ||
      !IsPowerOfTwo(cq.context.ub_jfc.cq_depth) ||
      sq.context.ub_jfs.tp_id >= (1U << 24U)) {
    return Status::Error(
        StatusCode::kTransportError,
        "HCOMM SQ/CQ context is incompatible with the 64-byte direct URMA "
        "READ path");
  }

  std::vector<DirectRegedBufferEntity> local_buffers;
  std::vector<DirectRegedBufferEntity> remote_buffers;
  try {
    local_buffers.resize(channel.local_buffer_count);
    remote_buffers.resize(channel.remote_buffer_count);
  } catch (const std::bad_alloc&) {
    return Status::Error(
        StatusCode::kOutOfMemory,
        "allocating the HCOMM direct-channel snapshot failed");
  }
  status = CopyFromDevice(
      local_buffers.data(), channel.local_buffer_address,
      local_buffers.size() * sizeof(local_buffers.front()),
      "HCOMM local MR table");
  if (!status.ok()) {
    return status;
  }
  status = CopyFromDevice(
      remote_buffers.data(), channel.remote_buffer_address,
      remote_buffers.size() * sizeof(remote_buffers.front()),
      "HCOMM remote MR table");
  if (!status.ok()) {
    return status;
  }

  const auto* local =
      FindRmaBuffer(local_buffers, local_address, local_bytes);
  const auto* remote =
      FindRmaBuffer(remote_buffers, remote_address, remote_bytes);
  if (local == nullptr || remote == nullptr) {
    return Status::Error(
        StatusCode::kTransportError,
        "requested HBM or imported Host DDR range is absent from the "
        "HCOMM channel MR tables");
  }
  const auto token_id =
      remote->buffer.rma.protection.memory.ub.token_id;
  if (token_id >= (1U << 20U)) {
    return Status::Error(StatusCode::kTransportError,
                         "HCOMM remote MR token id exceeds the SQE field");
  }

  std::uint64_t packed_sq_head = 0;
  std::uint32_t sq_tail = 0;
  std::uint32_t cq_tail = 0;
  status = CopyFromDevice(
      &packed_sq_head, sq.context.ub_jfs.head_address,
      sizeof(packed_sq_head), "HCOMM SQ head");
  if (!status.ok()) {
    return status;
  }
  status = CopyFromDevice(
      &sq_tail, sq.context.ub_jfs.tail_address, sizeof(sq_tail),
      "HCOMM SQ tail");
  if (!status.ok()) {
    return status;
  }
  status = CopyFromDevice(
      &cq_tail, cq.context.ub_jfc.tail_address, sizeof(cq_tail),
      "HCOMM CQ tail");
  if (!status.ok()) {
    return status;
  }
  const auto sq_head =
      static_cast<std::uint32_t>(packed_sq_head & 0xffffffffULL);
  const auto submitted_wqes =
      static_cast<std::uint32_t>(packed_sq_head >> 32U);
  if (sq_head != sq_tail || sq_tail != cq_tail ||
      submitted_wqes != cq_tail) {
    return Status::Error(
        StatusCode::kTransportError,
        "HCOMM channel has outstanding work and cannot transfer ownership "
        "to the direct AIV data plane");
  }

  bool any_remote_eid_byte = false;
  for (const auto byte : sq.context.ub_jfs.remote_eid) {
    any_remote_eid_byte = any_remote_eid_byte || byte != 0;
  }
  if (!any_remote_eid_byte) {
    return Status::Error(StatusCode::kTransportError,
                         "HCOMM SQ context contains an all-zero remote EID");
  }

  AivDirectUrmaChannel snapshot{};
  snapshot.flags = kAivUrmaChannelTokenEnabled;
  snapshot.sq_base = sq.context.ub_jfs.sq_address;
  snapshot.sq_doorbell = sq.context.ub_jfs.doorbell_address;
  snapshot.cq_base = cq.context.ub_jfc.cq_address;
  snapshot.cq_doorbell = cq.context.ub_jfc.doorbell_address;
  snapshot.remote_base = remote->buffer.rma.address;
  snapshot.remote_bytes = remote->buffer.rma.bytes;
  snapshot.local_base = local->buffer.rma.address;
  snapshot.local_bytes = local->buffer.rma.bytes;
  snapshot.sq_depth = sq.context.ub_jfs.sq_depth;
  snapshot.cq_depth = cq.context.ub_jfc.cq_depth;
  snapshot.wqe_size = sq.context.ub_jfs.wqe_size;
  snapshot.cqe_size = cq.context.ub_jfc.cqe_size;
  snapshot.tp_id = sq.context.ub_jfs.tp_id;
  snapshot.remote_token_id = token_id;
  snapshot.remote_token_value =
      remote->buffer.rma.protection.memory.ub.token_value;
  snapshot.sq_head = sq_head;
  snapshot.cq_tail = cq_tail;
  snapshot.sq_head_address = sq.context.ub_jfs.head_address;
  snapshot.sq_tail_address = sq.context.ub_jfs.tail_address;
  snapshot.cq_tail_address = cq.context.ub_jfc.tail_address;
  snapshot.wqe_count = submitted_wqes;
  std::memcpy(&snapshot.remote_eid_low,
              sq.context.ub_jfs.remote_eid,
              sizeof(snapshot.remote_eid_low));
  std::memcpy(&snapshot.remote_eid_high,
              sq.context.ub_jfs.remote_eid +
                  sizeof(snapshot.remote_eid_low),
              sizeof(snapshot.remote_eid_high));
  *direct_channel = snapshot;
  return Status::OK();
}

bool HcommUrmaResources::SupportsChannelName() const {
  return HasChannelName<HcommChannelDesc>::value;
}

std::uint64_t HcommUrmaResources::channel() const {
  return impl_->channel;
}

Status HcommUrmaResources::Close() {
  if (impl_->closed) {
    return Status::OK();
  }

  if (impl_->unproven_resource_operation != nullptr) {
    return Status::Error(
        StatusCode::kTransportError,
        std::string("HCOMM did not prove that a failed create/import left no ") +
            "resource: " + impl_->unproven_resource_operation +
            "; retaining the process-owned backing is required");
  }

  // Teardown is intentionally stateful and retryable.  A channel may still
  // refer to every imported/local MR, so a failed channel destroy is the one
  // dependency failure for which it is unsafe to attempt lower layers.  Once
  // the channel is gone, independent imports and registrations are drained
  // best-effort in reverse order; successful entries are forgotten while a
  // failed entry stays owned by this object for a future Close retry.
  if (impl_->channel != 0) {
    ChannelHandle destroying_channel = impl_->channel;
    const auto result = HcommChannelDestroy(&destroying_channel, 1);
    if (result != 0) {
      return HcommFailure("HcommChannelDestroy", result);
    }
    impl_->channel = 0;
    impl_->channel_abi_header = {};
    impl_->channel_name.clear();
  }

  Status first_failure = Status::OK();
  auto remember_failure = [&first_failure](const char* operation,
                                           std::int32_t result) {
    if (first_failure.ok()) {
      first_failure = HcommFailure(operation, result);
    }
  };
  if (impl_->endpoint != nullptr) {
    for (std::size_t index = impl_->imported_descriptors.size(); index != 0;) {
      --index;
      const auto& descriptor = impl_->imported_descriptors[index];
      const auto result =
          HcommMemUnimport(impl_->endpoint, descriptor.data(),
                           static_cast<std::uint32_t>(descriptor.size()));
      if (result != 0) {
        remember_failure("HcommMemUnimport", result);
        continue;
      }
      impl_->imported_descriptors.erase(
          impl_->imported_descriptors.begin() + index);
    }

    // Do not unregister local memory until every remote import is gone.  The
    // HCOMM API does not document whether a retained import can still use a
    // local MR owned by the same endpoint.
    if (!impl_->imported_descriptors.empty()) {
      return first_failure;
    }

    for (std::size_t index = impl_->registrations.size(); index != 0;) {
      --index;
      const auto result =
          HcommMemUnreg(impl_->endpoint, impl_->registrations[index].handle);
      if (result != 0) {
        remember_failure("HcommMemUnreg", result);
        continue;
      }
      impl_->registrations.erase(impl_->registrations.begin() + index);
    }

    // Endpoint destruction is last.  Retain it while any local registration
    // has an unproven unregister result, rather than handing the caller a
    // dangling HCOMM memory handle.
    if (!impl_->registrations.empty()) {
      return first_failure;
    }

    const auto result = HcommEndpointDestroy(impl_->endpoint);
    if (result != 0) {
      remember_failure("HcommEndpointDestroy", result);
      return first_failure;
    }
    impl_->endpoint = nullptr;
  }
  if (!first_failure.ok()) {
    return first_failure;
  }
  impl_->closed = true;
  return Status::OK();
}

}  // namespace kvaf::access::a5

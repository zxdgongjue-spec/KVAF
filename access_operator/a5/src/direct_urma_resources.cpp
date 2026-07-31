#include "kvaf/access_operator/a5/direct_urma_resources.h"

#include <acl/acl.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>

#include <unistd.h>

#include "kvaf_urma_provider_abi.h"

namespace kvaf::access::a5 {
namespace {

constexpr std::uint32_t kAicpuKfcKernelType = 5;
constexpr std::uint32_t kProviderRequiredFlags =
    KVAF_URMA_PROVIDER_CONNECTED |
    KVAF_URMA_PROVIDER_REMOTE_EID_NETWORK_ORDER |
    KVAF_URMA_PROVIDER_QUEUE_IS_AICPU_SHARED_VA;

struct RtAicpuArgsEx {
  void* args;
  void* host_input_info;
  void* kernel_offset_info;
  std::uint32_t args_size;
  std::uint16_t host_input_info_count;
  std::uint16_t kernel_offset_info_count;
  std::uint32_t so_name_offset;
  std::uint32_t kernel_name_offset;
  bool no_h2d_copy;
  std::uint16_t timeout;
  std::uint8_t reserved;
};

static_assert(sizeof(RtAicpuArgsEx) == 48,
              "unexpected CANN rtAicpuArgsEx ABI");

struct alignas(64) DirectQueueState {
  std::uint64_t packed_sq_head{0};
  std::uint32_t sq_tail{0};
  std::uint32_t cq_tail{0};
  std::uint64_t reserved[6]{};
};

static_assert(sizeof(DirectQueueState) == 64,
              "direct queue state must occupy one cache line");

bool IsPowerOfTwo(std::uint32_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

bool RangeContains(std::uint64_t base, std::uint64_t capacity,
                   std::uint64_t address, std::uint64_t bytes) {
  return address >= base && bytes <= capacity &&
         address - base <= capacity - bytes;
}

Status AclFailure(const char* operation, aclError result) {
  return Status::Error(StatusCode::kTransportError,
                       std::string(operation) + " failed, ACL ret=" +
                           std::to_string(result));
}

Status CopyFixedString(const std::string& source, char* destination,
                       std::size_t capacity, const char* name,
                       bool allow_empty = false) {
  if (destination == nullptr || capacity == 0 ||
      (!allow_empty && source.empty()) || source.size() >= capacity) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        std::string(name) + " is empty or exceeds the provider ABI");
  }
  std::memset(destination, 0, capacity);
  std::memcpy(destination, source.data(), source.size());
  return Status::OK();
}

Status ValidateProviderResult(
    const kvaf_urma_provider_result_v2& result,
    DirectUrmaExecutionDomain domain, std::uint64_t expected_session_id) {
  if (result.magic != KVAF_URMA_PROVIDER_MAGIC ||
      result.abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
      result.struct_size != sizeof(result) ||
      result.status != KVAF_URMA_PROVIDER_OK ||
      result.provider_status != 0 ||
      result.session_id != expected_session_id) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider returned an ABI or resource error: status=" +
            std::to_string(result.status) +
            " provider_status=" +
            std::to_string(result.provider_status));
  }
  if (result.local_base == 0 || result.local_bytes == 0 ||
      result.remote_base == 0 || result.remote_bytes == 0) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider returned an invalid memory view");
  }
  const std::uint32_t expected_private_abi =
      domain == DirectUrmaExecutionDomain::kHostProcess
          ? KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1
          : KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1;
  if (result.private_abi != expected_private_abi ||
      result.reserved[0] == 0) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider private UDMA ABI does not match its "
        "execution domain");
  }
  if (domain == DirectUrmaExecutionDomain::kHostProcess) {
    return Status::OK();
  }
  std::uint8_t eid_or = 0;
  for (const auto byte : result.remote_eid) {
    eid_or |= byte;
  }
  if ((result.flags & kProviderRequiredFlags) != kProviderRequiredFlags ||
      result.sq_base == 0 || result.sq_doorbell == 0 ||
      result.cq_base == 0 || result.cq_doorbell == 0 ||
      result.wqe_size != 64 || result.cqe_size != 64 ||
      !IsPowerOfTwo(result.sq_depth) ||
      !IsPowerOfTwo(result.cq_depth) ||
      result.sq_ring_shift >= 31U ||
      (1U << result.sq_ring_shift) != result.sq_depth ||
      result.tp_id >= (1U << 24U) ||
      result.remote_object_id >= (1U << 20U) || eid_or == 0 ||
      result.local_object_id >= (1U << 20U) ||
      result.sq_head != 0 || result.cq_tail != 0 ||
      result.wqe_count != 0 ||
      result.sq_head != result.cq_tail ||
      result.wqe_count != result.cq_tail) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider queue/token metadata is incompatible with "
        "the AIV 64-byte UDMA path");
  }
  return Status::OK();
}

std::uint64_t NextSessionId() {
  static std::atomic<std::uint64_t> sequence{1};
  const auto ticks = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto ordinal = sequence.fetch_add(1, std::memory_order_relaxed);
  std::uint64_t session =
      ticks ^ (static_cast<std::uint64_t>(::getpid()) << 32U) ^
      (ordinal * 0x9e3779b97f4a7c15ULL);
  if (session == 0) {
    session = ordinal == 0 ? 1 : ordinal;
  }
  return session;
}

}  // namespace

struct DirectUrmaResources::Impl {
  using ProviderOpen = int (*)(
      const kvaf_urma_provider_config_v2*,
      kvaf_urma_provider_result_v2*);
  using ProviderClose = int (*)(std::uint64_t);
  using AicpuLaunch = int (*)(
      std::uint32_t, const char*, std::uint32_t,
      const RtAicpuArgsEx*, void*, void*, std::uint32_t);

  DirectUrmaOpenSpec spec;
  kvaf_urma_provider_result_v2 result{};
  void* provider_library{nullptr};
  ProviderOpen provider_open{nullptr};
  ProviderClose provider_close{nullptr};
  void* runtime_library{nullptr};
  AicpuLaunch aicpu_launch{nullptr};
  aclrtStream stream{nullptr};
  DirectQueueState* queue_state{nullptr};
  bool queue_state_initialized{false};
  bool last_aicpu_launch_submitted{false};
  bool resource_quiescence_unknown{false};
  std::uint64_t session_id{0};
  bool provider_active{false};
  bool opened{false};
  bool closed{false};

  Status LoadRuntime() {
    static constexpr std::array<const char*, 2> kRuntimeLibraries{
        "libruntime.so", "libascendcl.so"};
    for (const auto* library : kRuntimeLibraries) {
      runtime_library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
      if (runtime_library == nullptr) {
        continue;
      }
      aicpu_launch = reinterpret_cast<AicpuLaunch>(
          dlsym(runtime_library, "rtAicpuKernelLaunchExWithArgs"));
      if (aicpu_launch != nullptr) {
        return Status::OK();
      }
      dlclose(runtime_library);
      runtime_library = nullptr;
    }
    const char* error = dlerror();
    return Status::Error(
        StatusCode::kUnavailable,
        std::string("rtAicpuKernelLaunchExWithArgs is unavailable: ") +
            (error == nullptr ? "unknown dlsym error" : error));
  }

  Status InvokeAicpu(void* args, std::uint32_t args_size,
                     std::uint32_t so_name_offset,
                     std::uint32_t kernel_name_offset,
                     const char* operation) {
    last_aicpu_launch_submitted = false;
    RtAicpuArgsEx launch_args{};
    launch_args.args = args;
    launch_args.args_size = args_size;
    launch_args.so_name_offset = so_name_offset;
    launch_args.kernel_name_offset = kernel_name_offset;
    const int result_code = aicpu_launch(
        kAicpuKfcKernelType, operation, 1, &launch_args, nullptr,
        stream, 0);
    if (result_code != 0) {
      return Status::Error(
          StatusCode::kTransportError,
          std::string(operation) + " launch failed, runtime ret=" +
              std::to_string(result_code));
    }
    last_aicpu_launch_submitted = true;
    const auto sync_result = aclrtSynchronizeStream(stream);
    if (sync_result != ACL_SUCCESS) {
      resource_quiescence_unknown = true;
      return AclFailure("aclrtSynchronizeStream(AICPU provider)",
                        sync_result);
    }
    return Status::OK();
  }
};

DirectUrmaResources::DirectUrmaResources()
    : impl_(std::make_unique<Impl>()) {}

DirectUrmaResources::~DirectUrmaResources() {
  (void)Close();
}

Status DirectUrmaResources::Open(const DirectUrmaOpenSpec& spec) {
  if (impl_->opened || impl_->closed || spec.local_address == 0 ||
      spec.local_allocation_bytes == 0 ||
      spec.segment_payload_bytes == 0 ||
      spec.control_timeout_seconds == 0 ||
      spec.control_timeout_seconds > 3600U) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "direct URMA resources are already used or the memory range is "
        "invalid");
  }
  if (spec.segment_payload_bytes >
          std::numeric_limits<std::uint32_t>::max() ||
      spec.segment_payload_bytes >
          std::numeric_limits<std::uint64_t>::max() / 2U) {
    return Status::Error(StatusCode::kOutOfRange,
                         "direct URMA segment payload is too large");
  }
  const std::uint64_t minimum_half =
      std::max<std::uint64_t>(spec.segment_payload_bytes, 4096U);
  if (spec.local_allocation_bytes < minimum_half * 2U) {
    return Status::Error(
        StatusCode::kOutOfRange,
        "direct URMA provider requires two page-aligned perftest halves");
  }

  kvaf_urma_provider_config_v2 config{};
  config.magic = KVAF_URMA_PROVIDER_MAGIC;
  config.abi_version = KVAF_URMA_PROVIDER_ABI_VERSION;
  config.struct_size = sizeof(config);
  config.role = spec.role == DirectUrmaRole::kServer
                    ? KVAF_URMA_PROVIDER_SERVER
                    : KVAF_URMA_PROVIDER_CLIENT;
  config.flags = KVAF_URMA_PROVIDER_USE_CTP |
                 (spec.execution_domain ==
                          DirectUrmaExecutionDomain::kHostProcess
                      ? KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI
                      : KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI);
  config.eid_index = spec.eid_index;
  config.control_port = spec.control_port;
  config.address_kind = KVAF_URMA_PROVIDER_ADDRESS_DEVICE_VA;
  config.control_timeout_seconds = spec.control_timeout_seconds;
  impl_->session_id = NextSessionId();
  config.session_id = impl_->session_id;
  config.local_address = spec.local_address;
  config.local_allocation_bytes = spec.local_allocation_bytes;
  config.segment_payload_bytes = spec.segment_payload_bytes;
  auto status = CopyFixedString(spec.urma_device, config.device_name,
                                sizeof(config.device_name),
                                "URMA device name");
  if (!status.ok()) {
    return status;
  }
  status = CopyFixedString(spec.local_eid, config.local_eid,
                           sizeof(config.local_eid), "local EID",
                           spec.role == DirectUrmaRole::kServer);
  if (!status.ok()) {
    return status;
  }
  status = CopyFixedString(spec.peer_eid, config.peer_eid,
                           sizeof(config.peer_eid), "peer EID",
                           spec.role == DirectUrmaRole::kServer);
  if (!status.ok()) {
    return status;
  }
  impl_->spec = spec;

  if (spec.execution_domain ==
      DirectUrmaExecutionDomain::kHostProcess) {
    if (spec.provider_library_path.empty()) {
      return Status::Error(
          StatusCode::kInvalidArgument,
          "Host direct URMA provider library path is empty");
    }
    impl_->provider_library = dlopen(
        spec.provider_library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (impl_->provider_library == nullptr) {
      const char* error = dlerror();
      return Status::Error(
          StatusCode::kUnavailable,
          std::string("dlopen(direct URMA provider) failed: ") +
              (error == nullptr ? "unknown dlopen error" : error));
    }
    impl_->provider_open = reinterpret_cast<Impl::ProviderOpen>(
        dlsym(impl_->provider_library, "kvaf_urma_provider_open_v2"));
    impl_->provider_close = reinterpret_cast<Impl::ProviderClose>(
        dlsym(impl_->provider_library, "kvaf_urma_provider_close_v2"));
    if (impl_->provider_open == nullptr ||
        impl_->provider_close == nullptr) {
      const char* error = dlerror();
      (void)Close();
      return Status::Error(
          StatusCode::kUnavailable,
          std::string("direct URMA provider ABI symbols are unavailable: ") +
              (error == nullptr ? "unknown dlsym error" : error));
    }
    const int provider_result =
        impl_->provider_open(&config, &impl_->result);
    if (provider_result != 0) {
      if ((impl_->result.flags &
           KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN) != 0) {
        impl_->resource_quiescence_unknown = true;
      }
      status = ValidateProviderResult(
          impl_->result, spec.execution_domain, impl_->session_id);
      (void)Close();
      return status.ok()
                 ? Status::Error(
                       StatusCode::kTransportError,
                       "direct URMA Host provider open failed")
                 : status;
    }
    impl_->provider_active = true;
  } else {
    if (spec.provider_soname.empty() ||
        spec.provider_soname.find('/') != std::string::npos) {
      return Status::Error(
          StatusCode::kInvalidArgument,
          "AICPU provider soname must be a basename");
    }
    status = impl_->LoadRuntime();
    if (!status.ok()) {
      return status;
    }
    const auto stream_result = aclrtCreateStream(&impl_->stream);
    if (stream_result != ACL_SUCCESS) {
      status = AclFailure("aclrtCreateStream(AICPU provider)",
                          stream_result);
      (void)Close();
      return status;
    }
    const auto queue_state_alloc = aclrtMalloc(
        reinterpret_cast<void**>(&impl_->queue_state),
        sizeof(DirectQueueState), ACL_MEM_MALLOC_HUGE_FIRST);
    if (queue_state_alloc != ACL_SUCCESS) {
      status = AclFailure(
          "aclrtMalloc(direct queue state)", queue_state_alloc);
      (void)Close();
      return status;
    }
    const DirectQueueState empty_queue_state{};
    auto queue_state_copy = aclrtMemcpy(
        impl_->queue_state, sizeof(empty_queue_state),
        &empty_queue_state, sizeof(empty_queue_state),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (queue_state_copy != ACL_SUCCESS) {
      status = AclFailure(
          "aclrtMemcpy(direct queue state initialize)",
          queue_state_copy);
      (void)Close();
      return status;
    }
    impl_->queue_state_initialized = true;

    kvaf_urma_provider_result_v2* device_result = nullptr;
    const auto alloc_result = aclrtMalloc(
        reinterpret_cast<void**>(&device_result),
        sizeof(*device_result), ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_result != ACL_SUCCESS) {
      status = AclFailure("aclrtMalloc(AICPU provider result)",
                          alloc_result);
      (void)Close();
      return status;
    }
    kvaf_urma_provider_result_v2 empty_result{};
    auto copy_result = aclrtMemcpy(
        device_result, sizeof(empty_result), &empty_result,
        sizeof(empty_result), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copy_result != ACL_SUCCESS) {
      (void)aclrtFree(device_result);
      status = AclFailure("aclrtMemcpy(AICPU result initialize)",
                          copy_result);
      (void)Close();
      return status;
    }

    kvaf_urma_aicpu_open_args_v2 open_args{};
    status = CopyFixedString(
        "aicpu_kvaf_urma_open_v2", open_args.kernel_name,
        sizeof(open_args.kernel_name), "AICPU provider kernel name");
    if (status.ok()) {
      status = CopyFixedString(
          spec.provider_soname, open_args.so_name,
          sizeof(open_args.so_name), "AICPU provider soname");
    }
    open_args.config = config;
    open_args.result_address =
        reinterpret_cast<std::uint64_t>(device_result);
    if (status.ok()) {
      // Close is idempotent even if the Open kernel is rejected before
      // creating a context. Mark the provider first so every launch/sync/copy
      // failure attempts an AICPU-side cleanup.
      impl_->provider_active = true;
      status = impl_->InvokeAicpu(
          &open_args, sizeof(open_args),
          offsetof(kvaf_urma_aicpu_open_args_v2, so_name),
          offsetof(kvaf_urma_aicpu_open_args_v2, kernel_name),
          "kvaf_direct_urma_open");
      if (!status.ok() &&
          !impl_->last_aicpu_launch_submitted) {
        impl_->provider_active = false;
      }
    }
    if (status.ok()) {
      copy_result = aclrtMemcpy(
          &impl_->result, sizeof(impl_->result), device_result,
          sizeof(impl_->result), ACL_MEMCPY_DEVICE_TO_HOST);
      if (copy_result != ACL_SUCCESS) {
        impl_->resource_quiescence_unknown = true;
        status = AclFailure("aclrtMemcpy(AICPU provider result)",
                            copy_result);
      } else if (impl_->result.magic == KVAF_URMA_PROVIDER_MAGIC &&
                 impl_->result.abi_version ==
                     KVAF_URMA_PROVIDER_ABI_VERSION &&
                 impl_->result.struct_size == sizeof(impl_->result) &&
                 impl_->result.status != KVAF_URMA_PROVIDER_OK) {
        // A complete error result proves that the AICPU entry point returned
        // without transferring ownership of a context to this nonce. In
        // particular, ALREADY_OPEN belongs to another caller and must never
        // trigger a close of that caller's singleton provider.
        if ((impl_->result.flags &
             KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN) != 0) {
          impl_->resource_quiescence_unknown = true;
        } else {
          impl_->provider_active = false;
        }
      }
    }
    if (!impl_->resource_quiescence_unknown) {
      const auto free_result = aclrtFree(device_result);
      if (status.ok() && free_result != ACL_SUCCESS) {
        impl_->resource_quiescence_unknown = true;
        status = AclFailure("aclrtFree(AICPU provider result)",
                            free_result);
      }
    }
    if (!status.ok()) {
      (void)Close();
      return status;
    }
  }

  status = ValidateProviderResult(
      impl_->result, spec.execution_domain, impl_->session_id);
  if (!status.ok()) {
    if (impl_->provider_active) {
      impl_->resource_quiescence_unknown = true;
    } else {
      (void)Close();
    }
    return status;
  }
  if (!RangeContains(impl_->result.local_base,
                     impl_->result.local_bytes, spec.local_address,
                     spec.segment_payload_bytes)) {
    if (impl_->provider_active) {
      impl_->resource_quiescence_unknown = true;
    } else {
      (void)Close();
    }
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider local MR does not contain the requested "
        "range");
  }

  impl_->opened = true;
  return Status::OK();
}

DirectUrmaMemoryView DirectUrmaResources::local_memory() const {
  return DirectUrmaMemoryView{impl_->result.local_base,
                              impl_->result.local_bytes};
}

DirectUrmaMemoryView DirectUrmaResources::remote_memory() const {
  return DirectUrmaMemoryView{impl_->result.remote_base,
                              impl_->result.remote_bytes};
}

Status DirectUrmaResources::SnapshotDirectChannel(
    Address local_address, std::uint64_t local_bytes,
    Address remote_address, std::uint64_t remote_bytes,
    AivDirectUrmaChannel* direct_channel) const {
  if (!impl_->opened || impl_->closed ||
      impl_->spec.execution_domain !=
          DirectUrmaExecutionDomain::kAicpu ||
      impl_->queue_state == nullptr || direct_channel == nullptr ||
      local_address == 0 || local_bytes == 0 || remote_address == 0 ||
      remote_bytes == 0 ||
      !RangeContains(impl_->result.local_base,
                     impl_->result.local_bytes,
                     local_address, local_bytes) ||
      !RangeContains(impl_->result.remote_base,
                     impl_->result.remote_bytes,
                     remote_address, remote_bytes)) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "invalid direct URMA channel snapshot range or lifecycle");
  }

  DirectQueueState state{};
  const auto copy_result = aclrtMemcpy(
      &state, sizeof(state), impl_->queue_state, sizeof(state),
      ACL_MEMCPY_DEVICE_TO_HOST);
  if (copy_result != ACL_SUCCESS) {
    return AclFailure("aclrtMemcpy(direct queue state snapshot)",
                      copy_result);
  }
  const auto sq_head =
      static_cast<std::uint32_t>(state.packed_sq_head);
  const auto wqe_count =
      static_cast<std::uint32_t>(state.packed_sq_head >> 32U);
  if (sq_head != state.sq_tail ||
      state.sq_tail != state.cq_tail ||
      wqe_count != state.cq_tail) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA queue has outstanding or incoherent work");
  }

  // Match udma_u_import_jetty()/udma_set_sqe() in the source-matched UDMA
  // provider: public urma_eid_t.raw is network order, while the hardware SQE
  // receives the fully reversed target_jetty.le_eid byte array.
  std::array<std::uint8_t, 16> wqe_eid{};
  std::reverse_copy(
      std::begin(impl_->result.remote_eid),
      std::end(impl_->result.remote_eid), wqe_eid.begin());

  AivDirectUrmaChannel snapshot{};
  snapshot.flags =
      (impl_->result.flags & KVAF_URMA_PROVIDER_TOKEN_ENABLED) != 0
          ? kAivUrmaChannelTokenEnabled
          : 0U;
  snapshot.sq_base = impl_->result.sq_base;
  snapshot.sq_doorbell = impl_->result.sq_doorbell;
  snapshot.cq_base = impl_->result.cq_base;
  snapshot.cq_doorbell = impl_->result.cq_doorbell;
  snapshot.remote_base = remote_address;
  snapshot.remote_bytes = remote_bytes;
  snapshot.local_base = local_address;
  snapshot.local_bytes = local_bytes;
  snapshot.sq_depth = impl_->result.sq_depth;
  snapshot.cq_depth = impl_->result.cq_depth;
  snapshot.wqe_size = impl_->result.wqe_size;
  snapshot.cqe_size = impl_->result.cqe_size;
  snapshot.tp_id = impl_->result.tp_id;
  snapshot.remote_token_id = impl_->result.remote_object_id;
  snapshot.remote_token_value = impl_->result.remote_token_value;
  snapshot.sq_head = sq_head;
  snapshot.cq_tail = state.cq_tail;
  const std::uint64_t queue_state_base =
      reinterpret_cast<std::uint64_t>(impl_->queue_state);
  snapshot.sq_head_address =
      queue_state_base + offsetof(DirectQueueState, packed_sq_head);
  snapshot.sq_tail_address =
      queue_state_base + offsetof(DirectQueueState, sq_tail);
  snapshot.cq_tail_address =
      queue_state_base + offsetof(DirectQueueState, cq_tail);
  snapshot.wqe_count = wqe_count;
  std::memcpy(&snapshot.remote_eid_low, wqe_eid.data(),
              sizeof(snapshot.remote_eid_low));
  std::memcpy(&snapshot.remote_eid_high,
              wqe_eid.data() + sizeof(snapshot.remote_eid_low),
              sizeof(snapshot.remote_eid_high));
  *direct_channel = snapshot;
  return Status::OK();
}

Status DirectUrmaResources::Close() {
  if (impl_->closed) {
    return Status::OK();
  }
  if (impl_->resource_quiescence_unknown) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct provider resource quiescence is unknown; retain all "
        "resources until device reset");
  }

  if (impl_->provider_active) {
    Status close_status = Status::OK();
    bool provider_was_closed = false;
    if (impl_->spec.execution_domain ==
        DirectUrmaExecutionDomain::kHostProcess) {
      if (impl_->provider_close == nullptr) {
        close_status = Status::Error(
            StatusCode::kTransportError,
            "direct URMA Host provider close symbol is unavailable");
      } else if (impl_->provider_close(impl_->session_id) != 0) {
        close_status = Status::Error(
            StatusCode::kTransportError,
            "direct URMA Host provider close failed");
      } else {
        provider_was_closed = true;
      }
    } else if (impl_->aicpu_launch != nullptr &&
               impl_->stream != nullptr) {
      kvaf_urma_aicpu_close_args_v2 close_args{};
      DirectQueueState state{};
      if (impl_->queue_state != nullptr &&
          impl_->queue_state_initialized) {
        const auto copy_result = aclrtMemcpy(
            &state, sizeof(state), impl_->queue_state, sizeof(state),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (copy_result != ACL_SUCCESS) {
          return AclFailure(
              "aclrtMemcpy(direct queue state before close)",
              copy_result);
        }
      } else {
        state.packed_sq_head =
            static_cast<std::uint64_t>(impl_->result.sq_head) |
            (static_cast<std::uint64_t>(impl_->result.wqe_count) << 32U);
        state.cq_tail = impl_->result.cq_tail;
      }
      close_args.final_sq_head =
          static_cast<std::uint32_t>(state.packed_sq_head);
      close_args.final_cq_tail = state.cq_tail;
      close_args.session_id = impl_->session_id;
      if (close_args.final_sq_head != state.cq_tail ||
          static_cast<std::uint32_t>(
              state.packed_sq_head >> 32U) != state.cq_tail) {
        return Status::Error(
            StatusCode::kTransportError,
            "direct URMA queue is not drained before provider close");
      }
      close_status = CopyFixedString(
          "aicpu_kvaf_urma_close_v2",
          close_args.kernel_name, sizeof(close_args.kernel_name),
          "AICPU provider close kernel");
      if (close_status.ok()) {
        close_status = CopyFixedString(
            impl_->spec.provider_soname, close_args.so_name,
            sizeof(close_args.so_name), "AICPU provider soname");
      }
      std::int32_t* device_close_result = nullptr;
      if (close_status.ok()) {
        const auto alloc_result = aclrtMalloc(
            reinterpret_cast<void**>(&device_close_result),
            sizeof(*device_close_result), ACL_MEM_MALLOC_HUGE_FIRST);
        if (alloc_result != ACL_SUCCESS) {
          close_status = AclFailure(
              "aclrtMalloc(AICPU close result)", alloc_result);
        }
      }
      constexpr std::int32_t kCloseNotExecuted =
          std::numeric_limits<std::int32_t>::min();
      if (close_status.ok()) {
        const auto copy_result = aclrtMemcpy(
            device_close_result, sizeof(*device_close_result),
            &kCloseNotExecuted, sizeof(kCloseNotExecuted),
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (copy_result != ACL_SUCCESS) {
          close_status = AclFailure(
              "aclrtMemcpy(AICPU close result initialize)", copy_result);
        }
      }
      close_args.result_address =
          reinterpret_cast<std::uint64_t>(device_close_result);
      if (close_status.ok()) {
        close_status = impl_->InvokeAicpu(
            &close_args, sizeof(close_args),
            offsetof(kvaf_urma_aicpu_close_args_v2, so_name),
            offsetof(kvaf_urma_aicpu_close_args_v2, kernel_name),
            "kvaf_direct_urma_close");
      }
      std::int32_t close_result = kCloseNotExecuted;
      if (close_status.ok()) {
        const auto copy_result = aclrtMemcpy(
            &close_result, sizeof(close_result), device_close_result,
            sizeof(*device_close_result), ACL_MEMCPY_DEVICE_TO_HOST);
        if (copy_result != ACL_SUCCESS) {
          close_status = AclFailure(
              "aclrtMemcpy(AICPU close result)", copy_result);
        } else if (close_result != 0) {
          close_status = Status::Error(
              StatusCode::kTransportError,
              "AICPU direct URMA provider close failed, provider ret=" +
                  std::to_string(close_result));
        } else {
          provider_was_closed = true;
        }
      }
      if (device_close_result != nullptr &&
          !impl_->resource_quiescence_unknown) {
        const auto free_result = aclrtFree(device_close_result);
        if (close_status.ok() && free_result != ACL_SUCCESS) {
          close_status = AclFailure(
              "aclrtFree(AICPU close result)", free_result);
        }
      }
    }
    if (provider_was_closed) {
      impl_->provider_active = false;
    }
    if (!close_status.ok()) {
      return close_status;
    }
  }
  impl_->opened = false;

  if (impl_->queue_state != nullptr) {
    const auto result = aclrtFree(impl_->queue_state);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtFree(direct queue state)", result);
    }
    impl_->queue_state = nullptr;
    impl_->queue_state_initialized = false;
  }
  if (impl_->stream != nullptr) {
    const auto result = aclrtDestroyStream(impl_->stream);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtDestroyStream(AICPU provider)", result);
    }
    impl_->stream = nullptr;
  }
  if (impl_->runtime_library != nullptr) {
    dlclose(impl_->runtime_library);
    impl_->runtime_library = nullptr;
    impl_->aicpu_launch = nullptr;
  }
  if (impl_->provider_library != nullptr) {
    dlclose(impl_->provider_library);
    impl_->provider_library = nullptr;
    impl_->provider_open = nullptr;
    impl_->provider_close = nullptr;
  }
  impl_->closed = true;
  impl_->session_id = 0;
  return Status::OK();
}

}  // namespace kvaf::access::a5

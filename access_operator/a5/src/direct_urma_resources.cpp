#include "kvaf/access_operator/a5/direct_urma_resources.h"

#include <acl/acl.h>
#include <runtime/rt_external_kernel.h>

#include <arpa/inet.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>

#include <unistd.h>

#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"
#include "kvaf_urma_provider_abi.h"

namespace kvaf::access::a5 {
namespace {

// CANN's public rt_external_kernel.h accepts the kernel type as uint32_t but
// does not expose the private rtKernelType_t enum.  The matching runtime ABI
// assigns AICPU_KFC the stable value 5; keep that ABI value local so this
// control plane builds against the public SDK rather than a private header.
constexpr std::uint32_t kAicpuKfcKernelType = 5U;
constexpr std::uint32_t kProviderRequiredFlags =
    KVAF_URMA_PROVIDER_CONNECTED |
    KVAF_URMA_PROVIDER_REMOTE_EID_NETWORK_ORDER |
    KVAF_URMA_PROVIDER_QUEUE_IS_AICPU_SHARED_VA;
constexpr std::uint32_t kEpochOwnerPollMicroseconds = 1000U;
constexpr std::uint32_t kEpochOwnerCloseRetryMicroseconds = 1000U;
constexpr std::uint32_t kEpochOwnerReadyAivPollBudget = 1U;
constexpr std::uint32_t kEpochOwnerReadyRetryMicroseconds = 10000U;
constexpr std::uint32_t kEpochOwnerShutdownAivPollBudget = 20000000U;

static_assert(sizeof(rtAicpuArgsEx_t) == 48,
              "unexpected CANN rtAicpuArgsEx ABI");
static_assert(offsetof(rtAicpuArgsEx_t, args) == 0 &&
                  offsetof(rtAicpuArgsEx_t, hostInputInfoPtr) == 8 &&
                  offsetof(rtAicpuArgsEx_t, kernelOffsetInfoPtr) == 16 &&
                  offsetof(rtAicpuArgsEx_t, argsSize) == 24 &&
                  offsetof(rtAicpuArgsEx_t, hostInputInfoNum) == 28 &&
                  offsetof(rtAicpuArgsEx_t, kernelOffsetInfoNum) == 30 &&
                  offsetof(rtAicpuArgsEx_t, soNameAddrOffset) == 32 &&
                  offsetof(rtAicpuArgsEx_t, kernelNameAddrOffset) == 36 &&
                  offsetof(rtAicpuArgsEx_t, isNoNeedH2DCopy) == 40 &&
                  offsetof(rtAicpuArgsEx_t, timeout) == 42 &&
                  offsetof(rtAicpuArgsEx_t, reserved) == 44,
              "unexpected CANN rtAicpuArgsEx field layout");

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

int NumericIpFamily(const std::string& address) {
  in_addr ipv4{};
  in6_addr ipv6{};
  if (!address.empty() &&
      inet_pton(AF_INET, address.c_str(), &ipv4) == 1) {
    return AF_INET;
  }
  if (!address.empty() &&
      inet_pton(AF_INET6, address.c_str(), &ipv6) == 1) {
    return AF_INET6;
  }
  return AF_UNSPEC;
}

int HexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return 10 + value - 'a';
  }
  if (value >= 'A' && value <= 'F') {
    return 10 + value - 'A';
  }
  return -1;
}

// URMA EIDs are 128-bit transport identifiers.  Their conventional text
// form resembles an IPv6 address, but validating one must not imply that a
// Linux IP socket, interface, route, or IPv6 stack participates in the path.
bool ParseUrmaEid(const std::string& text,
                  std::array<std::uint8_t, 16>* eid) {
  if (eid == nullptr) {
    return false;
  }
  std::array<char, 32> compact{};
  std::size_t length = 0;
  for (const char value : text) {
    if (value == ':' || value == '-') {
      continue;
    }
    if (length == compact.size() || HexNibble(value) < 0) {
      return false;
    }
    compact[length++] = value;
  }
  if (length != compact.size()) {
    return false;
  }
  bool any_nonzero = false;
  for (std::size_t index = 0; index < eid->size(); ++index) {
    const int high = HexNibble(compact[index * 2U]);
    const int low = HexNibble(compact[index * 2U + 1U]);
    (*eid)[index] =
        static_cast<std::uint8_t>((high << 4) | low);
    any_nonzero = any_nonzero || (*eid)[index] != 0;
  }
  return any_nonzero;
}

std::string FormatUrmaEid(
    const std::array<std::uint8_t, 16>& eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string canonical;
  canonical.reserve(39);
  for (std::size_t index = 0; index < eid.size(); ++index) {
    if (index != 0 && index % 2U == 0) {
      canonical.push_back(':');
    }
    canonical.push_back(kHex[eid[index] >> 4U]);
    canonical.push_back(kHex[eid[index] & 0x0fU]);
  }
  return canonical;
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
    DirectUrmaExecutionDomain domain, std::uint64_t expected_session_id,
    const std::string& expected_peer_eid) {
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
  std::array<std::uint8_t, 16> expected_peer{};
  if (!ParseUrmaEid(expected_peer_eid, &expected_peer) ||
      !std::equal(expected_peer.begin(), expected_peer.end(),
                  std::begin(result.remote_eid))) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider returned a remote EID different from the "
        "requested peer");
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
  if (result.reserved[1] != KVAF_UDMA_SOURCE_CAPABILITY_MAGIC ||
      result.reserved[2] !=
          KVAF_UDMA_TEARDOWN_SOURCE_OWNED_CHECKED_RETRYABLE_V1 ||
      result.reserved[3] == 0) {
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider did not prove the source-owned checked "
        "teardown/public-ABI runtime contract");
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

bool EpochOwnerStatusHeaderIsValid(
    const kvaf_urma_epoch_owner_status_v1& status,
    std::uint64_t expected_epoch, std::uint64_t expected_session) {
  return status.magic == KVAF_URMA_EPOCH_OWNER_MAGIC &&
         status.abi_version == KVAF_URMA_EPOCH_OWNER_ABI_VERSION &&
         status.struct_size == sizeof(status) &&
         status.epoch == expected_epoch &&
         status.session_id == expected_session;
}

bool EpochOwnerTeardownIsComplete(
    const kvaf_urma_epoch_owner_status_v1& status) {
  return status.state == KVAF_URMA_EPOCH_OWNER_STATE_CLOSED &&
         status.result_code == KVAF_URMA_EPOCH_OWNER_RESULT_OK &&
         status.provider_status == KVAF_URMA_PROVIDER_CLOSE_OK &&
         status.sequence == KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE &&
         status.close_attempts != 0 &&
         (status.flags &
          (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
           KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE)) ==
             (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
              KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE) &&
         (status.flags & (KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                          KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE)) == 0;
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

std::uint64_t NextEpochId() {
  static std::atomic<std::uint64_t> sequence{1};
  const auto ordinal = sequence.fetch_add(1, std::memory_order_relaxed);
  std::uint64_t epoch =
      (static_cast<std::uint64_t>(::getpid()) << 32U) ^
      (ordinal * 0xd6e8feb86659fd93ULL);
  if (epoch == 0) {
    epoch = ordinal == 0 ? 1 : ordinal;
  }
  return epoch;
}

}  // namespace

struct DirectUrmaResources::Impl {
  using ProviderOpen = int (*)(
      const kvaf_urma_provider_config_v2*,
      kvaf_urma_provider_result_v2*);
  using ProviderClose = int (*)(std::uint64_t);
  using AicpuLaunch = decltype(&rtAicpuKernelLaunchExWithArgs);

  DirectUrmaOpenSpec spec;
  kvaf_urma_provider_result_v2 result{};
  void* provider_library{nullptr};
  ProviderOpen provider_open{nullptr};
  ProviderClose provider_close{nullptr};
  void* runtime_library{nullptr};
  AicpuLaunch aicpu_launch{nullptr};
  aclrtStream owner_stream{nullptr};
  aclrtStream control_stream{nullptr};
  DirectQueueState* queue_state{nullptr};
  kvaf_urma_provider_result_v2* device_open_result{nullptr};
  kvaf_urma_epoch_owner_command_v1* owner_command{nullptr};
  kvaf_urma_epoch_owner_status_v1* owner_status{nullptr};
  AicpuEpochOwnerAivResult* owner_aiv_result{nullptr};
  mutable std::mutex data_plane_mutex;
  bool queue_state_initialized{false};
  bool owner_launch_submitted{false};
  bool owner_joined{false};
  bool shutdown_control_submitted{false};
  bool shutdown_control_quiesced{false};
  bool shutdown_result_validated{false};
  bool data_plane_frozen{false};
  bool data_plane_work_enqueued{false};
  bool data_plane_quiesced{true};
  bool resource_quiescence_unknown{false};
  std::uint64_t epoch_id{0};
  std::uint64_t session_id{0};
  bool provider_active{false};
  bool opened{false};
  bool closed{false};
  DirectUrmaClosePhase close_phase{DirectUrmaClosePhase::kNotStarted};
  DirectUrmaCloseDisposition close_disposition{
      DirectUrmaCloseDisposition::kNotStarted};

  bool HasRetainedState() const {
    return opened || closed || provider_library != nullptr ||
           provider_open != nullptr || provider_close != nullptr ||
           runtime_library != nullptr || aicpu_launch != nullptr ||
           owner_stream != nullptr || control_stream != nullptr ||
           queue_state != nullptr || device_open_result != nullptr ||
           owner_command != nullptr || owner_status != nullptr ||
           owner_aiv_result != nullptr || queue_state_initialized ||
           owner_launch_submitted || owner_joined ||
           shutdown_control_submitted || shutdown_control_quiesced ||
           shutdown_result_validated || data_plane_frozen ||
           data_plane_work_enqueued || !data_plane_quiesced ||
           resource_quiescence_unknown ||
           epoch_id != 0 || session_id != 0 || provider_active ||
           close_phase != DirectUrmaClosePhase::kNotStarted ||
           close_disposition != DirectUrmaCloseDisposition::kNotStarted;
  }

  void MarkPreLaunchRetryable() {
    close_phase = DirectUrmaClosePhase::kAicpuPreLaunch;
    close_disposition =
        DirectUrmaCloseDisposition::kPreLaunchRetryable;
  }

  void MarkResultReadPending() {
    close_phase = DirectUrmaClosePhase::kAicpuResultPending;
    close_disposition =
        DirectUrmaCloseDisposition::kAicpuResultReadPending;
  }

  void MarkProviderRetryable() {
    close_phase = DirectUrmaClosePhase::kProviderRetryable;
    close_disposition = DirectUrmaCloseDisposition::kProviderRetryable;
  }

  void MarkProviderClosed() {
    provider_active = false;
    close_phase = DirectUrmaClosePhase::kProviderClosed;
    close_disposition =
        DirectUrmaCloseDisposition::kTransportComplete;
  }

  void MarkUnsafeUnknown() {
    resource_quiescence_unknown = true;
    close_phase = DirectUrmaClosePhase::kUnsafeUnknown;
    close_disposition = DirectUrmaCloseDisposition::kUnsafeUnknown;
  }

  Status LoadRuntime() {
    // The executable is already linked to the build-manifest-pinned
    // libascendcl. Reopen that loaded object instead of searching an arbitrary
    // libruntime.so earlier in LD_LIBRARY_PATH (which may be a stub or another
    // CANN version). dlsym also searches this object's dependency closure.
    runtime_library = dlopen(
        "libascendcl.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (runtime_library != nullptr) {
      aicpu_launch = reinterpret_cast<AicpuLaunch>(
          dlsym(runtime_library, "rtAicpuKernelLaunchExWithArgs"));
      if (aicpu_launch != nullptr) {
        return Status::OK();
      }
      if (dlclose(runtime_library) != 0) {
        return Status::Error(
            StatusCode::kInternal,
            "failed to release pinned libascendcl after dlsym failure");
      }
      runtime_library = nullptr;
    }
    const char* error = dlerror();
    return Status::Error(
        StatusCode::kUnavailable,
        std::string("rtAicpuKernelLaunchExWithArgs is unavailable in the "
                    "already loaded libascendcl: ") +
            (error == nullptr ? "unknown dlsym error" : error));
  }

  Status LaunchEpochOwner(void* args, std::uint32_t args_size,
                          std::uint32_t so_name_offset,
                          std::uint32_t kernel_name_offset,
                          const char* operation) {
    rtAicpuArgsEx_t launch_args{};
    launch_args.args = args;
    launch_args.argsSize = args_size;
    launch_args.soNameAddrOffset = so_name_offset;
    launch_args.kernelNameAddrOffset = kernel_name_offset;
    // A runtime watchdog is safe for the resource-free probe but not for an
    // invocation that owns UDMA/UMMU handles. Zero selects the public
    // cooperative/no-forced-timeout path; shutdown is driven by the GM
    // command and the provider's own checked retry loop.
    launch_args.timeout = 0;
    // A nonzero launch return cannot prove that KFC never observed the
    // request. Arm retained ownership before crossing the runtime ABI.
    owner_launch_submitted = true;
    const auto result_code = aicpu_launch(
        kAicpuKfcKernelType, operation, 1, &launch_args, nullptr,
        reinterpret_cast<rtStream_t>(owner_stream), 0);
    if (result_code != 0) {
      return Status::Error(
          StatusCode::kTransportError,
          std::string(operation) + " launch failed, runtime ret=" +
              std::to_string(result_code));
    }
    return Status::OK();
  }

  Status SynchronizeWithRetry(aclrtStream target,
                              const char* operation) {
    if (target == nullptr) {
      return Status::Error(StatusCode::kInternal,
                           std::string(operation) +
                               " has no stream");
    }
    const auto first_sync_result = aclrtSynchronizeStream(target);
    aclError final_sync_result = first_sync_result;
    if (first_sync_result != ACL_SUCCESS) {
      // A transient scheduler/runtime wake-up error must not immediately turn
      // into an unrecoverable ownership state. Retry the same stream once;
      // no result buffer or provider backing is touched between attempts.
      final_sync_result = aclrtSynchronizeStream(target);
    }
    if (final_sync_result != ACL_SUCCESS) {
      resource_quiescence_unknown = true;
      return Status::Error(
          StatusCode::kTransportError,
          std::string(operation) + " failed twice, first=" +
              std::to_string(first_sync_result) + " second=" +
              std::to_string(final_sync_result));
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
  if (impl_->HasRetainedState() || spec.local_address == 0 ||
      spec.local_allocation_bytes == 0 ||
      spec.segment_payload_bytes == 0 ||
      spec.control_timeout_seconds == 0 ||
      spec.control_timeout_seconds > 3600U) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "direct URMA resources are already used, retain cleanup state, or "
        "the memory range is invalid");
  }
  if (spec.segment_payload_bytes >
          std::numeric_limits<std::uint32_t>::max() ||
      spec.segment_payload_bytes >
          std::numeric_limits<std::uint64_t>::max() / 2U) {
    return Status::Error(StatusCode::kOutOfRange,
                         "direct URMA segment payload is too large");
  }
  std::array<std::uint8_t, 16> local_eid{};
  std::array<std::uint8_t, 16> peer_eid{};
  if (!ParseUrmaEid(spec.local_eid, &local_eid) ||
      !ParseUrmaEid(spec.peer_eid, &peer_eid) ||
      local_eid == peer_eid) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "local/peer URMA EIDs must be distinct non-zero 128-bit values");
  }
  if (spec.management_mode == DirectUrmaManagementMode::kTcp) {
    const int control_family = NumericIpFamily(
        spec.control_server_address);
    const int bind_family = spec.control_bind_address.empty()
                                ? AF_UNSPEC
                                : NumericIpFamily(
                                      spec.control_bind_address);
    if (control_family == AF_UNSPEC ||
        (spec.role == DirectUrmaRole::kServer &&
         !spec.control_bind_address.empty()) ||
        (!spec.control_bind_address.empty() &&
         bind_family != control_family)) {
      return Status::Error(
          StatusCode::kInvalidArgument,
          "TCP control server/bind addresses must be numeric, use one IP "
          "family, and keep the server bind empty for wildcard listen");
    }
  } else if (spec.management_mode == DirectUrmaManagementMode::kUrma) {
    if (spec.management_jetty_id == 0 ||
        spec.management_jetty_id >= 1024U) {
      return Status::Error(
          StatusCode::kInvalidArgument,
          "URMA management Jetty ID must be in the range 1..1023");
    }
  } else {
    return Status::Error(StatusCode::kInvalidArgument,
                         "unknown direct URMA management mode");
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
  const std::uint32_t management_flag =
      spec.management_mode == DirectUrmaManagementMode::kUrma
          ? KVAF_URMA_PROVIDER_URMA_MANAGEMENT
          : KVAF_URMA_PROVIDER_EXPLICIT_TCP_MANAGEMENT;
  config.flags = KVAF_URMA_PROVIDER_USE_CTP | management_flag |
                 (spec.execution_domain ==
                          DirectUrmaExecutionDomain::kHostProcess
                      ? KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI
                      : KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI);
  config.eid_index = spec.eid_index;
  config.control_port =
      spec.management_mode == DirectUrmaManagementMode::kTcp
          ? spec.control_port
          : 0U;
  config.management_jetty_id =
      spec.management_mode == DirectUrmaManagementMode::kUrma
          ? spec.management_jetty_id
          : 0U;
  config.address_kind = KVAF_URMA_PROVIDER_ADDRESS_DEVICE_VA;
  config.control_timeout_seconds = spec.control_timeout_seconds;
  impl_->session_id = NextSessionId();
  // A DirectUrmaResources object owns exactly one transport epoch in both
  // execution domains.  Host resources are still opened/closed in-process;
  // Device resources additionally carry this nonce through the long-lived
  // AICPU owner command/status protocol.
  impl_->epoch_id = NextEpochId();
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
  if (spec.management_mode == DirectUrmaManagementMode::kTcp) {
    status = CopyFixedString(
        spec.control_server_address, config.control_server_address,
        sizeof(config.control_server_address),
        "TCP control server address");
    if (status.ok()) {
      status = CopyFixedString(
          spec.control_bind_address, config.control_bind_address,
          sizeof(config.control_bind_address), "TCP control bind address",
          true);
    }
  } else {
    const std::string canonical_peer_eid = FormatUrmaEid(peer_eid);
    const std::string canonical_local_eid = FormatUrmaEid(local_eid);
    status = CopyFixedString(
        canonical_peer_eid, config.management_peer_address,
        sizeof(config.management_peer_address),
        "URMA management peer EID");
    if (status.ok()) {
      status = CopyFixedString(
          canonical_local_eid, config.management_local_address,
          sizeof(config.management_local_address),
          "URMA management local EID");
    }
  }
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
      const bool complete_result =
          impl_->result.magic == KVAF_URMA_PROVIDER_MAGIC &&
          impl_->result.abi_version == KVAF_URMA_PROVIDER_ABI_VERSION &&
          impl_->result.struct_size == sizeof(impl_->result);
      const bool owns_session =
          complete_result &&
          impl_->result.session_id == impl_->session_id;
      const bool cleanup_retryable =
          owns_session &&
          (impl_->result.flags &
           KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE) != 0;
      const bool cleanup_uncertain =
          owns_session &&
          (impl_->result.flags &
           KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN) != 0;
      if (!complete_result ||
          ((impl_->result.flags &
            (KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE |
             KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN)) != 0 &&
           !owns_session) ||
          cleanup_uncertain) {
        impl_->resource_quiescence_unknown = true;
      }
      /*
       * A retryable failed Open still owns this exact provider session.  Keep
       * Close armed so the same nonce can resume the provider's teardown
       * cursor.  A contradictory "return failure/result OK" is also treated
       * as active and closed conservatively.
       */
      impl_->provider_active = cleanup_retryable || cleanup_uncertain ||
          (owns_session &&
           impl_->result.status == KVAF_URMA_PROVIDER_OK);
      status = ValidateProviderResult(
          impl_->result, spec.execution_domain, impl_->session_id,
          spec.peer_eid);
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
    const auto owner_stream_result =
        aclrtCreateStream(&impl_->owner_stream);
    if (owner_stream_result != ACL_SUCCESS) {
      status = AclFailure("aclrtCreateStream(AICPU epoch owner)",
                          owner_stream_result);
      (void)Close();
      return status;
    }
    const auto control_stream_result =
        aclrtCreateStream(&impl_->control_stream);
    if (control_stream_result != ACL_SUCCESS) {
      status = AclFailure("aclrtCreateStream(AIV epoch control)",
                          control_stream_result);
      (void)Close();
      return status;
    }
    if (impl_->owner_stream == impl_->control_stream) {
      status = Status::Error(
          StatusCode::kInternal,
          "AICPU owner and AIV control streams are not distinct");
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

    const auto alloc_result = aclrtMalloc(
        reinterpret_cast<void**>(&impl_->device_open_result),
        sizeof(*impl_->device_open_result), ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_result != ACL_SUCCESS) {
      status = AclFailure("aclrtMalloc(AICPU provider result)",
                          alloc_result);
      (void)Close();
      return status;
    }
    kvaf_urma_provider_result_v2 empty_result{};
    auto copy_result = aclrtMemcpy(
        impl_->device_open_result, sizeof(empty_result), &empty_result,
        sizeof(empty_result), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copy_result != ACL_SUCCESS) {
      status = AclFailure("aclrtMemcpy(AICPU result initialize)",
                          copy_result);
      (void)Close();
      return status;
    }

    const auto command_alloc = aclrtMalloc(
        reinterpret_cast<void**>(&impl_->owner_command),
        sizeof(*impl_->owner_command), ACL_MEM_MALLOC_HUGE_FIRST);
    if (command_alloc != ACL_SUCCESS) {
      status = AclFailure("aclrtMalloc(epoch owner command)",
                          command_alloc);
      (void)Close();
      return status;
    }
    const kvaf_urma_epoch_owner_command_v1 empty_command{};
    copy_result = aclrtMemcpy(
        impl_->owner_command, sizeof(empty_command), &empty_command,
        sizeof(empty_command), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copy_result != ACL_SUCCESS) {
      status = AclFailure("aclrtMemcpy(epoch owner command initialize)",
                          copy_result);
      (void)Close();
      return status;
    }

    const auto status_alloc = aclrtMalloc(
        reinterpret_cast<void**>(&impl_->owner_status),
        sizeof(*impl_->owner_status), ACL_MEM_MALLOC_HUGE_FIRST);
    if (status_alloc != ACL_SUCCESS) {
      status = AclFailure("aclrtMalloc(epoch owner status)", status_alloc);
      (void)Close();
      return status;
    }
    const kvaf_urma_epoch_owner_status_v1 empty_owner_status{};
    copy_result = aclrtMemcpy(
        impl_->owner_status, sizeof(empty_owner_status),
        &empty_owner_status, sizeof(empty_owner_status),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (copy_result != ACL_SUCCESS) {
      status = AclFailure("aclrtMemcpy(epoch owner status initialize)",
                          copy_result);
      (void)Close();
      return status;
    }

    const auto aiv_result_alloc = aclrtMalloc(
        reinterpret_cast<void**>(&impl_->owner_aiv_result),
        sizeof(*impl_->owner_aiv_result), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aiv_result_alloc != ACL_SUCCESS) {
      status = AclFailure("aclrtMalloc(epoch owner AIV result)",
                          aiv_result_alloc);
      (void)Close();
      return status;
    }
    const AicpuEpochOwnerAivResult empty_aiv_result{};
    copy_result = aclrtMemcpy(
        impl_->owner_aiv_result, sizeof(empty_aiv_result),
        &empty_aiv_result, sizeof(empty_aiv_result),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (copy_result != ACL_SUCCESS) {
      status = AclFailure("aclrtMemcpy(epoch owner AIV result initialize)",
                          copy_result);
      (void)Close();
      return status;
    }

    kvaf_urma_aicpu_epoch_owner_args_v1 open_args{};
    status = CopyFixedString(
        "aicpu_kvaf_urma_epoch_owner_v1", open_args.kernel_name,
        sizeof(open_args.kernel_name), "AICPU epoch-owner kernel name");
    if (status.ok()) {
      status = CopyFixedString(
          spec.provider_soname, open_args.so_name,
          sizeof(open_args.so_name), "AICPU provider soname");
    }
    open_args.config = config;
    open_args.epoch = impl_->epoch_id;
    open_args.command_address =
        reinterpret_cast<std::uint64_t>(impl_->owner_command);
    open_args.status_address =
        reinterpret_cast<std::uint64_t>(impl_->owner_status);
    open_args.open_result_address =
        reinterpret_cast<std::uint64_t>(impl_->device_open_result);
    open_args.lease_timeout_milliseconds = 0;
    open_args.poll_interval_microseconds = kEpochOwnerPollMicroseconds;
    open_args.close_retry_limit = 0;
    open_args.close_retry_interval_microseconds =
        kEpochOwnerCloseRetryMicroseconds;
    open_args.magic = KVAF_URMA_EPOCH_OWNER_MAGIC;
    open_args.abi_version = KVAF_URMA_EPOCH_OWNER_ABI_VERSION;
    open_args.struct_size = sizeof(open_args);
    open_args.flags = 0;
    if (status.ok()) {
      // Arm retained ownership before the asynchronous launch. The owner
      // stream must not be synchronized here: READY is observed by an AIV
      // task on the distinct control stream while this invocation stays live.
      impl_->provider_active = true;
      status = impl_->LaunchEpochOwner(
          &open_args, sizeof(open_args),
          offsetof(kvaf_urma_aicpu_epoch_owner_args_v1, so_name),
          offsetof(kvaf_urma_aicpu_epoch_owner_args_v1, kernel_name),
          "kvaf_direct_urma_epoch_owner");
      if (!status.ok()) {
        impl_->MarkUnsafeUnknown();
      }
    }
    AicpuEpochOwnerAivResult ready_result{};
    kvaf_urma_epoch_owner_status_v1 ready_status{};
    while (status.ok()) {
      // READY observation must never introduce a second timeout racing the
      // provider's own Open cleanup.  Take one coherent AIV snapshot per
      // launch, then yield on Host before observing the same owner again.
      // The outer runner owns the operational supervision deadline; if it
      // expires, the live worker and active-session record are retained.
      LaunchAivEpochOwnerWaitReady(
          impl_->control_stream, impl_->owner_status,
          impl_->device_open_result, impl_->owner_aiv_result,
          impl_->epoch_id, impl_->session_id,
          kEpochOwnerReadyAivPollBudget);
      status = impl_->SynchronizeWithRetry(
          impl_->control_stream,
          "aclrtSynchronizeStream(AIV epoch READY)");
      if (!status.ok()) {
        break;
      }
      copy_result = aclrtMemcpy(
          &ready_result, sizeof(ready_result), impl_->owner_aiv_result,
          sizeof(ready_result), ACL_MEMCPY_DEVICE_TO_HOST);
      if (copy_result == ACL_SUCCESS) {
        copy_result = aclrtMemcpy(
            &ready_status, sizeof(ready_status), impl_->owner_status,
            sizeof(ready_status), ACL_MEMCPY_DEVICE_TO_HOST);
      }
      if (copy_result == ACL_SUCCESS) {
        copy_result = aclrtMemcpy(
            &impl_->result, sizeof(impl_->result),
            impl_->device_open_result, sizeof(impl_->result),
            ACL_MEMCPY_DEVICE_TO_HOST);
      }
      if (copy_result != ACL_SUCCESS) {
        status = AclFailure("aclrtMemcpy(epoch owner READY evidence)",
                            copy_result);
        break;
      }
      const bool ready_still_pending =
          ready_result.status == static_cast<std::uint32_t>(
              AicpuEpochOwnerAivStatus::kTimedOut) &&
          (ready_result.observed_owner_state ==
               KVAF_URMA_EPOCH_OWNER_STATE_EMPTY ||
           ready_result.observed_owner_state ==
               KVAF_URMA_EPOCH_OWNER_STATE_OPENING);
      if (!ready_still_pending) {
        break;
      }

      // One AIV poll task is intentionally finite. A slow TCP/resource Open
      // gets another task on the same control stream until READY or a proved
      // terminal owner state; the AICPU owner itself is never relaunched.
      const AicpuEpochOwnerAivResult empty_retry_result{};
      copy_result = aclrtMemcpy(
          impl_->owner_aiv_result, sizeof(empty_retry_result),
          &empty_retry_result, sizeof(empty_retry_result),
          ACL_MEMCPY_HOST_TO_DEVICE);
      if (copy_result != ACL_SUCCESS) {
        status = AclFailure(
            "aclrtMemcpy(epoch READY retry initialize)", copy_result);
      } else {
        (void)::usleep(kEpochOwnerReadyRetryMicroseconds);
      }
    }
    const bool ready_observed =
        status.ok() &&
        ready_result.magic == kAicpuEpochOwnerAivResultMagic &&
        ready_result.abi_version ==
            kAicpuEpochOwnerAivResultAbiVersion &&
        ready_result.struct_size == sizeof(ready_result) &&
        ready_result.status == static_cast<std::uint32_t>(
            AicpuEpochOwnerAivStatus::kReadyObserved) &&
        ready_result.epoch == impl_->epoch_id &&
        ready_result.session_id == impl_->session_id &&
        ready_result.observed_owner_state ==
            KVAF_URMA_EPOCH_OWNER_STATE_READY &&
        EpochOwnerStatusHeaderIsValid(
            ready_status, impl_->epoch_id, impl_->session_id) &&
        ready_status.state == KVAF_URMA_EPOCH_OWNER_STATE_READY &&
        ready_status.result_code == KVAF_URMA_EPOCH_OWNER_RESULT_OK &&
        (ready_status.flags &
         (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE)) ==
            (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
             KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE) &&
        (ready_status.flags &
         (KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE |
          KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE)) == 0;
    if (!ready_observed) {
      const bool failed_before_resource_ownership =
          (ready_status.result_code ==
               KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED ||
           ready_status.result_code ==
               KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED) &&
          impl_->result.status != KVAF_URMA_PROVIDER_OK;
      const bool opened_then_closed_before_ready =
          (ready_status.result_code ==
               KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED ||
           ready_status.result_code ==
               KVAF_URMA_EPOCH_OWNER_RESULT_LEASE_EXPIRED) &&
          impl_->result.status == KVAF_URMA_PROVIDER_OK &&
          ready_status.close_attempts != 0 &&
          ready_status.provider_status ==
              KVAF_URMA_PROVIDER_CLOSE_OK;
      const bool open_failed_safe =
          status.ok() &&
          ready_result.status == static_cast<std::uint32_t>(
              AicpuEpochOwnerAivStatus::kOwnerTerminal) &&
          EpochOwnerStatusHeaderIsValid(
              ready_status, impl_->epoch_id, impl_->session_id) &&
          ready_status.state ==
              KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE &&
          (failed_before_resource_ownership ||
           opened_then_closed_before_ready) &&
          ready_status.sequence == 0 &&
          (ready_status.flags &
           (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
            KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE)) ==
              (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
               KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE) &&
          (ready_status.flags &
           (KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
            KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE)) == 0 &&
          impl_->result.magic == KVAF_URMA_PROVIDER_MAGIC &&
          impl_->result.abi_version ==
              KVAF_URMA_PROVIDER_ABI_VERSION &&
          impl_->result.struct_size == sizeof(impl_->result) &&
          impl_->result.session_id == impl_->session_id;
      if (open_failed_safe) {
        const auto join_status = impl_->SynchronizeWithRetry(
            impl_->owner_stream,
            "aclrtSynchronizeStream(failed epoch owner)");
        if (join_status.ok()) {
          impl_->owner_joined = true;
          impl_->MarkProviderClosed();
          status = Status::Error(
              StatusCode::kTransportError,
              "AICPU epoch owner failed Open and completed safe teardown: "
              "provider_result_status=" +
                  std::to_string(impl_->result.status) +
              " provider_status=" +
                  std::to_string(impl_->result.provider_status) +
              " owner_result_code=" +
                  std::to_string(ready_status.result_code));
        } else {
          status = join_status;
        }
      } else if (status.ok()) {
        impl_->MarkUnsafeUnknown();
        status = Status::Error(
            StatusCode::kTransportError,
            "AIV did not observe a valid resource-owning epoch READY: "
            "aiv_status=" + std::to_string(ready_result.status) +
            " aiv_observed_owner_state=" +
                std::to_string(ready_result.observed_owner_state) +
            " aiv_poll_count=" +
                std::to_string(ready_result.poll_count) +
            " owner_state=" + std::to_string(ready_status.state) +
            " owner_result_code=" +
                std::to_string(ready_status.result_code) +
            " owner_provider_status=" +
                std::to_string(ready_status.provider_status) +
            " owner_flags=" + std::to_string(ready_status.flags) +
            " provider_result_status=" +
                std::to_string(impl_->result.status) +
            " provider_status=" +
                std::to_string(impl_->result.provider_status));
      }
      (void)Close();
      return status;
    }
    impl_->close_phase = DirectUrmaClosePhase::kAicpuEpochOwnerReady;
  }

  status = ValidateProviderResult(
      impl_->result, spec.execution_domain, impl_->session_id,
      spec.peer_eid);
  if (!status.ok()) {
    // The Open task has completed and this object still owns the exact
    // session.  A malformed/incompatible result is not itself evidence of
    // in-flight DMA: use the checked Close path and quarantine only if that
    // close cannot prove teardown.
    (void)Close();
    return status;
  }
  if (!RangeContains(impl_->result.local_base,
                     impl_->result.local_bytes, spec.local_address,
                     spec.segment_payload_bytes)) {
    (void)Close();
    return Status::Error(
        StatusCode::kTransportError,
        "direct URMA provider local MR does not contain the requested "
        "range");
  }

  impl_->opened = true;
  impl_->data_plane_frozen = false;
  impl_->data_plane_work_enqueued = false;
  impl_->data_plane_quiesced = true;
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

std::uint64_t DirectUrmaResources::epoch() const {
  return impl_->epoch_id;
}

Status DirectUrmaResources::SnapshotDirectChannel(
    Address local_address, std::uint64_t local_bytes,
    Address remote_address, std::uint64_t remote_bytes,
    AivDirectUrmaChannel* direct_channel) const {
  std::lock_guard<std::mutex> data_plane_lock(
      impl_->data_plane_mutex);
  if (!impl_->opened || impl_->closed ||
      impl_->data_plane_frozen || !impl_->data_plane_quiesced ||
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

Status DirectUrmaResources::EnqueueBatch(
    const DeviceBatchGetParams* device_params,
    DeviceCompletion* device_completion,
    AivDirectUrmaChannel* direct_channel) {
  std::lock_guard<std::mutex> data_plane_lock(
      impl_->data_plane_mutex);
  if (!impl_->opened || impl_->closed || impl_->data_plane_frozen ||
      impl_->resource_quiescence_unknown ||
      impl_->spec.execution_domain !=
          DirectUrmaExecutionDomain::kAicpu ||
      impl_->control_stream == nullptr || device_params == nullptr ||
      device_completion == nullptr || direct_channel == nullptr) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "direct URMA batch cannot be submitted outside an active, "
        "unfrozen Device epoch");
  }

  // Arm the retained-work cursor before crossing the asynchronous AscendC
  // launch boundary.  There is no Host-visible return value from the launch;
  // any later exception or synchronization error must therefore retain this
  // resources object and all DMA-reachable backing.
  impl_->data_plane_work_enqueued = true;
  impl_->data_plane_quiesced = false;
  LaunchAivSimtUrmaH2D(
      impl_->control_stream, device_params, device_completion,
      direct_channel);
  return Status::OK();
}

Status DirectUrmaResources::SynchronizeDataPlane() {
  std::lock_guard<std::mutex> data_plane_lock(
      impl_->data_plane_mutex);
  if (impl_->closed) {
    return Status::OK();
  }
  if (!impl_->opened || impl_->resource_quiescence_unknown ||
      impl_->spec.execution_domain !=
          DirectUrmaExecutionDomain::kAicpu ||
      impl_->control_stream == nullptr) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        "direct URMA data plane is not an active Device epoch");
  }
  if (impl_->data_plane_quiesced) {
    return Status::OK();
  }
  const auto status = impl_->SynchronizeWithRetry(
      impl_->control_stream,
      "aclrtSynchronizeStream(AIV direct URMA batch)");
  if (!status.ok()) {
    // SynchronizeWithRetry already marks resource quiescence unknown.  Do not
    // permit Close to publish a software-only shutdown proof for a stream
    // whose execution state is indeterminate.
    return status;
  }
  impl_->data_plane_quiesced = true;
  return Status::OK();
}

bool DirectUrmaResources::transport_teardown_complete() const {
  return !impl_->provider_active &&
         !impl_->resource_quiescence_unknown;
}

bool DirectUrmaResources::transport_teardown_retryable() const {
  if (!impl_->provider_active || impl_->resource_quiescence_unknown) {
    return false;
  }
  return impl_->close_disposition ==
             DirectUrmaCloseDisposition::kPreLaunchRetryable ||
         impl_->close_disposition ==
             DirectUrmaCloseDisposition::kAicpuResultReadPending ||
         impl_->close_disposition ==
             DirectUrmaCloseDisposition::kAivShutdownPreLaunchRetryable ||
         impl_->close_disposition ==
             DirectUrmaCloseDisposition::kAivShutdownResultReadPending ||
         impl_->close_disposition ==
             DirectUrmaCloseDisposition::kAicpuEpochOwnerStatusReadPending ||
         impl_->close_disposition ==
             DirectUrmaCloseDisposition::kProviderRetryable;
}

DirectUrmaClosePhase DirectUrmaResources::close_phase() const {
  return impl_->close_phase;
}

DirectUrmaCloseDisposition DirectUrmaResources::close_disposition() const {
  return impl_->close_disposition;
}

Status DirectUrmaResources::Close() {
  std::lock_guard<std::mutex> data_plane_lock(
      impl_->data_plane_mutex);
  if (impl_->closed) {
    return Status::OK();
  }
  // Freeze first, under the same mutex used by EnqueueBatch.  Every accepted
  // data kernel was placed on control_stream before this point, so the final
  // shutdown kernel below is FIFO-ordered after all of them.  This stream
  // ordering, rather than a Host snapshot of equal software counters, is the
  // proof that Close cannot race a scheduled-but-not-started batch.
  impl_->data_plane_frozen = true;
  if (impl_->resource_quiescence_unknown) {
    impl_->MarkUnsafeUnknown();
    return Status::Error(
        StatusCode::kTransportError,
        "direct provider resource quiescence is unknown; retain all "
        "resources until device reset");
  }

  if (impl_->provider_active) {
    if (impl_->spec.execution_domain ==
        DirectUrmaExecutionDomain::kHostProcess) {
      impl_->close_phase = DirectUrmaClosePhase::kHostProviderCall;
      if (impl_->provider_close == nullptr) {
        impl_->MarkUnsafeUnknown();
        return Status::Error(
            StatusCode::kTransportError,
            "direct URMA Host provider close symbol is unavailable");
      }
      const int provider_close_result =
          impl_->provider_close(impl_->session_id);
      if (provider_close_result == KVAF_URMA_PROVIDER_CLOSE_OK) {
        impl_->MarkProviderClosed();
      } else if (provider_close_result ==
                 KVAF_URMA_PROVIDER_CLOSE_RETRYABLE) {
        impl_->MarkProviderRetryable();
        return Status::Error(
            StatusCode::kTransportError,
            "direct URMA Host provider close is retryable");
      } else {
        impl_->MarkUnsafeUnknown();
        return Status::Error(
            StatusCode::kTransportError,
            "direct URMA Host provider close is unsafe or unknown, ret=" +
                std::to_string(provider_close_result));
      }
    } else {
      if (impl_->owner_stream == nullptr ||
          impl_->control_stream == nullptr ||
          !impl_->owner_launch_submitted ||
          impl_->owner_command == nullptr ||
          impl_->owner_status == nullptr ||
          impl_->owner_aiv_result == nullptr ||
          impl_->queue_state == nullptr ||
          !impl_->queue_state_initialized || impl_->epoch_id == 0 ||
          impl_->session_id == 0) {
        impl_->MarkUnsafeUnknown();
        return Status::Error(
            StatusCode::kTransportError,
            "active AICPU epoch owner lost a stream or shared record");
      }

      if (!impl_->shutdown_control_submitted) {
        impl_->close_phase = DirectUrmaClosePhase::kAivShutdownPreLaunch;
        impl_->close_disposition =
            DirectUrmaCloseDisposition::kAivShutdownPreLaunchRetryable;

        // A previous READY observation remains in this cache line. Reset it
        // before launching the final control task so a Host-side reread can
        // never mistake stale READY evidence for a submitted shutdown.
        const AicpuEpochOwnerAivResult empty_result{};
        const auto initialize_result = aclrtMemcpy(
            impl_->owner_aiv_result, sizeof(empty_result), &empty_result,
            sizeof(empty_result), ACL_MEMCPY_HOST_TO_DEVICE);
        if (initialize_result != ACL_SUCCESS) {
          return AclFailure(
              "aclrtMemcpy(epoch owner shutdown result initialize)",
              initialize_result);
        }

        const std::uint64_t queue_state_base =
            reinterpret_cast<std::uint64_t>(impl_->queue_state);
        // There is no Host-visible return code for an AscendC launch. Arm the
        // monotonic cursor first: any exception after this boundary must only
        // synchronize/read this exact launch, never publish a second command.
        impl_->shutdown_control_submitted = true;
        impl_->close_phase =
            DirectUrmaClosePhase::kAivShutdownResultPending;
        impl_->close_disposition =
            DirectUrmaCloseDisposition::kAivShutdownResultReadPending;
        LaunchAivEpochOwnerShutdown(
            impl_->control_stream, impl_->owner_command,
            impl_->owner_status, impl_->owner_aiv_result,
            queue_state_base +
                offsetof(DirectQueueState, packed_sq_head),
            queue_state_base + offsetof(DirectQueueState, sq_tail),
            queue_state_base + offsetof(DirectQueueState, cq_tail),
            impl_->epoch_id, impl_->session_id,
            kEpochOwnerShutdownAivPollBudget);
      }

      if (!impl_->shutdown_control_quiesced) {
        const auto sync_status = impl_->SynchronizeWithRetry(
            impl_->control_stream,
            "aclrtSynchronizeStream(AIV epoch shutdown)");
        if (!sync_status.ok()) {
          impl_->MarkUnsafeUnknown();
          return sync_status;
        }
        impl_->shutdown_control_quiesced = true;
        impl_->data_plane_quiesced = true;
      }

      if (!impl_->shutdown_result_validated) {
        AicpuEpochOwnerAivResult shutdown_result{};
        kvaf_urma_epoch_owner_command_v1 shutdown_command{};
        auto copy_result = aclrtMemcpy(
            &shutdown_result, sizeof(shutdown_result),
            impl_->owner_aiv_result, sizeof(shutdown_result),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (copy_result == ACL_SUCCESS) {
          copy_result = aclrtMemcpy(
              &shutdown_command, sizeof(shutdown_command),
              impl_->owner_command, sizeof(shutdown_command),
              ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (copy_result != ACL_SUCCESS) {
          impl_->close_phase =
              DirectUrmaClosePhase::kAivShutdownResultPending;
          impl_->close_disposition =
              DirectUrmaCloseDisposition::kAivShutdownResultReadPending;
          return AclFailure(
              "aclrtMemcpy(epoch owner shutdown evidence)", copy_result);
        }

        const bool observed_live_owner =
            shutdown_result.observed_owner_state ==
                KVAF_URMA_EPOCH_OWNER_STATE_READY ||
            shutdown_result.observed_owner_state ==
                KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING;
        const bool counters_quiesced =
            shutdown_result.observed_sq_head ==
                shutdown_result.observed_sq_tail &&
            shutdown_result.observed_sq_tail ==
                shutdown_result.observed_cq_tail &&
            shutdown_result.observed_cq_tail ==
                shutdown_result.observed_wqe_count;
        const bool result_valid =
            shutdown_result.magic == kAicpuEpochOwnerAivResultMagic &&
            shutdown_result.abi_version ==
                kAicpuEpochOwnerAivResultAbiVersion &&
            shutdown_result.struct_size == sizeof(shutdown_result) &&
            shutdown_result.status == static_cast<std::uint32_t>(
                AicpuEpochOwnerAivStatus::kShutdownSubmitted) &&
            shutdown_result.epoch == impl_->epoch_id &&
            shutdown_result.session_id == impl_->session_id &&
            observed_live_owner && counters_quiesced;
        const bool command_valid =
            shutdown_command.magic == KVAF_URMA_EPOCH_OWNER_MAGIC &&
            shutdown_command.abi_version ==
                KVAF_URMA_EPOCH_OWNER_ABI_VERSION &&
            shutdown_command.struct_size == sizeof(shutdown_command) &&
            shutdown_command.opcode ==
                KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED &&
            shutdown_command.epoch == impl_->epoch_id &&
            shutdown_command.session_id == impl_->session_id &&
            shutdown_command.sequence ==
                KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE &&
            shutdown_command.reserved[0] == 0 &&
            shutdown_command.final_sq_head ==
                shutdown_result.observed_sq_head &&
            shutdown_command.final_sq_tail ==
                shutdown_result.observed_sq_tail &&
            shutdown_command.final_cq_tail ==
                shutdown_result.observed_cq_tail &&
            shutdown_command.final_wqe_count ==
                shutdown_result.observed_wqe_count;
        if (!result_valid || !command_valid) {
          impl_->MarkUnsafeUnknown();
          const bool queue_not_quiesced =
              shutdown_result.status == static_cast<std::uint32_t>(
                  AicpuEpochOwnerAivStatus::kQueueNotQuiesced);
          return Status::Error(
              StatusCode::kTransportError,
              queue_not_quiesced
                  ? "AIV refused epoch shutdown because the URMA queue is "
                    "not quiesced"
                  : "AIV epoch shutdown evidence is invalid or incomplete");
        }
        impl_->shutdown_result_validated = true;
      }

      if (!impl_->owner_joined) {
        impl_->close_phase =
            DirectUrmaClosePhase::kAicpuEpochOwnerJoinPending;
        impl_->close_disposition =
            DirectUrmaCloseDisposition::kAicpuEpochOwnerJoinPending;
        const auto join_status = impl_->SynchronizeWithRetry(
            impl_->owner_stream,
            "aclrtSynchronizeStream(AICPU epoch owner join)");
        if (!join_status.ok()) {
          impl_->MarkUnsafeUnknown();
          return join_status;
        }
        impl_->owner_joined = true;
      }

      impl_->close_phase =
          DirectUrmaClosePhase::kAicpuEpochOwnerStatusPending;
      impl_->close_disposition =
          DirectUrmaCloseDisposition::kAicpuEpochOwnerStatusReadPending;
      kvaf_urma_epoch_owner_status_v1 final_owner_status{};
      const auto status_copy_result = aclrtMemcpy(
          &final_owner_status, sizeof(final_owner_status),
          impl_->owner_status, sizeof(final_owner_status),
          ACL_MEMCPY_DEVICE_TO_HOST);
      if (status_copy_result != ACL_SUCCESS) {
        // The owner stream is already joined. A retry is now a pure read of
        // the retained terminal cache line; it cannot relaunch or re-close.
        return AclFailure(
            "aclrtMemcpy(epoch owner terminal status)",
            status_copy_result);
      }
      if (!EpochOwnerStatusHeaderIsValid(
              final_owner_status, impl_->epoch_id,
              impl_->session_id) ||
          !EpochOwnerTeardownIsComplete(final_owner_status)) {
        impl_->MarkUnsafeUnknown();
        return Status::Error(
            StatusCode::kTransportError,
            "AICPU epoch owner joined without proving checked teardown");
      }
      impl_->MarkProviderClosed();
    }
  }
  impl_->opened = false;
  impl_->close_phase = DirectUrmaClosePhase::kLocalCleanup;
  impl_->close_disposition =
      DirectUrmaCloseDisposition::kTransportComplete;

  // These tiny shared records remain object-owned until provider teardown is
  // proven. Clear a pointer only after aclrtFree succeeds so a transient
  // local cleanup error can be retried without losing the allocation handle.
  if (impl_->owner_aiv_result != nullptr) {
    const auto result = aclrtFree(impl_->owner_aiv_result);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtFree(epoch owner AIV result)", result);
    }
    impl_->owner_aiv_result = nullptr;
  }
  if (impl_->owner_status != nullptr) {
    const auto result = aclrtFree(impl_->owner_status);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtFree(epoch owner status)", result);
    }
    impl_->owner_status = nullptr;
  }
  if (impl_->owner_command != nullptr) {
    const auto result = aclrtFree(impl_->owner_command);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtFree(epoch owner command)", result);
    }
    impl_->owner_command = nullptr;
  }
  if (impl_->device_open_result != nullptr) {
    const auto result = aclrtFree(impl_->device_open_result);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtFree(AICPU provider result)", result);
    }
    impl_->device_open_result = nullptr;
  }

  if (impl_->queue_state != nullptr) {
    const auto result = aclrtFree(impl_->queue_state);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtFree(direct queue state)", result);
    }
    impl_->queue_state = nullptr;
    impl_->queue_state_initialized = false;
  }
  if (impl_->control_stream != nullptr) {
    const auto result = aclrtDestroyStream(impl_->control_stream);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtDestroyStream(AIV epoch control)", result);
    }
    impl_->control_stream = nullptr;
  }
  if (impl_->owner_stream != nullptr) {
    // A submitted owner stream is destroyable only after its invocation has
    // joined.  Reaching local cleanup with an unjoined submitted owner would
    // violate the provider lifetime proof and must retain every handle.
    if (impl_->owner_launch_submitted && !impl_->owner_joined) {
      impl_->MarkUnsafeUnknown();
      return Status::Error(
          StatusCode::kTransportError,
          "cannot destroy an unjoined AICPU epoch-owner stream");
    }
    const auto result = aclrtDestroyStream(impl_->owner_stream);
    if (result != ACL_SUCCESS) {
      return AclFailure("aclrtDestroyStream(AICPU epoch owner)", result);
    }
    impl_->owner_stream = nullptr;
  }
  if (impl_->runtime_library != nullptr) {
    if (dlclose(impl_->runtime_library) != 0) {
      const char* error = dlerror();
      return Status::Error(
          StatusCode::kInternal,
          std::string("dlclose(AICPU runtime library) failed: ") +
              (error == nullptr ? "unknown dlclose error" : error));
    }
    impl_->runtime_library = nullptr;
    impl_->aicpu_launch = nullptr;
  }
  if (impl_->provider_library != nullptr) {
    if (dlclose(impl_->provider_library) != 0) {
      const char* error = dlerror();
      return Status::Error(
          StatusCode::kInternal,
          std::string("dlclose(direct URMA provider) failed: ") +
              (error == nullptr ? "unknown dlclose error" : error));
    }
    impl_->provider_library = nullptr;
    impl_->provider_open = nullptr;
    impl_->provider_close = nullptr;
  }
  impl_->closed = true;
  impl_->data_plane_work_enqueued = false;
  impl_->data_plane_quiesced = true;
  impl_->epoch_id = 0;
  impl_->session_id = 0;
  impl_->close_phase = DirectUrmaClosePhase::kComplete;
  impl_->close_disposition =
      DirectUrmaCloseDisposition::kTransportComplete;
  return Status::OK();
}

}  // namespace kvaf::access::a5

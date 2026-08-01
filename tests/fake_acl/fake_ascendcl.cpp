#include <acl/acl.h>
#include <runtime/rt_external_kernel.h>

#include <arpa/inet.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "fake_acl_control.h"
#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"
#include "kvaf_urma_provider_abi.h"

namespace {

std::string g_scenario;
int g_sync_calls = 0;
int g_close_launches = 0;
int g_free_calls = 0;
int g_destroy_stream_calls = 0;
int g_d2h_copies = 0;
std::uint32_t g_last_kernel_type = 0;
std::uint64_t g_open_session = 0;
std::uint64_t g_last_close_session = 0;
std::uint64_t g_epoch = 0;
aclrtStream g_owner_stream = nullptr;
aclrtStream g_control_stream = nullptr;
aclrtStream g_data_stream = nullptr;
kvaf_urma_epoch_owner_command_v1* g_owner_command = nullptr;
kvaf_urma_epoch_owner_status_v1* g_owner_status = nullptr;
kvaf_urma_provider_result_v2* g_device_open_result = nullptr;
kvaf::access::a5::AicpuEpochOwnerAivResult* g_owner_aiv_result = nullptr;
bool g_shutdown_submitted = false;
bool g_owner_joined = false;
bool g_shutdown_read_failed = false;
bool g_terminal_status_read_failed = false;
int g_aiv_result_h2d_copies = 0;
int g_ready_wait_launches = 0;
int g_data_launches = 0;

bool Scenario(const char* name) {
  return g_scenario == name;
}

void FillSuccessfulOpen(const kvaf_urma_provider_config_v2& config,
                        kvaf_urma_provider_result_v2* result) {
  std::memset(result, 0, sizeof(*result));
  result->magic = KVAF_URMA_PROVIDER_MAGIC;
  result->abi_version = KVAF_URMA_PROVIDER_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KVAF_URMA_PROVIDER_OK;
  result->provider_status = 0;
  result->flags = KVAF_URMA_PROVIDER_CONNECTED |
                  KVAF_URMA_PROVIDER_REMOTE_EID_NETWORK_ORDER |
                  KVAF_URMA_PROVIDER_QUEUE_IS_AICPU_SHARED_VA;
  result->sq_depth = 8;
  result->cq_depth = 8;
  result->sq_base = 0x100000;
  result->sq_doorbell = 0x100100;
  result->cq_base = 0x101000;
  result->cq_doorbell = 0x101100;
  result->local_base = config.local_address;
  result->local_bytes = config.local_allocation_bytes;
  result->remote_base = 0x200000;
  result->remote_bytes = config.local_allocation_bytes;
  result->wqe_size = 64;
  result->cqe_size = 64;
  result->tp_id = 1;
  result->remote_object_id = 2;
  result->local_object_id = 3;
  result->sq_ring_shift = 3;
  result->private_abi = KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1;
  result->session_id = config.session_id;
  result->reserved[0] = 1;
  result->reserved[1] = KVAF_UDMA_SOURCE_CAPABILITY_MAGIC;
  result->reserved[2] =
      KVAF_UDMA_TEARDOWN_SOURCE_OWNED_CHECKED_RETRYABLE_V1;
  result->reserved[3] = 1;
  (void)inet_pton(AF_INET6,
                  "0000:0000:007f:0400:0010:0000:df0a:0b00",
                  result->remote_eid);
}

}  // namespace

extern "C" void kvaf_fake_acl_set_scenario(const char* scenario) {
  g_scenario = scenario == nullptr ? "" : scenario;
  g_sync_calls = 0;
  g_close_launches = 0;
  g_free_calls = 0;
  g_destroy_stream_calls = 0;
  g_d2h_copies = 0;
  g_last_kernel_type = 0;
  g_open_session = 0;
  g_last_close_session = 0;
  g_epoch = 0;
  g_owner_stream = nullptr;
  g_control_stream = nullptr;
  g_data_stream = nullptr;
  g_owner_command = nullptr;
  g_owner_status = nullptr;
  g_device_open_result = nullptr;
  g_owner_aiv_result = nullptr;
  g_shutdown_submitted = false;
  g_owner_joined = false;
  g_shutdown_read_failed = false;
  g_terminal_status_read_failed = false;
  g_aiv_result_h2d_copies = 0;
  g_ready_wait_launches = 0;
  g_data_launches = 0;
}

extern "C" int kvaf_fake_acl_sync_calls(void) {
  return g_sync_calls;
}

extern "C" int kvaf_fake_acl_close_launches(void) {
  return g_close_launches;
}

extern "C" int kvaf_fake_acl_free_calls(void) {
  return g_free_calls;
}

extern "C" int kvaf_fake_acl_destroy_stream_calls(void) {
  return g_destroy_stream_calls;
}

extern "C" unsigned long long kvaf_fake_acl_open_session(void) {
  return g_open_session;
}

extern "C" unsigned long long kvaf_fake_acl_last_close_session(void) {
  return g_last_close_session;
}

extern "C" uint32_t kvaf_fake_acl_last_kernel_type(void) {
  return g_last_kernel_type;
}

extern "C" int kvaf_fake_acl_data_launches(void) {
  return g_data_launches;
}

extern "C" int kvaf_fake_acl_data_stream_matches_control(void) {
  return g_data_stream != nullptr && g_data_stream == g_control_stream;
}

extern "C" int kvaf_urma_provider_open_v2(
    const kvaf_urma_provider_config_v2* config,
    kvaf_urma_provider_result_v2* result) {
  if (config == nullptr || result == nullptr) {
    return -1;
  }
  g_open_session = config->session_id;
  FillSuccessfulOpen(*config, result);
  result->private_abi = KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1;
  result->flags = KVAF_URMA_PROVIDER_CONNECTED |
                  KVAF_URMA_PROVIDER_REMOTE_EID_NETWORK_ORDER;
  if (Scenario("host_open_cleanup_retryable")) {
    result->status = KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED;
    result->provider_status = -3;
    result->flags = KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE;
    return -1;
  }
  return 0;
}

extern "C" int kvaf_urma_provider_close_v2(std::uint64_t session_id) {
  ++g_close_launches;
  g_last_close_session = session_id;
  if (Scenario("host_open_cleanup_retryable") && g_close_launches == 1) {
    return KVAF_URMA_PROVIDER_CLOSE_RETRYABLE;
  }
  if (Scenario("host_close_unsafe")) {
    return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
  }
  return KVAF_URMA_PROVIDER_CLOSE_OK;
}

extern "C" aclError aclInit(const char*) { return ACL_SUCCESS; }
extern "C" aclError aclFinalize(void) { return ACL_SUCCESS; }
extern "C" aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
extern "C" aclError aclrtResetDevice(int32_t) { return ACL_SUCCESS; }

extern "C" aclError aclrtCreateStream(aclrtStream* stream) {
  *stream = std::malloc(1);
  return *stream == nullptr ? 1 : ACL_SUCCESS;
}

extern "C" aclError aclrtDestroyStream(aclrtStream stream) {
  ++g_destroy_stream_calls;
  if (Scenario("destroy_stream_retry") &&
      g_destroy_stream_calls == 1) {
    return 71;
  }
  std::free(stream);
  return ACL_SUCCESS;
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream stream) {
  ++g_sync_calls;
  if (Scenario("sync_retry") && g_sync_calls == 1) {
    return 81;
  }
  if (Scenario("sync_twice") && g_sync_calls <= 2) {
    return 82;
  }
  if (Scenario("owner_join_sync_twice") &&
      stream == g_owner_stream && g_shutdown_submitted) {
    return 83;
  }
  if (stream == g_owner_stream && g_shutdown_submitted) {
    g_owner_joined = true;
  }
  return ACL_SUCCESS;
}

extern "C" aclError aclrtMalloc(void** address, size_t bytes, int) {
  *address = std::malloc(bytes);
  return *address == nullptr ? 1 : ACL_SUCCESS;
}

extern "C" aclError aclrtFree(void* address) {
  ++g_free_calls;
  if (Scenario("free_retry") && g_free_calls == 1) {
    return 91;
  }
  std::free(address);
  return ACL_SUCCESS;
}

extern "C" aclError aclrtMemcpy(void* destination,
                                 size_t destination_bytes,
                                 const void* source,
                                 size_t source_bytes, int kind) {
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
    ++g_d2h_copies;
    if (Scenario("open_d2h_failure") && !g_shutdown_submitted &&
        source == g_owner_aiv_result) {
      return 101;
    }
    if (Scenario("close_d2h_retry") && g_shutdown_submitted &&
        source == g_owner_aiv_result && !g_shutdown_read_failed) {
      g_shutdown_read_failed = true;
      return 102;
    }
    if (Scenario("close_status_d2h_retry") && g_owner_joined &&
        source == g_owner_status && !g_terminal_status_read_failed) {
      g_terminal_status_read_failed = true;
      return 104;
    }
  } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE &&
             destination == g_owner_aiv_result) {
    ++g_aiv_result_h2d_copies;
    if (Scenario("close_prelaunch_h2d_retry") &&
        g_aiv_result_h2d_copies == 1) {
      return 103;
    }
  }
  if (source_bytes > destination_bytes) {
    return 1;
  }
  std::memcpy(destination, source, source_bytes);
  return ACL_SUCCESS;
}

extern "C" rtError_t rtAicpuKernelLaunchExWithArgs(
    uint32_t kernel_type, const char_t* op_name, uint32_t,
    const rtAicpuArgsEx_t* args_info, rtSmDesc_t*, rtStream_t stream,
    uint32_t) {
  g_last_kernel_type = kernel_type;
  if (std::strcmp(op_name, "kvaf_direct_urma_epoch_owner") == 0) {
    auto* args = static_cast<kvaf_urma_aicpu_epoch_owner_args_v1*>(
        args_info->args);
    auto* result = reinterpret_cast<kvaf_urma_provider_result_v2*>(
        args->open_result_address);
    auto* owner_status =
        reinterpret_cast<kvaf_urma_epoch_owner_status_v1*>(
            args->status_address);
    g_owner_stream = reinterpret_cast<aclrtStream>(stream);
    g_owner_command =
        reinterpret_cast<kvaf_urma_epoch_owner_command_v1*>(
            args->command_address);
    g_owner_status = owner_status;
    g_device_open_result = result;
    g_open_session = args->config.session_id;
    g_epoch = args->epoch;
    if (Scenario("open_launch_failure")) {
      return 130;
    }
    FillSuccessfulOpen(args->config, result);
    std::memset(owner_status, 0, sizeof(*owner_status));
    owner_status->magic = KVAF_URMA_EPOCH_OWNER_MAGIC;
    owner_status->abi_version = KVAF_URMA_EPOCH_OWNER_ABI_VERSION;
    owner_status->struct_size = sizeof(*owner_status);
    owner_status->epoch = args->epoch;
    owner_status->session_id = args->config.session_id;
    owner_status->result_code = KVAF_URMA_EPOCH_OWNER_RESULT_OK;
    owner_status->flags = KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                          KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE;
    owner_status->state = KVAF_URMA_EPOCH_OWNER_STATE_READY;
    if (Scenario("foreign_already_open")) {
      result->status = KVAF_URMA_PROVIDER_ALREADY_OPEN;
      result->provider_status = -1;
      result->flags = 0;
      owner_status->result_code = KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED;
      owner_status->flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
      owner_status->state =
          KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE;
    } else if (Scenario("clock_failed_before_open")) {
      result->status = KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED;
      result->provider_status = -1;
      result->flags = 0;
      owner_status->result_code =
          KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED;
      owner_status->flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
      owner_status->state =
          KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE;
    } else if (Scenario("lease_expired_after_open")) {
      owner_status->result_code =
          KVAF_URMA_EPOCH_OWNER_RESULT_LEASE_EXPIRED;
      owner_status->provider_status = KVAF_URMA_PROVIDER_CLOSE_OK;
      owner_status->close_attempts = 1;
      owner_status->flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
      owner_status->state =
          KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE;
      ++g_close_launches;
      g_last_close_session = args->config.session_id;
    } else if (Scenario("open_failed_safe_bad_result")) {
      // Contradictory evidence: the status claims safe failed Open while the
      // provider result still reports success. The wrapper must retain it as
      // ownership-unknown rather than freeing the shared backing.
      owner_status->result_code = KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED;
      owner_status->flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
      owner_status->state =
          KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE;
    } else if (Scenario("open_cleanup_retryable")) {
      result->status = KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED;
      result->provider_status = -3;
      result->flags = KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE;
      owner_status->result_code = KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED;
      owner_status->flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
      owner_status->state =
          KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE;
      ++g_close_launches;
      g_last_close_session = args->config.session_id;
    } else if (Scenario("runtime_capability_partial")) {
      result->reserved[2] =
          KVAF_UDMA_TEARDOWN_SOURCE_OWNED_PARTIAL_RETRY_V1;
    } else if (Scenario("runtime_public_abi_missing")) {
      result->reserved[3] = 0;
    }
    return 0;
  }
  return 1;
}

namespace kvaf::access::a5 {

void LaunchAivSimtUrmaH2D(
    void* stream, const DeviceBatchGetParams*, DeviceCompletion*,
    AivDirectUrmaChannel*) {
  ++g_data_launches;
  g_data_stream = reinterpret_cast<aclrtStream>(stream);
}

void LaunchAivEpochOwnerWaitReady(
    void* stream, const void* owner_status, const void*,
    AicpuEpochOwnerAivResult* aiv_result, std::uint64_t epoch,
    std::uint64_t session_id, std::uint32_t) {
  g_control_stream = reinterpret_cast<aclrtStream>(stream);
  g_owner_aiv_result = aiv_result;
  ++g_ready_wait_launches;
  const auto* status =
      static_cast<const kvaf_urma_epoch_owner_status_v1*>(owner_status);
  AicpuEpochOwnerAivResult result{};
  result.epoch = epoch;
  result.session_id = session_id;
  result.observed_owner_state = status->state;
  result.observed_sequence = status->sequence;
  if (Scenario("ready_timeout_once") && g_ready_wait_launches == 1) {
    result.status = static_cast<std::uint32_t>(
        AicpuEpochOwnerAivStatus::kTimedOut);
    *aiv_result = result;
    return;
  }
  result.status = static_cast<std::uint32_t>(
      status->state == KVAF_URMA_EPOCH_OWNER_STATE_READY
          ? AicpuEpochOwnerAivStatus::kReadyObserved
          : AicpuEpochOwnerAivStatus::kOwnerTerminal);
  *aiv_result = result;
}

void LaunchAivEpochOwnerShutdown(
    void* stream, void* command, const void* owner_status,
    AicpuEpochOwnerAivResult* aiv_result,
    std::uint64_t packed_sq_head_address,
    std::uint64_t sq_tail_address, std::uint64_t cq_tail_address,
    std::uint64_t epoch, std::uint64_t session_id, std::uint32_t) {
  if (g_control_stream == nullptr) {
    g_control_stream = reinterpret_cast<aclrtStream>(stream);
  }
  ++g_close_launches;
  g_last_close_session = session_id;
  g_shutdown_submitted = true;
  const auto* status =
      static_cast<const kvaf_urma_epoch_owner_status_v1*>(owner_status);
  const auto packed_sq_head =
      *reinterpret_cast<const std::uint64_t*>(packed_sq_head_address);
  const auto sq_head = static_cast<std::uint32_t>(packed_sq_head);
  const auto wqe_count =
      static_cast<std::uint32_t>(packed_sq_head >> 32U);
  const auto sq_tail =
      *reinterpret_cast<const std::uint32_t*>(sq_tail_address);
  const auto cq_tail =
      *reinterpret_cast<const std::uint32_t*>(cq_tail_address);

  AicpuEpochOwnerAivResult result{};
  result.epoch = epoch;
  result.session_id = session_id;
  result.observed_owner_state = status->state;
  result.observed_sequence = status->sequence;
  result.observed_sq_head = sq_head;
  result.observed_sq_tail = sq_tail;
  result.observed_cq_tail = cq_tail;
  result.observed_wqe_count = wqe_count;
  if (Scenario("close_queue_not_drained")) {
    result.observed_sq_head = 1;
  }
  const bool quiesced = result.observed_sq_head ==
                            result.observed_sq_tail &&
                        result.observed_sq_tail ==
                            result.observed_cq_tail &&
                        result.observed_cq_tail ==
                            result.observed_wqe_count;
  if (!quiesced) {
    result.status = static_cast<std::uint32_t>(
        AicpuEpochOwnerAivStatus::kQueueNotQuiesced);
    *aiv_result = result;
    return;
  }

  auto* shutdown =
      static_cast<kvaf_urma_epoch_owner_command_v1*>(command);
  std::memset(shutdown, 0, sizeof(*shutdown));
  shutdown->magic = KVAF_URMA_EPOCH_OWNER_MAGIC;
  shutdown->abi_version = KVAF_URMA_EPOCH_OWNER_ABI_VERSION;
  shutdown->struct_size = sizeof(*shutdown);
  shutdown->epoch = epoch;
  shutdown->session_id = session_id;
  shutdown->sequence = KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE;
  shutdown->final_sq_head = sq_head;
  shutdown->final_sq_tail = sq_tail;
  shutdown->final_cq_tail = cq_tail;
  shutdown->final_wqe_count = wqe_count;
  if (Scenario("close_command_reserved_nonzero")) {
    shutdown->reserved[0] = 1;
  }
  shutdown->opcode =
      KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED;
  result.status = static_cast<std::uint32_t>(
      AicpuEpochOwnerAivStatus::kShutdownSubmitted);
  *aiv_result = result;

  auto* mutable_status =
      const_cast<kvaf_urma_epoch_owner_status_v1*>(status);
  mutable_status->sequence = KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE;
  if (Scenario("close_unsafe")) {
    mutable_status->state =
        KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING;
    mutable_status->result_code =
        KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_UNSAFE;
    mutable_status->flags = KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                            KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE;
  } else {
    mutable_status->state = KVAF_URMA_EPOCH_OWNER_STATE_CLOSED;
    mutable_status->result_code = KVAF_URMA_EPOCH_OWNER_RESULT_OK;
    mutable_status->provider_status = KVAF_URMA_PROVIDER_CLOSE_OK;
    mutable_status->close_attempts = 1;
    mutable_status->flags =
        KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
        KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
    if (Scenario("close_terminal_incomplete")) {
      mutable_status->close_attempts = 0;
      mutable_status->flags =
          KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE;
    }
  }
}

}  // namespace kvaf::access::a5

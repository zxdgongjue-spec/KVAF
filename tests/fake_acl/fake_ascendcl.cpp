#include <acl/acl.h>
#include <runtime/rt_external_kernel.h>

#include <arpa/inet.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "fake_acl_control.h"
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
  (void)inet_pton(AF_INET6, config.peer_eid, result->remote_eid);
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

extern "C" aclError aclrtSynchronizeStream(aclrtStream) {
  ++g_sync_calls;
  if (Scenario("sync_retry") && g_sync_calls == 1) {
    return 81;
  }
  if (Scenario("sync_twice") && g_sync_calls <= 2) {
    return 82;
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
    if (Scenario("open_d2h_failure") && g_d2h_copies == 1) {
      return 101;
    }
    if (Scenario("close_d2h_retry") && g_d2h_copies == 3) {
      return 102;
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
    const rtAicpuArgsEx_t* args_info, rtSmDesc_t*, rtStream_t, uint32_t) {
  g_last_kernel_type = kernel_type;
  if (std::strcmp(op_name, "kvaf_direct_urma_open") == 0) {
    auto* args = static_cast<kvaf_urma_aicpu_open_args_v2*>(
        args_info->args);
    auto* result = reinterpret_cast<kvaf_urma_provider_result_v2*>(
        args->result_address);
    g_open_session = args->config.session_id;
    FillSuccessfulOpen(args->config, result);
    if (Scenario("foreign_already_open")) {
      result->status = KVAF_URMA_PROVIDER_ALREADY_OPEN;
      result->provider_status = -1;
      result->flags = 0;
    }
    return 0;
  }
  if (std::strcmp(op_name, "kvaf_direct_urma_close") == 0) {
    auto* args = static_cast<kvaf_urma_aicpu_close_args_v2*>(
        args_info->args);
    ++g_close_launches;
    g_last_close_session = args->session_id;
    *reinterpret_cast<std::int32_t*>(args->result_address) = 0;
    return 0;
  }
  return 1;
}

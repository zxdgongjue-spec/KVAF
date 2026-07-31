#ifndef KVAF_URMA_PROVIDER_ABI_H
#define KVAF_URMA_PROVIDER_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define KVAF_URMA_PROVIDER_EXPORT \
  __attribute__((visibility("default")))
#else
#define KVAF_URMA_PROVIDER_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KVAF_URMA_PROVIDER_MAGIC 0x4b555250U
#define KVAF_URMA_PROVIDER_ABI_VERSION 2U
#define KVAF_URMA_PROVIDER_NAME_BYTES 32U
#define KVAF_URMA_PROVIDER_EID_TEXT_BYTES 64U

enum kvaf_urma_provider_role {
  KVAF_URMA_PROVIDER_SERVER = 0,
  KVAF_URMA_PROVIDER_CLIENT = 1,
};

enum kvaf_urma_provider_config_flag {
  KVAF_URMA_PROVIDER_USE_CTP = 1U << 0,
  KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI = 1U << 1,
  KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI = 1U << 2,
};

enum kvaf_urma_provider_address_kind {
  // The address is already visible to the selected URMA device. For Host DDR
  // this is the DVA returned by halHostRegister(HOST_MEM_MAP_DEV); for the
  // Device endpoint this is an HBM device VA returned by aclrtMalloc.
  KVAF_URMA_PROVIDER_ADDRESS_DEVICE_VA = 1,
};

enum kvaf_udma_private_abi {
  KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1 = 1,
  /*
   * The device provider runs inside the AICPU KFC process and exports the
   * ordinary UDMA mmap addresses used by UB_MicroBenchmark_new. This enum
   * identifies that tested perftest AIV layout and does not mean
   * CREATE_JETTY_EX(tg_id) was used.
   */
  KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1 = 2,
};

enum kvaf_urma_provider_result_flag {
  KVAF_URMA_PROVIDER_CONNECTED = 1U << 0,
  KVAF_URMA_PROVIDER_TOKEN_ENABLED = 1U << 1,
  // urma_eid_t.raw is network order. The source-matched UDMA provider reverses
  // those 16 bytes into target_jetty.le_eid before writing a normal SQE.
  // A raw AIV SQE consumer must reproduce that conversion exactly once.
  KVAF_URMA_PROVIDER_REMOTE_EID_NETWORK_ORDER = 1U << 2,
  /*
   * The adapter has already applied the UDMA SQ hardware-doorbell offset and
   * the queue addresses belong to the AICPU shared device-VA domain.  AIV
   * reachability is proven by the data-path smoke transfer, not by this bit.
   */
  KVAF_URMA_PROVIDER_QUEUE_IS_AICPU_SHARED_VA = 1U << 3,
  // Resource creation or teardown returned without proof that every DMA
  // object was destroyed. The caller must retain backing memory until reset.
  KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN = 1U << 4,
};

enum kvaf_urma_provider_status {
  KVAF_URMA_PROVIDER_OK = 0,
  KVAF_URMA_PROVIDER_INVALID_ARGUMENT = 1,
  KVAF_URMA_PROVIDER_ALREADY_OPEN = 2,
  KVAF_URMA_PROVIDER_PARSE_FAILED = 3,
  KVAF_URMA_PROVIDER_LOCAL_CONFIG_FAILED = 4,
  KVAF_URMA_PROVIDER_CONNECT_FAILED = 5,
  KVAF_URMA_PROVIDER_REMOTE_CONFIG_FAILED = 6,
  KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED = 7,
  KVAF_URMA_PROVIDER_RESOURCE_INCOMPATIBLE = 8,
  KVAF_URMA_PROVIDER_NOT_OPEN = 9,
};

// Stable configuration copied either directly to the Host provider or
// through the AICPU KFC argument block. local_address is always the address
// visible to the selected URMA device: the halHostRegister-produced DVA for
// Host DDR and an HBM device VA for the AICPU endpoint.
// segment_payload_bytes is the perftest half-buffer size; the adapter
// registers two halves and never posts the legacy warm-up WQE.
struct kvaf_urma_provider_config_v2 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t role;
  uint32_t flags;
  uint32_t eid_index;
  uint32_t jetty_index;
  uint32_t control_port;
  uint32_t address_kind;
  // Deadline applied to every control-socket send/receive after connect.
  uint32_t control_timeout_seconds;
  // Non-zero caller nonce. A close operation may destroy only the provider
  // context created by the same nonce.
  uint64_t session_id;
  uint64_t local_address;
  uint64_t local_allocation_bytes;
  uint64_t segment_payload_bytes;
  char device_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char local_eid[KVAF_URMA_PROVIDER_EID_TEXT_BYTES];
  char peer_eid[KVAF_URMA_PROVIDER_EID_TEXT_BYTES];
  uint64_t reserved[3];
};

// Provider-private pointers never cross this ABI. Only device-visible queue,
// doorbell, memory and token metadata are returned.
struct kvaf_urma_provider_result_v2 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  int32_t provider_status;
  uint32_t flags;
  uint32_t sq_depth;
  uint32_t cq_depth;
  uint64_t sq_base;
  uint64_t sq_doorbell;
  uint64_t cq_base;
  uint64_t cq_doorbell;
  uint64_t local_base;
  uint64_t local_bytes;
  uint64_t remote_base;
  uint64_t remote_bytes;
  uint32_t wqe_size;
  uint32_t cqe_size;
  uint32_t tp_id;
  // 20-bit UDMA object id written to SQE word3. This is not the public
  // URMA token_id; current A5 UDMA exposes that value as tid << 8.
  uint32_t remote_object_id;
  uint32_t remote_token_value;
  // 20-bit UDMA object id of the local registered segment. It is returned
  // for resource diagnostics; the validated compact READ WQE leaves its
  // local SGE token word zero and carries the local VA directly.
  uint32_t local_object_id;
  uint32_t local_token_value;
  // log2(number of 64-byte WQEBBs in the raw SQ ring). This is different
  // from log2(wqe_size): one normal liburma READ may consume two WQEBBs,
  // while the validated AIV compact READ consumes exactly one.
  uint32_t sq_ring_shift;
  uint8_t remote_eid[16];
  uint32_t sq_head;
  uint32_t cq_tail;
  uint32_t wqe_count;
  // enum kvaf_udma_private_abi selected by the isolated queue accessor.
  uint32_t private_abi;
  uint64_t session_id;
  // reserved[0] carries the compile-time private-layout signature.
  uint64_t reserved[11];
};

// rtAicpuKernelLaunchExWithArgs locates the first two string fields by
// offset. result_address is a device address writable by the AICPU kernel.
struct kvaf_urma_aicpu_open_args_v2 {
  char kernel_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char so_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  struct kvaf_urma_provider_config_v2 config;
  uint64_t result_address;
  uint64_t reserved[7];
};

struct kvaf_urma_aicpu_close_args_v2 {
  char kernel_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char so_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  uint32_t final_sq_head;
  uint32_t final_cq_tail;
  // Device address of an int32_t written by the AICPU entry point. This is
  // required because stream synchronization reports task execution, not the
  // C function's return code.
  uint64_t result_address;
  uint64_t session_id;
  uint64_t reserved[5];
};

KVAF_URMA_PROVIDER_EXPORT int kvaf_urma_provider_open_v2(
    const struct kvaf_urma_provider_config_v2* config,
    struct kvaf_urma_provider_result_v2* result);
KVAF_URMA_PROVIDER_EXPORT int kvaf_urma_provider_close_v2(
    uint64_t session_id);
KVAF_URMA_PROVIDER_EXPORT uint32_t
kvaf_urma_provider_private_abi_v2(void);
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_open_v2(
    void* opaque_args);
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_close_v2(
    void* opaque_args);

#ifdef __cplusplus
}

static_assert(sizeof(kvaf_urma_provider_config_v2) == 256,
              "direct URMA provider config ABI must be 256 bytes");
static_assert(sizeof(kvaf_urma_provider_result_v2) == 256,
              "direct URMA provider result ABI must be 256 bytes");
static_assert(sizeof(kvaf_urma_aicpu_open_args_v2) == 384,
              "direct URMA AICPU open ABI must be 384 bytes");
static_assert(sizeof(kvaf_urma_aicpu_close_args_v2) == 128,
              "direct URMA AICPU close ABI must be 128 bytes");
static_assert(offsetof(kvaf_urma_provider_config_v2, session_id) == 40,
              "unexpected direct URMA config session offset");
static_assert(offsetof(kvaf_urma_provider_config_v2, local_address) == 48,
              "unexpected direct URMA config address offset");
static_assert(offsetof(kvaf_urma_provider_result_v2, session_id) == 160,
              "unexpected direct URMA result session offset");
static_assert(offsetof(kvaf_urma_aicpu_close_args_v2, result_address) == 72,
              "unexpected direct URMA close result offset");
static_assert(offsetof(kvaf_urma_aicpu_close_args_v2, session_id) == 80,
              "unexpected direct URMA close session offset");
#else
_Static_assert(sizeof(struct kvaf_urma_provider_config_v2) == 256,
               "direct URMA provider config ABI must be 256 bytes");
_Static_assert(sizeof(struct kvaf_urma_provider_result_v2) == 256,
               "direct URMA provider result ABI must be 256 bytes");
_Static_assert(sizeof(struct kvaf_urma_aicpu_open_args_v2) == 384,
               "direct URMA AICPU open ABI must be 384 bytes");
_Static_assert(sizeof(struct kvaf_urma_aicpu_close_args_v2) == 128,
               "direct URMA AICPU close ABI must be 128 bytes");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2, session_id) == 40,
               "unexpected direct URMA config session offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2, local_address) == 48,
               "unexpected direct URMA config address offset");
_Static_assert(offsetof(struct kvaf_urma_provider_result_v2, session_id) == 160,
               "unexpected direct URMA result session offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_close_args_v2,
                        result_address) == 72,
               "unexpected direct URMA close result offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_close_args_v2, session_id) == 80,
               "unexpected direct URMA close session offset");
#endif

#endif

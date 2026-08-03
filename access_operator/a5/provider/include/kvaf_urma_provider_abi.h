#ifndef KVAF_URMA_PROVIDER_ABI_H
#define KVAF_URMA_PROVIDER_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define KVAF_URMA_PROVIDER_EXPORT \
  __attribute__((visibility("default")))
#define KVAF_URMA_PROVIDER_CACHE_ALIGNED \
  __attribute__((aligned(64)))
#else
#define KVAF_URMA_PROVIDER_EXPORT
#define KVAF_URMA_PROVIDER_CACHE_ALIGNED
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KVAF_URMA_PROVIDER_MAGIC 0x4b555250U
#define KVAF_URMA_PROVIDER_ABI_VERSION 2U
#define KVAF_URMA_PROVIDER_NAME_BYTES 32U
#define KVAF_URMA_PROVIDER_CONTROL_ADDRESS_BYTES 64U

/*
 * Resource-free AICPU owner probe ABI.  This protocol proves only that one
 * bounded AICPU invocation can remain alive while another execution stream
 * publishes a release command.  It neither creates URMA/UMMU resources nor
 * establishes a process-affinity contract for separate KFC launches.
 *
 * Command and status deliberately occupy different cache lines.  The command
 * line has one external producer and the status line has one AICPU producer;
 * consumers synchronize through opcode/state with release/acquire atomics.
 */
#define KVAF_URMA_OWNER_PROBE_MAGIC 0x4b4f5052U /* "KOPR" */
#define KVAF_URMA_OWNER_PROBE_ABI_VERSION 1U
#define KVAF_URMA_OWNER_PROBE_CACHE_LINE_BYTES 64U
#define KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE UINT64_C(1)
#define KVAF_URMA_OWNER_PROBE_MAX_TIMEOUT_MILLISECONDS 120000U
#define KVAF_URMA_OWNER_PROBE_MAX_POLL_MICROSECONDS 10000U

enum kvaf_urma_owner_probe_command {
  KVAF_URMA_OWNER_PROBE_COMMAND_NONE = 0,
  KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE = 1,
};

enum kvaf_urma_owner_probe_state {
  KVAF_URMA_OWNER_PROBE_STATE_EMPTY = 0,
  KVAF_URMA_OWNER_PROBE_STATE_READY = 1,
  KVAF_URMA_OWNER_PROBE_STATE_RELEASED = 2,
  KVAF_URMA_OWNER_PROBE_STATE_TIMED_OUT = 3,
  KVAF_URMA_OWNER_PROBE_STATE_FAILED = 4,
};

enum kvaf_urma_owner_probe_result {
  KVAF_URMA_OWNER_PROBE_RESULT_OK = 0,
  KVAF_URMA_OWNER_PROBE_RESULT_TIMED_OUT = 1,
  KVAF_URMA_OWNER_PROBE_RESULT_CLOCK_FAILED = 2,
  KVAF_URMA_OWNER_PROBE_RESULT_INVALID_COMMAND = 3,
  KVAF_URMA_OWNER_PROBE_RESULT_CAPABILITY_LEDGER_UNAVAILABLE = 4,
};

/* External producer-owned cache line.  Publish opcode last with release. */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_owner_probe_command_v1 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint64_t session_id;
  uint64_t sequence;
  uint64_t reserved[4];
};

/* AICPU owner-owned cache line.  Publish state last with release. */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_owner_probe_status_v1 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t state;
  uint64_t session_id;
  uint64_t sequence;
  int32_t result_code;
  uint32_t reserved0;
  uint64_t elapsed_nanoseconds;
  uint64_t reserved[2];
};

/*
 * KFC locates the two names in the first 64 bytes.  timeout_milliseconds is
 * mandatory and capped above; poll_interval_microseconds may be zero to select
 * the provider's conservative one-millisecond default.
 */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_aicpu_owner_probe_args_v1 {
  char kernel_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char so_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t timeout_milliseconds;
  uint64_t session_id;
  uint64_t command_address;
  uint64_t status_address;
  uint32_t poll_interval_microseconds;
  uint32_t reserved0;
  uint64_t reserved[2];
};

/*
 * Device-runtime URMA compatibility probe.
 *
 * This is deliberately separate from the source-linked direct provider.  A
 * single AICPU invocation dlopens the Device OS liburma.so.0.  Discover only
 * resolves symbols and closes the handle, but dlopen may execute official
 * constructors and is therefore supervised/quarantined like every AICPU
 * mode.  Resource modes create objects through symbols resolved from that
 * exact handle and tear them down in reverse order.  The probe never unlocks
 * the direct H2D runner by itself.
 */
#define KVAF_URMA_RUNTIME_PROBE_MAGIC 0x4b525450U /* "KRTP" */
#define KVAF_URMA_RUNTIME_PROBE_ABI_VERSION 2U
#define KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN \
  UINT64_C(0x4b52545000020001)
#define KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES 32U
#define KVAF_URMA_RUNTIME_PROBE_CACHE_LINE_BYTES 64U
#define KVAF_URMA_RUNTIME_PROBE_DEFAULT_QUEUE_DEPTH 64U
#define KVAF_URMA_RUNTIME_PROBE_MAX_QUEUE_DEPTH 4096U

enum kvaf_urma_runtime_probe_mode {
  /* Loader-only: no urma_init/resources; official constructors may execute. */
  KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER = 0,
  /* Resource-bearing one-invocation context/JFC/JFR create-delete A/B. */
  KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE = 1,
  /* Append-only: own init/uninit and prove provider device discovery. */
  KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE = 2,
};

enum kvaf_urma_runtime_probe_outcome {
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK = 0,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_INVALID_ARGUMENT = 1,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_LIBRARY_UNAVAILABLE = 2,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_SYMBOL_UNAVAILABLE = 3,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_INIT_FAILED = 4,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED = 5,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_DEVICE_NOT_FOUND = 6,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONTEXT_CREATE_FAILED_UNPROVEN = 7,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFC_CREATE_FAILED_UNPROVEN = 8,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFR_CREATE_FAILED_UNPROVEN = 9,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE = 10,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONCURRENT_USE = 11,
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED = 12,
  /* Requested package identity differs from the provider's compiled ID. */
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH = 13,
  /* The sealed capability ledger required for every official DSO call is absent. */
  KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE = 14,
};

enum kvaf_urma_runtime_probe_stage {
  KVAF_URMA_RUNTIME_PROBE_STAGE_NONE = 0,
  KVAF_URMA_RUNTIME_PROBE_STAGE_LOAD_LIBRARY = 1,
  KVAF_URMA_RUNTIME_PROBE_STAGE_RESOLVE_SYMBOLS = 2,
  KVAF_URMA_RUNTIME_PROBE_STAGE_INIT = 3,
  KVAF_URMA_RUNTIME_PROBE_STAGE_GET_DEVICE = 4,
  KVAF_URMA_RUNTIME_PROBE_STAGE_CREATE_CONTEXT = 5,
  KVAF_URMA_RUNTIME_PROBE_STAGE_CREATE_JFC = 6,
  KVAF_URMA_RUNTIME_PROBE_STAGE_CREATE_JFR = 7,
  KVAF_URMA_RUNTIME_PROBE_STAGE_DELETE_JFR = 8,
  KVAF_URMA_RUNTIME_PROBE_STAGE_DELETE_JFC = 9,
  KVAF_URMA_RUNTIME_PROBE_STAGE_DELETE_CONTEXT = 10,
  KVAF_URMA_RUNTIME_PROBE_STAGE_UNINIT = 11,
  KVAF_URMA_RUNTIME_PROBE_STAGE_CLOSE_LIBRARY = 12,
  KVAF_URMA_RUNTIME_PROBE_STAGE_COMPLETE = 13,
};

enum kvaf_urma_runtime_probe_cleanup {
  KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFR = 1U << 0,
  KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFC = 1U << 1,
  KVAF_URMA_RUNTIME_PROBE_CLEANUP_CONTEXT = 1U << 2,
  KVAF_URMA_RUNTIME_PROBE_CLEANUP_RUNTIME = 1U << 3,
  KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY = 1U << 4,
};

enum kvaf_urma_runtime_probe_flag {
  /* urma_init returned EEXIST, so this invocation must not call uninit. */
  KVAF_URMA_RUNTIME_PROBE_RUNTIME_BORROWED = 1U << 0,
  /* A public struct compiled into this provider may not match Device OS. */
  KVAF_URMA_RUNTIME_PROBE_PUBLIC_ABI_UNPROVEN = 1U << 1,
  /* create returned no handle; the public API cannot prove no orphan exists. */
  KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN = 1U << 2,
  KVAF_URMA_RUNTIME_PROBE_LIBRARY_PATH_TRUNCATED = 1U << 3,
  KVAF_URMA_RUNTIME_PROBE_RESOURCE_OPERATION_ATTEMPTED = 1U << 4,
  /* A prior invocation retained or could not prove release of an obligation. */
  KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED = 1U << 5,
  KVAF_URMA_RUNTIME_PROBE_LOADER_ERROR_TRUNCATED = 1U << 6,
  /* A symbol resolved from a different DSO than the resolved urma_init. */
  KVAF_URMA_RUNTIME_PROBE_SYMBOL_ORIGIN_MISMATCH = 1U << 7,
  /*
   * dlopen constructors or a failed urma_init may have changed opaque
   * official-runtime state.  Neither a successful public init/uninit cycle
   * nor dlclose proves that all provider, process and kernel side effects were
   * reversed in the undocumented AICPU caller domain.  This probe therefore
   * never clears the bit; only a vendor cleanup contract or a boot boundary
   * may retire the resulting quarantine.
   */
  KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN = 1U << 8,
};

#define KVAF_URMA_RUNTIME_PROBE_ALL_CLEANUP_BITS ((uint32_t)( \
    KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFR | \
    KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFC | \
    KVAF_URMA_RUNTIME_PROBE_CLEANUP_CONTEXT | \
    KVAF_URMA_RUNTIME_PROBE_CLEANUP_RUNTIME | \
    KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY))
#define KVAF_URMA_RUNTIME_PROBE_ALL_FLAG_BITS ((uint32_t)( \
    KVAF_URMA_RUNTIME_PROBE_RUNTIME_BORROWED | \
    KVAF_URMA_RUNTIME_PROBE_PUBLIC_ABI_UNPROVEN | \
    KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN | \
    KVAF_URMA_RUNTIME_PROBE_LIBRARY_PATH_TRUNCATED | \
    KVAF_URMA_RUNTIME_PROBE_RESOURCE_OPERATION_ATTEMPTED | \
    KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED | \
    KVAF_URMA_RUNTIME_PROBE_LOADER_ERROR_TRUNCATED | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ORIGIN_MISMATCH | \
    KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN))

#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_INIT (UINT64_C(1) << 0)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNINIT (UINT64_C(1) << 1)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE (UINT64_C(1) << 2)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_CONTEXT (UINT64_C(1) << 3)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_CONTEXT (UINT64_C(1) << 4)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFC (UINT64_C(1) << 5)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JFC (UINT64_C(1) << 6)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFR (UINT64_C(1) << 7)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JFR (UINT64_C(1) << 8)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFC (UINT64_C(1) << 9)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFC_OPT (UINT64_C(1) << 10)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFC_OPT (UINT64_C(1) << 11)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFC (UINT64_C(1) << 12)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFC (UINT64_C(1) << 13)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFC (UINT64_C(1) << 14)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFS (UINT64_C(1) << 15)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFS_OPT (UINT64_C(1) << 16)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFS_OPT (UINT64_C(1) << 17)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFS (UINT64_C(1) << 18)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFS (UINT64_C(1) << 19)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFS (UINT64_C(1) << 20)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFR (UINT64_C(1) << 21)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFR_OPT (UINT64_C(1) << 22)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFR_OPT (UINT64_C(1) << 23)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFR (UINT64_C(1) << 24)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFR (UINT64_C(1) << 25)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFR (UINT64_C(1) << 26)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JETTY (UINT64_C(1) << 27)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JETTY_OPT (UINT64_C(1) << 28)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JETTY_OPT (UINT64_C(1) << 29)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JETTY (UINT64_C(1) << 30)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JETTY (UINT64_C(1) << 31)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JETTY (UINT64_C(1) << 32)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_REGISTER_SEG (UINT64_C(1) << 33)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNREGISTER_SEG (UINT64_C(1) << 34)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_SEG (UINT64_C(1) << 35)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_SEG (UINT64_C(1) << 36)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JFS_WR (UINT64_C(1) << 37)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JFR_WR (UINT64_C(1) << 38)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_POLL_JFC (UINT64_C(1) << 39)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JETTY (UINT64_C(1) << 40)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JETTY (UINT64_C(1) << 41)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_JETTY (UINT64_C(1) << 42)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_JETTY (UINT64_C(1) << 43)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_BIND_JETTY (UINT64_C(1) << 44)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNBIND_JETTY (UINT64_C(1) << 45)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_DEVICE (UINT64_C(1) << 46)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_STR_TO_EID (UINT64_C(1) << 47)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE_BY_EID (UINT64_C(1) << 48)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_EID_LIST (UINT64_C(1) << 49)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_EID_LIST (UINT64_C(1) << 50)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_TOKEN_ID (UINT64_C(1) << 51)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_TOKEN_ID (UINT64_C(1) << 52)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_TP_LIST (UINT64_C(1) << 53)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_JETTY_EX (UINT64_C(1) << 54)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JETTY_SEND_WR \
    (UINT64_C(1) << 55)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JETTY_RECV_WR \
    (UINT64_C(1) << 56)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_MODIFY_JETTY (UINT64_C(1) << 57)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_JETTY (UINT64_C(1) << 58)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_MODIFY_JFR (UINT64_C(1) << 59)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_JFR (UINT64_C(1) << 60)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_ASYNC_EVENT (UINT64_C(1) << 61)
#define KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACK_ASYNC_EVENT (UINT64_C(1) << 62)

#define KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_INIT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNINIT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE))
#define KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_CONTEXT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_CONTEXT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JFR))
#define KVAF_URMA_RUNTIME_PROBE_CANDIDATE_MODERN_QUEUE_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFC_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFC_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFS_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFS_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFR_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFR_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JETTY_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JETTY_OPT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JETTY))
#define KVAF_URMA_RUNTIME_PROBE_MODERN_QUEUE_SYMBOLS \
    KVAF_URMA_RUNTIME_PROBE_CANDIDATE_MODERN_QUEUE_SYMBOLS
#define KVAF_URMA_RUNTIME_PROBE_SEGMENT_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_REGISTER_SEG | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNREGISTER_SEG | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_SEG | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_SEG))
#define KVAF_URMA_RUNTIME_PROBE_CANDIDATE_PUBLIC_POSTING_SYMBOLS \
    ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JFS_WR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JFR_WR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_POLL_JFC))
#define KVAF_URMA_RUNTIME_PROBE_DATA_PLANE_SYMBOLS \
    KVAF_URMA_RUNTIME_PROBE_CANDIDATE_PUBLIC_POSTING_SYMBOLS
#define KVAF_URMA_RUNTIME_PROBE_LEGACY_JETTY_LINK_WRAPPER_SYMBOLS \
    ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_BIND_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNBIND_JETTY))
#define KVAF_URMA_RUNTIME_PROBE_JETTY_LINK_SYMBOLS \
    KVAF_URMA_RUNTIME_PROBE_LEGACY_JETTY_LINK_WRAPPER_SYMBOLS
/* Candidate modern queue wrappers; no KVAF end-to-end semantics implied. */
#define KVAF_URMA_RUNTIME_PROBE_CANDIDATE_AIV_CONTROL_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_CONTEXT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_CONTEXT | \
    KVAF_URMA_RUNTIME_PROBE_CANDIDATE_MODERN_QUEUE_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_SEGMENT_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_CANDIDATE_PUBLIC_POSTING_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_LEGACY_JETTY_LINK_WRAPPER_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_DEVICE))

/*
 * Exact public-symbol surface called by the checked direct-URMA UB management
 * and teardown implementation compiled by this revision.  Completeness still
 * proves neither function semantics nor AICPU support; official implementations
 * can export stubs, reject modes, or contain lifecycle defects.
 */
#define KVAF_URMA_RUNTIME_PROBE_CURRENT_ENDPOINT_DISCOVERY_SYMBOLS \
    ((uint64_t)(KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_DEVICE | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_STR_TO_EID | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE_BY_EID | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_EID_LIST | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_EID_LIST))
#define KVAF_URMA_RUNTIME_PROBE_CURRENT_RESOURCE_LIFECYCLE_SYMBOLS \
    ((uint64_t)(KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_SEGMENT_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_TOKEN_ID | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_TOKEN_ID))
#define KVAF_URMA_RUNTIME_PROBE_CURRENT_CTP_LINK_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_TP_LIST | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_JETTY_EX))
#define KVAF_URMA_RUNTIME_PROBE_CURRENT_UB_MANAGEMENT_IO_SYMBOLS \
    ((uint64_t)(KVAF_URMA_RUNTIME_PROBE_SYMBOL_POLL_JFC | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JETTY_SEND_WR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JETTY_RECV_WR))
#define KVAF_URMA_RUNTIME_PROBE_CURRENT_CHECKED_TEARDOWN_SYMBOLS \
    ((uint64_t)(KVAF_URMA_RUNTIME_PROBE_SYMBOL_MODIFY_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_JETTY | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_MODIFY_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_JFR | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_ASYNC_EVENT | \
    KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACK_ASYNC_EVENT))
#define KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS \
    ((uint64_t)(KVAF_URMA_RUNTIME_PROBE_CURRENT_ENDPOINT_DISCOVERY_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_CURRENT_RESOURCE_LIFECYCLE_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_CURRENT_CTP_LINK_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_CURRENT_UB_MANAGEMENT_IO_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_CURRENT_CHECKED_TEARDOWN_SYMBOLS))
#define KVAF_URMA_RUNTIME_PROBE_ALL_SYMBOLS ((uint64_t)( \
    KVAF_URMA_RUNTIME_PROBE_CANDIDATE_AIV_CONTROL_SYMBOLS | \
    KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS))

struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_runtime_probe_result_v2 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t outcome;
  uint32_t primary_outcome;
  uint32_t stage;
  int32_t operation_status;
  int32_t operation_errno;
  uint32_t flags;
  uint32_t cleanup_required;
  uint32_t cleanup_completed;
  uint32_t jfc_config_size;
  uint32_t jfr_config_size;
  uint32_t queue_depth;
  /* Required for this mode versus actually exported by this exact handle. */
  uint64_t required_symbol_mask;
  uint64_t resolved_symbol_mask;
  /* errno is diagnostic only and is valid only when this field is one. */
  uint32_t operation_errno_valid;
  /* Header version used to compile the probe; not a runtime ABI handshake. */
  uint32_t compiled_urma_api_version;
  char runtime_library_path[128];
  /* Sanitized dlerror text; diagnostic only, never parsed as an ABI signal. */
  char loader_error[64];
  /* Host-generated request binding; all fields must echo exactly. */
  uint64_t request_nonce[2];
  uint64_t provider_abi_token;
  uint32_t mode;
  uint32_t eid_index;
  int32_t logical_device_id;
  uint32_t args_size;
  /* Provider-compiled source/config build identity, not a self-file hash. */
  uint8_t provider_build_id_sha256[
      KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES];
  uint8_t reserved[40];
};

/* KFC requires kernel_name and so_name at stable offsets in the first 64B. */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_aicpu_runtime_probe_args_v2 {
  char kernel_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char so_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char device_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t eid_index;
  uint64_t result_address;
  uint32_t queue_depth;
  uint32_t mode;
  uint64_t request_nonce[2];
  uint64_t expected_provider_abi_token;
  int32_t logical_device_id;
  uint32_t reserved0;
  /* Must match the build ID compiled into the selected provider DSO. */
  uint8_t expected_provider_build_id_sha256[
      KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES];
};

/*
 * Resource-owning AICPU epoch ABI.  Unlike the separate bounded
 * owner-concurrency probe, both this epoch owner and the runtime-URMA probe
 * above may enter library code with undocumented constructor/runtime side
 * effects; neither is classified as resource-free.
 * One invocation calls provider Open, publishes READY, remains alive while
 * AIV submits any number of batches to the exported SQ/CQ, and performs the
 * checked provider Close itself after a cooperative shutdown record proves
 * that the queue is drained.  Process-local provider pointers therefore
 * never cross a KFC launch boundary.
 *
 * A lease is diagnostic, not permission to destroy live transport state.
 * Expiry without a valid SHUTDOWN_QUIESCED record leaves the invocation alive
 * in UNSAFE_WAITING.  A producer may repair and republish the command; only a
 * command whose epoch, session, sequence and four queue counters match can
 * start Close.
 */
#define KVAF_URMA_EPOCH_OWNER_MAGIC 0x4b45504fU /* "KEPO" */
#define KVAF_URMA_EPOCH_OWNER_ABI_VERSION 1U
#define KVAF_URMA_EPOCH_OWNER_CACHE_LINE_BYTES 64U
#define KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE UINT64_C(1)
#define KVAF_URMA_EPOCH_OWNER_MAX_LEASE_MILLISECONDS 3600000U
#define KVAF_URMA_EPOCH_OWNER_MAX_POLL_MICROSECONDS 10000U
#define KVAF_URMA_EPOCH_OWNER_MAX_CLOSE_RETRY_LIMIT 1024U
#define KVAF_URMA_EPOCH_OWNER_MAX_CLOSE_RETRY_MICROSECONDS 1000000U

enum kvaf_urma_epoch_owner_command {
  KVAF_URMA_EPOCH_OWNER_COMMAND_NONE = 0,
  KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED = 1,
};

enum kvaf_urma_epoch_owner_state {
  KVAF_URMA_EPOCH_OWNER_STATE_EMPTY = 0,
  KVAF_URMA_EPOCH_OWNER_STATE_OPENING = 1,
  KVAF_URMA_EPOCH_OWNER_STATE_READY = 2,
  KVAF_URMA_EPOCH_OWNER_STATE_CLOSING = 3,
  KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING = 4,
  KVAF_URMA_EPOCH_OWNER_STATE_CLOSED = 5,
  KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE = 6,
  KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING = 7,
};

enum kvaf_urma_epoch_owner_result {
  KVAF_URMA_EPOCH_OWNER_RESULT_OK = 0,
  KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED = 1,
  KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED = 2,
  KVAF_URMA_EPOCH_OWNER_RESULT_INVALID_COMMAND = 3,
  KVAF_URMA_EPOCH_OWNER_RESULT_QUEUE_NOT_QUIESCED = 4,
  KVAF_URMA_EPOCH_OWNER_RESULT_LEASE_EXPIRED = 5,
  KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_RETRY_LIMIT = 6,
  KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_UNSAFE = 7,
  /* Resource creation is compiled out until a sealed ledger is consumed. */
  KVAF_URMA_EPOCH_OWNER_RESULT_CAPABILITY_LEDGER_UNAVAILABLE = 8,
};

enum kvaf_urma_epoch_owner_status_flag {
  KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID = 1U << 0,
  KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE = 1U << 1,
  KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE = 1U << 2,
  KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE = 1U << 3,
};

/* AIV/Host producer-owned cache line. Publish opcode last with release. */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_epoch_owner_command_v1 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint64_t epoch;
  uint64_t session_id;
  uint64_t sequence;
  uint32_t final_sq_head;
  uint32_t final_sq_tail;
  uint32_t final_cq_tail;
  uint32_t final_wqe_count;
  uint64_t reserved[1];
};

/* AICPU owner-owned cache line. Publish state last with release. */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_epoch_owner_status_v1 {
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t state;
  uint64_t epoch;
  uint64_t session_id;
  uint64_t sequence;
  int32_t result_code;
  int32_t provider_status;
  uint32_t close_attempts;
  uint32_t flags;
  uint64_t elapsed_nanoseconds;
};

enum kvaf_urma_provider_role {
  KVAF_URMA_PROVIDER_SERVER = 0,
  KVAF_URMA_PROVIDER_CLIENT = 1,
};

enum kvaf_urma_provider_config_flag {
  KVAF_URMA_PROVIDER_USE_CTP = 1U << 0,
  KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI = 1U << 1,
  KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI = 1U << 2,
  /*
   * The two fixed text slots carry TCP management addresses, never URMA
   * EIDs. Requiring this bit makes an old binary/new provider (or the
   * reverse) fail closed even though the fixed 256-byte wire layout is
   * intentionally preserved.
   */
  KVAF_URMA_PROVIDER_EXPLICIT_TCP_MANAGEMENT = 1U << 3,
  /*
   * Bootstrap and all perftest metadata exchange use a dedicated URMA
   * RM+CTP management Jetty.  The 128-bit EID strings in the two fixed text
   * slots are URMA identifiers; they are never passed to an IPv4/IPv6 socket.
   * Exactly one management flag must be set, which makes a new caller paired
   * with an old provider (or the reverse) fail closed.
   */
  KVAF_URMA_PROVIDER_URMA_MANAGEMENT = 1U << 4,
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
  // The provider retained the exact session and failed public handle. Calling
  // Close again for the same session resumes checked teardown; backing memory
  // must remain owned between attempts, but hardware reset is not yet implied.
  KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE = 1U << 5,
};

/*
 * Runtime evidence carried in result.reserved[1..3].  A build manifest is
 * necessary for artifact provenance, but it cannot prove that the loaded
 * Host/AICPU instance actually selected the source-owned UDMA provider.
 */
#define KVAF_UDMA_SOURCE_CAPABILITY_MAGIC \
  UINT64_C(0x4b56414655444d41) /* "KVAFUDMA" */
#define KVAF_UDMA_TEARDOWN_SOURCE_OWNED_PARTIAL_RETRY_V1 \
  UINT64_C(0x0000000000010001)
#define KVAF_UDMA_TEARDOWN_SOURCE_OWNED_CHECKED_RETRYABLE_V1 \
  UINT64_C(0x0000000000020001)

enum kvaf_urma_provider_close_status {
  KVAF_URMA_PROVIDER_CLOSE_OK = 0,
  // The same provider instance still owns a precise retry cursor and handle.
  KVAF_URMA_PROVIDER_CLOSE_RETRYABLE = 1,
  // Ownership/protocol state is ambiguous. Retrying cannot prove quiescence.
  KVAF_URMA_PROVIDER_CLOSE_UNSAFE = 2,
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
  /* No caller-controlled field can authorize resources while this is set. */
  KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE = 10,
};

/*
 * Stable, failure-only diagnostics for the URMA management bootstrap.  The
 * provider copies this record into result.reserved[4..8] before checked
 * teardown begins, so a Device/AICPU Open failure remains observable by the
 * Host even when AICPU stderr is not collected.  This extends the meaning of
 * previously-reserved words without changing the 256-byte result ABI.
 */
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_MAGIC \
  UINT64_C(0x4b5641464d474d54) /* "KVAFMGMT" */
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_VERSION 1U
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_MAGIC 4U
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_STAGE 5U
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_ERRNO 6U
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_STATUS 7U
#define KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_FLAGS 8U

/*
 * CREATE_JFR-only command snapshot. reserved[9] preserves the complete opaque
 * 64-bit JFC handle. reserved[10] packs jfc_id plus version, context page
 * shift, validation flags and the bounded F_GETFD errno. The outer management
 * diagnostic magic/stage is the publication marker. This snapshot is
 * observation-only: no allocator, pinning, queue or command bit is changed.
 */
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_VERSION 1U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_RESERVED_HANDLE 9U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_RESERVED_LAYOUT 10U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_VERSION_SHIFT 0U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_VERSION_MASK 0x3U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_PAGE_SHIFT_SHIFT 2U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_PAGE_SHIFT_MASK 0x1fU
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_FLAGS_SHIFT 7U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_FLAGS_MASK 0xfffffU
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_FD_ERRNO_SHIFT 27U
#define KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_FD_ERRNO_MASK 0x1fU

enum kvaf_urma_jfr_command_diagnostic_flag {
  KVAF_URMA_JFR_DIAGNOSTIC_RQ_ADDRESS_VALID = 1U << 0,
  KVAF_URMA_JFR_DIAGNOSTIC_RQ_HW_PAGE_ALIGNED = 1U << 1,
  KVAF_URMA_JFR_DIAGNOSTIC_RQ_RANGE_VALID = 1U << 2,
  KVAF_URMA_JFR_DIAGNOSTIC_INDEX_ADDRESS_VALID = 1U << 3,
  KVAF_URMA_JFR_DIAGNOSTIC_INDEX_HW_PAGE_ALIGNED = 1U << 4,
  KVAF_URMA_JFR_DIAGNOSTIC_INDEX_RANGE_VALID = 1U << 5,
  KVAF_URMA_JFR_DIAGNOSTIC_RECORD_DB_VALID = 1U << 6,
  KVAF_URMA_JFR_DIAGNOSTIC_SLEEP_DB_VALID = 1U << 7,
  KVAF_URMA_JFR_DIAGNOSTIC_JFC_HANDLE_VALID = 1U << 8,
  KVAF_URMA_JFR_DIAGNOSTIC_INPUT_SIZE_MATCH = 1U << 9,
  KVAF_URMA_JFR_DIAGNOSTIC_OUTPUT_SIZE_MATCH = 1U << 10,
  KVAF_URMA_JFR_DIAGNOSTIC_INPUT_ADDRESS_MATCH = 1U << 11,
  KVAF_URMA_JFR_DIAGNOSTIC_OUTPUT_ADDRESS_MATCH = 1U << 12,
  KVAF_URMA_JFR_DIAGNOSTIC_CONTEXT_FD_VALID = 1U << 13,
  KVAF_URMA_JFR_DIAGNOSTIC_JFC_CONTEXT_MATCH = 1U << 14,
  KVAF_URMA_JFR_DIAGNOSTIC_JFC_FD_MATCH = 1U << 15,
  KVAF_URMA_JFR_DIAGNOSTIC_JFC_REGISTRY_IDENTITY_MATCH = 1U << 16,
  KVAF_URMA_JFR_DIAGNOSTIC_NON_PIN = 1U << 17,
  KVAF_URMA_JFR_DIAGNOSTIC_DTU = 1U << 18,
  KVAF_URMA_JFR_DIAGNOSTIC_HUGEPAGE = 1U << 19,
};

enum kvaf_urma_management_diagnostic_flag {
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RAW_ERRNO_VALID = 1U << 0,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_EFFECTIVE_ERRNO_VALID = 1U << 1,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_OPERATION_STATUS_VALID = 1U << 2,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_AUXILIARY_STATUS_VALID = 1U << 3,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_CONTEXT_SVA_SEPARATE = 1U << 4,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_USER_ANONYMOUS_MEMORY = 1U << 5,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_AUX_PROVIDER_STAGE_ID = 1U << 6,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_AUX_ROLLBACK_STATUS = 1U << 7,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_AUX_TP_COUNT = 1U << 8,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_AUX_UMMU_LAST_ERROR = 1U << 9,
  KVAF_URMA_MANAGEMENT_DIAGNOSTIC_AUX_COMPLETION_STATUS = 1U << 10,
};

/* Values are append-only: logs and result artifacts may outlive a build. */
enum kvaf_urma_management_diagnostic_stage {
  KVAF_URMA_MANAGEMENT_STAGE_NONE = 0,
  KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_ESTABLISH_ARGUMENTS = 1,
  KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_ARGUMENTS = 2,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_OWNER = 3,
  KVAF_URMA_MANAGEMENT_STAGE_RESOLVE_DEVICE_EID = 4,
  KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_DEVICE_EID = 5,
  KVAF_URMA_MANAGEMENT_STAGE_CREATE_CONTEXT = 6,
  KVAF_URMA_MANAGEMENT_STAGE_CREATE_JFC = 7,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_SEND_BUFFER = 8,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_RECEIVE_BUFFER = 9,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_HANDSHAKE_BUFFER = 10,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_SEND_TOKEN = 11,
  KVAF_URMA_MANAGEMENT_STAGE_REGISTER_SEND_SEGMENT = 12,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_RECEIVE_TOKEN = 13,
  KVAF_URMA_MANAGEMENT_STAGE_REGISTER_RECEIVE_SEGMENT = 14,
  KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_HANDSHAKE_TOKEN = 15,
  KVAF_URMA_MANAGEMENT_STAGE_REGISTER_HANDSHAKE_SEGMENT = 16,
  KVAF_URMA_MANAGEMENT_STAGE_CREATE_JFR = 17,
  KVAF_URMA_MANAGEMENT_STAGE_CREATE_JETTY = 18,
  KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_SERVER_JETTY_ID = 19,
  KVAF_URMA_MANAGEMENT_STAGE_SERVER_HANDSHAKE = 20,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_PARSE_PEER_EID = 21,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_GET_TP_LIST = 22,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_VALIDATE_TP_LIST = 23,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_IMPORT_PEER_JETTY_EX = 24,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_POST_HANDSHAKE_SEND = 25,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_POLL_HANDSHAKE_CQE = 26,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_HANDSHAKE = 27,
  KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_PEER_EID = 28,
  KVAF_URMA_MANAGEMENT_STAGE_POST_CLIENT_RECEIVE = 29,
  KVAF_URMA_MANAGEMENT_STAGE_SERVER_VALIDATE_PEER_CTP = 30,
  KVAF_URMA_MANAGEMENT_STAGE_SERVER_GET_TP_LIST = 31,
  KVAF_URMA_MANAGEMENT_STAGE_SERVER_VALIDATE_TP_LIST = 32,
  KVAF_URMA_MANAGEMENT_STAGE_SERVER_IMPORT_PEER_JETTY_EX = 33,
  KVAF_URMA_MANAGEMENT_STAGE_CLIENT_VALIDATE_PEER_CTP = 34,
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
  // TCP listener port. Ignored by the URMA management backend.
  uint32_t control_port;
  uint32_t address_kind;
  // Shared monotonic deadline for management bootstrap and every exchange.
  uint32_t control_timeout_seconds;
  // Non-zero caller nonce. A close operation may destroy only the provider
  // context created by the same nonce.
  uint64_t session_id;
  uint64_t local_address;
  uint64_t local_allocation_bytes;
  uint64_t segment_payload_bytes;
  char device_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  union {
    // Numeric IPv4/IPv6 client connect target in TCP management mode.
    char control_server_address[
        KVAF_URMA_PROVIDER_CONTROL_ADDRESS_BYTES];
    // Peer/server URMA EID in UB management mode. The server carries its
    // expected peer EID for validation even though bootstrap learns the
    // client's Jetty ID from the receive CQE.
    char management_peer_address[
        KVAF_URMA_PROVIDER_CONTROL_ADDRESS_BYTES];
  };
  union {
    // Optional numeric client bind address in TCP management mode.
    char control_bind_address[
        KVAF_URMA_PROVIDER_CONTROL_ADDRESS_BYTES];
    // Local URMA EID in UB management mode.
    char management_local_address[
        KVAF_URMA_PROVIDER_CONTROL_ADDRESS_BYTES];
  };
  // Fixed, non-zero server management Jetty ID. It is deliberately separate
  // from the TCP port and from the data Jetty created by perftest.
  uint32_t management_jetty_id;
  uint32_t reserved0;
  uint64_t reserved[2];
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
  // reserved[1] is KVAF_UDMA_SOURCE_CAPABILITY_MAGIC.
  // reserved[2] is the exact source-owned teardown capability.
  // reserved[3] is the non-zero source/public URMA layout signature checked
  // inside the provider before any URMA device is selected.
  // reserved[4..8] carry a failure-only management diagnostic when
  // reserved[4] equals KVAF_URMA_MANAGEMENT_DIAGNOSTIC_MAGIC. reserved[5]
  // packs version:stage, reserved[6] packs effective_errno:raw_errno,
  // reserved[7] packs auxiliary_status:operation_status, and reserved[8]
  // carries enum kvaf_urma_management_diagnostic_flag.  The auxiliary-kind
  // bits are mandatory whenever auxiliary_status is valid; callers must not
  // infer that this word is a hardware completion code. reserved[9..10]
  // carry the failure-only CREATE_JFR command snapshot described above.
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

/*
 * KFC locates the two names in the first 64 bytes.  A zero lease selects
 * cooperative-shutdown-only mode. close_retry_limit is an escalation
 * threshold, not a permission to return: after that many checked RETRYABLE
 * results the invocation reports UNSAFE_WAITING but continues retrying the
 * same process-local teardown cursor.
 */
struct KVAF_URMA_PROVIDER_CACHE_ALIGNED kvaf_urma_aicpu_epoch_owner_args_v1 {
  char kernel_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  char so_name[KVAF_URMA_PROVIDER_NAME_BYTES];
  struct kvaf_urma_provider_config_v2 config;
  uint64_t epoch;
  uint64_t command_address;
  uint64_t status_address;
  uint64_t open_result_address;
  uint32_t lease_timeout_milliseconds;
  uint32_t poll_interval_microseconds;
  uint32_t close_retry_limit;
  uint32_t close_retry_interval_microseconds;
  uint32_t magic;
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t flags;
};

KVAF_URMA_PROVIDER_EXPORT int kvaf_urma_provider_open_v2(
    const struct kvaf_urma_provider_config_v2* config,
    struct kvaf_urma_provider_result_v2* result);
KVAF_URMA_PROVIDER_EXPORT int kvaf_urma_provider_close_v2(
    uint64_t session_id);
KVAF_URMA_PROVIDER_EXPORT uint32_t
kvaf_urma_provider_private_abi_v2(void);
// The AICPU entry points return the KFC task status. Once opaque_args and its
// result address are valid they return zero even when the provider operation
// fails; provider status is reported only through the device result block.
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_open_v2(
    void* opaque_args);
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_close_v2(
    void* opaque_args);
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_owner_probe_v1(
    void* opaque_args);
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_rt_probe_v2(
    void* opaque_args);
KVAF_URMA_PROVIDER_EXPORT int aicpu_kvaf_urma_epoch_owner_v1(
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
static_assert(sizeof(kvaf_urma_owner_probe_command_v1) == 64,
              "owner probe command ABI must be 64 bytes");
static_assert(alignof(kvaf_urma_owner_probe_command_v1) == 64,
              "owner probe command must own one cache line");
static_assert(sizeof(kvaf_urma_owner_probe_status_v1) == 64,
              "owner probe status ABI must be 64 bytes");
static_assert(alignof(kvaf_urma_owner_probe_status_v1) == 64,
              "owner probe status must own one cache line");
static_assert(sizeof(kvaf_urma_aicpu_owner_probe_args_v1) == 128,
              "owner probe AICPU args ABI must be 128 bytes");
static_assert(alignof(kvaf_urma_aicpu_owner_probe_args_v1) == 64,
              "owner probe AICPU args must be cache-line aligned");
static_assert(offsetof(kvaf_urma_owner_probe_command_v1, opcode) == 12,
              "unexpected owner probe command opcode offset");
static_assert(offsetof(kvaf_urma_owner_probe_status_v1, state) == 12,
              "unexpected owner probe status state offset");
static_assert(offsetof(kvaf_urma_aicpu_owner_probe_args_v1,
                       command_address) == 88,
              "unexpected owner probe command address offset");
static_assert(offsetof(kvaf_urma_aicpu_owner_probe_args_v1,
                       status_address) == 96,
              "unexpected owner probe status address offset");
static_assert(sizeof(kvaf_urma_runtime_probe_result_v2) == 384,
              "runtime URMA probe result ABI must be 384 bytes");
static_assert(alignof(kvaf_urma_runtime_probe_result_v2) == 64,
              "runtime URMA probe result must be cache-line aligned");
static_assert(sizeof(kvaf_urma_aicpu_runtime_probe_args_v2) == 192,
              "runtime URMA probe AICPU args ABI must be 192 bytes");
static_assert(alignof(kvaf_urma_aicpu_runtime_probe_args_v2) == 64,
              "runtime URMA probe args must be cache-line aligned");
static_assert(offsetof(kvaf_urma_aicpu_runtime_probe_args_v2,
                       result_address) == 112,
              "unexpected runtime URMA probe result offset");
static_assert(offsetof(kvaf_urma_aicpu_runtime_probe_args_v2,
                       request_nonce) == 128,
              "unexpected runtime URMA probe nonce offset");
static_assert(offsetof(kvaf_urma_runtime_probe_result_v2,
                       required_symbol_mask) == 56,
              "unexpected runtime URMA required-symbol offset");
static_assert(offsetof(kvaf_urma_runtime_probe_result_v2,
                       runtime_library_path) == 80,
              "unexpected runtime URMA library-path offset");
static_assert(offsetof(kvaf_urma_runtime_probe_result_v2,
                       request_nonce) == 272,
              "unexpected runtime URMA result nonce offset");
static_assert(offsetof(kvaf_urma_runtime_probe_result_v2,
                       provider_build_id_sha256) == 312,
              "unexpected runtime URMA provider build-ID offset");
static_assert(offsetof(kvaf_urma_aicpu_runtime_probe_args_v2,
                       expected_provider_build_id_sha256) == 160,
              "unexpected runtime URMA expected build-ID offset");
static_assert(KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS ==
                  UINT64_C(0x7fffcb9e000001ff),
              "current direct-UB public symbol manifest drifted");
static_assert((KVAF_URMA_RUNTIME_PROBE_ALL_SYMBOLS &
               (UINT64_C(1) << 63)) == 0U,
              "runtime URMA symbol bit 63 must remain reserved");
static_assert(sizeof(kvaf_urma_epoch_owner_command_v1) == 64,
              "epoch owner command ABI must be 64 bytes");
static_assert(alignof(kvaf_urma_epoch_owner_command_v1) == 64,
              "epoch owner command must own one cache line");
static_assert(sizeof(kvaf_urma_epoch_owner_status_v1) == 64,
              "epoch owner status ABI must be 64 bytes");
static_assert(alignof(kvaf_urma_epoch_owner_status_v1) == 64,
              "epoch owner status must own one cache line");
static_assert(sizeof(kvaf_urma_aicpu_epoch_owner_args_v1) == 384,
              "epoch owner AICPU args ABI must be 384 bytes");
static_assert(alignof(kvaf_urma_aicpu_epoch_owner_args_v1) == 64,
              "epoch owner AICPU args must be cache-line aligned");
static_assert(offsetof(kvaf_urma_epoch_owner_command_v1, opcode) == 12,
              "unexpected epoch owner command opcode offset");
static_assert(offsetof(kvaf_urma_epoch_owner_command_v1,
                       final_sq_head) == 40,
              "unexpected epoch owner final SQ head offset");
static_assert(offsetof(kvaf_urma_epoch_owner_status_v1, state) == 12,
              "unexpected epoch owner status state offset");
static_assert(offsetof(kvaf_urma_aicpu_epoch_owner_args_v1, epoch) == 320,
              "unexpected epoch owner epoch offset");
static_assert(offsetof(kvaf_urma_aicpu_epoch_owner_args_v1,
                       command_address) == 328,
              "unexpected epoch owner command address offset");
static_assert(offsetof(kvaf_urma_aicpu_epoch_owner_args_v1,
                       open_result_address) == 344,
              "unexpected epoch owner open result address offset");
static_assert(offsetof(kvaf_urma_aicpu_epoch_owner_args_v1, magic) == 368,
              "unexpected epoch owner args magic offset");
static_assert(offsetof(kvaf_urma_provider_config_v2, session_id) == 40,
              "unexpected direct URMA config session offset");
static_assert(offsetof(kvaf_urma_provider_config_v2, local_address) == 48,
              "unexpected direct URMA config address offset");
static_assert(offsetof(kvaf_urma_provider_config_v2,
                       control_server_address) == 104,
              "unexpected direct URMA control server address offset");
static_assert(offsetof(kvaf_urma_provider_config_v2,
                       control_bind_address) == 168,
              "unexpected direct URMA control bind address offset");
static_assert(offsetof(kvaf_urma_provider_config_v2,
                       management_peer_address) == 104,
              "unexpected direct URMA management peer EID offset");
static_assert(offsetof(kvaf_urma_provider_config_v2,
                       management_local_address) == 168,
              "unexpected direct URMA management local EID offset");
static_assert(offsetof(kvaf_urma_provider_config_v2,
                       management_jetty_id) == 232,
              "unexpected direct URMA management Jetty offset");
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
_Static_assert(sizeof(struct kvaf_urma_owner_probe_command_v1) == 64,
               "owner probe command ABI must be 64 bytes");
_Static_assert(_Alignof(struct kvaf_urma_owner_probe_command_v1) == 64,
               "owner probe command must own one cache line");
_Static_assert(sizeof(struct kvaf_urma_owner_probe_status_v1) == 64,
               "owner probe status ABI must be 64 bytes");
_Static_assert(_Alignof(struct kvaf_urma_owner_probe_status_v1) == 64,
               "owner probe status must own one cache line");
_Static_assert(sizeof(struct kvaf_urma_aicpu_owner_probe_args_v1) == 128,
               "owner probe AICPU args ABI must be 128 bytes");
_Static_assert(_Alignof(struct kvaf_urma_aicpu_owner_probe_args_v1) == 64,
               "owner probe AICPU args must be cache-line aligned");
_Static_assert(offsetof(struct kvaf_urma_owner_probe_command_v1, opcode) == 12,
               "unexpected owner probe command opcode offset");
_Static_assert(offsetof(struct kvaf_urma_owner_probe_status_v1, state) == 12,
               "unexpected owner probe status state offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_owner_probe_args_v1,
                        command_address) == 88,
               "unexpected owner probe command address offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_owner_probe_args_v1,
                        status_address) == 96,
               "unexpected owner probe status address offset");
_Static_assert(sizeof(struct kvaf_urma_runtime_probe_result_v2) == 384,
               "runtime URMA probe result ABI must be 384 bytes");
_Static_assert(_Alignof(struct kvaf_urma_runtime_probe_result_v2) == 64,
               "runtime URMA probe result must be cache-line aligned");
_Static_assert(sizeof(struct kvaf_urma_aicpu_runtime_probe_args_v2) == 192,
               "runtime URMA probe AICPU args ABI must be 192 bytes");
_Static_assert(_Alignof(struct kvaf_urma_aicpu_runtime_probe_args_v2) == 64,
               "runtime URMA probe args must be cache-line aligned");
_Static_assert(offsetof(struct kvaf_urma_aicpu_runtime_probe_args_v2,
                        result_address) == 112,
               "unexpected runtime URMA probe result offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_runtime_probe_args_v2,
                        request_nonce) == 128,
               "unexpected runtime URMA probe nonce offset");
_Static_assert(offsetof(struct kvaf_urma_runtime_probe_result_v2,
                        required_symbol_mask) == 56,
               "unexpected runtime URMA required-symbol offset");
_Static_assert(offsetof(struct kvaf_urma_runtime_probe_result_v2,
                        runtime_library_path) == 80,
               "unexpected runtime URMA library-path offset");
_Static_assert(offsetof(struct kvaf_urma_runtime_probe_result_v2,
                        request_nonce) == 272,
               "unexpected runtime URMA result nonce offset");
_Static_assert(offsetof(struct kvaf_urma_runtime_probe_result_v2,
                        provider_build_id_sha256) == 312,
               "unexpected runtime URMA provider build-ID offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_runtime_probe_args_v2,
                        expected_provider_build_id_sha256) == 160,
               "unexpected runtime URMA expected build-ID offset");
_Static_assert(KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS ==
                   UINT64_C(0x7fffcb9e000001ff),
               "current direct-UB public symbol manifest drifted");
_Static_assert((KVAF_URMA_RUNTIME_PROBE_ALL_SYMBOLS &
                (UINT64_C(1) << 63)) == 0U,
               "runtime URMA symbol bit 63 must remain reserved");
_Static_assert(sizeof(struct kvaf_urma_epoch_owner_command_v1) == 64,
               "epoch owner command ABI must be 64 bytes");
_Static_assert(_Alignof(struct kvaf_urma_epoch_owner_command_v1) == 64,
               "epoch owner command must own one cache line");
_Static_assert(sizeof(struct kvaf_urma_epoch_owner_status_v1) == 64,
               "epoch owner status ABI must be 64 bytes");
_Static_assert(_Alignof(struct kvaf_urma_epoch_owner_status_v1) == 64,
               "epoch owner status must own one cache line");
_Static_assert(sizeof(struct kvaf_urma_aicpu_epoch_owner_args_v1) == 384,
               "epoch owner AICPU args ABI must be 384 bytes");
_Static_assert(_Alignof(struct kvaf_urma_aicpu_epoch_owner_args_v1) == 64,
               "epoch owner AICPU args must be cache-line aligned");
_Static_assert(offsetof(struct kvaf_urma_epoch_owner_command_v1, opcode) == 12,
               "unexpected epoch owner command opcode offset");
_Static_assert(offsetof(struct kvaf_urma_epoch_owner_command_v1,
                        final_sq_head) == 40,
               "unexpected epoch owner final SQ head offset");
_Static_assert(offsetof(struct kvaf_urma_epoch_owner_status_v1, state) == 12,
               "unexpected epoch owner status state offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_epoch_owner_args_v1,
                        epoch) == 320,
               "unexpected epoch owner epoch offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_epoch_owner_args_v1,
                        command_address) == 328,
               "unexpected epoch owner command address offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_epoch_owner_args_v1,
                        open_result_address) == 344,
               "unexpected epoch owner open result address offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_epoch_owner_args_v1,
                        magic) == 368,
               "unexpected epoch owner args magic offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2, session_id) == 40,
               "unexpected direct URMA config session offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2, local_address) == 48,
               "unexpected direct URMA config address offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2,
                        control_server_address) == 104,
               "unexpected direct URMA control server address offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2,
                        control_bind_address) == 168,
               "unexpected direct URMA control bind address offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2,
                        management_peer_address) == 104,
               "unexpected direct URMA management peer EID offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2,
                        management_local_address) == 168,
               "unexpected direct URMA management local EID offset");
_Static_assert(offsetof(struct kvaf_urma_provider_config_v2,
                        management_jetty_id) == 232,
               "unexpected direct URMA management Jetty offset");
_Static_assert(offsetof(struct kvaf_urma_provider_result_v2, session_id) == 160,
               "unexpected direct URMA result session offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_close_args_v2,
                        result_address) == 72,
               "unexpected direct URMA close result offset");
_Static_assert(offsetof(struct kvaf_urma_aicpu_close_args_v2, session_id) == 80,
               "unexpected direct URMA close session offset");
#endif

#endif

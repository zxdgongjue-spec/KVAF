#ifndef KVAF_AIV_BATCH_GET_ABI_H_
#define KVAF_AIV_BATCH_GET_ABI_H_

#include <stdint.h>

// Shared host/AIV ABI for the graph-safe ExecuteBatchGet kernel.  These
// records are copied verbatim between host code, HBM and the AIV kernel.
enum {
  KVAF_AIV_BATCH_GET_STATUS_BYTES = 384,
  KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES = 64,
  KVAF_AIV_BATCH_GET_MAX_AIV_CORES = 48,
  KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE = 2,
  KVAF_AIV_BATCH_GET_WQE_DATA_BYTES = 656,
  KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT = 128,
  KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_BYTES =
      KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT *
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES,
  // Shared local-memory contract used by the AIV kernel and launcher.
  KVAF_AIV_BATCH_GET_LOCAL_MEMORY_BYTES = 160 * 1024,
};

static const uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC = 0x4b5641464247444fULL;
static const uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC =
    0x4b56414650524f46ULL;  // "KVAFPROF"
static const uint64_t KVAF_AIV_BATCH_GET_PROFILE_RING_MAGIC =
    0x4b5641465052494eULL;  // "KVAFPRIN"

// Optional, per-core kernel timeline. The host enables collection by placing
// KVAF_AIV_BATCH_GET_PROFILE_MAGIC in `magic` before launch. The kernel keeps
// the magic in its output, so repeated launches can reuse the same status
// buffer. GetSystemCycle frequency is platform-specific and the caller must
// supply the clock MHz when converting cycles to time.
typedef struct KvafBatchGetCoreProfile {
  uint64_t magic;
  uint64_t kernel_start_cycle;
  uint64_t setup_done_cycle;
  uint64_t compact_done_cycle;
  uint64_t first_doorbell_cycle;
  uint64_t post_done_cycle;
  uint64_t completion_done_cycle;
  uint64_t kernel_finish_cycle;
  uint64_t active_requests;
  uint64_t total_posts;
  uint64_t jetty_completion_wait_cycles[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
  uint64_t jetty_completion_polls[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
  uint64_t reserved[26];
} KvafBatchGetCoreProfile;

// Keep the historical 384-byte stride even though ExecuteBatchGet uses only
// the first cache line on the normal path. This keeps the verified host/device
// allocation ABI stable while using the reserved area for opt-in profiling.
typedef struct KvafBatchGetCoreStatus {
  uint64_t magic;
  uint64_t processed_request_count;
  uint64_t result;
  uint64_t completed_batch_count;
  // With multiple Jettys these two diagnostic indices report Jetty 0.
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t failed_request_index;
  uint64_t last_wqe_protocol;
  KvafBatchGetCoreProfile profile;
} KvafBatchGetCoreStatus;

// Per-core, device-resident profile history. The AIV core owning one entry
// writes this structure, so no device atomic is needed. `sample_count` tracks
// every profiled kernel call; `record_count` tracks calls that passed the
// host-configured duration threshold and were written into the ring.
typedef struct KvafBatchGetProfileRingHeader {
  uint64_t magic;
  uint64_t sample_count;
  uint64_t record_count;
  uint64_t capacity;
} KvafBatchGetProfileRingHeader;

// Keep the complete hardware timeline plus the final queue/status values for
// one AIV core invocation. `sequence` is local to that core and monotonically
// increases even when duration filtering skips a record.
typedef struct KvafBatchGetProfileRecord {
  uint64_t sequence;
  uint64_t result;
  uint64_t processed_request_count;
  uint64_t completed_batch_count;
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t failed_request_index;
  uint64_t last_wqe_protocol;
  KvafBatchGetCoreProfile profile;
} KvafBatchGetProfileRecord;

// Persistent, AIV-owned software indices for one USER_CTL SQ/CQ pair. Queue
// arrays are core-major: [block_count][jettys_per_core].
typedef struct KvafBatchGetQueueState {
  uint32_t magic;
  uint32_t initialized;
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t completed_batch_count;
  uint64_t last_error;
  uint64_t reserved[3];
} KvafBatchGetQueueState;

// Device view of one Direct-HCCP transport plus one acquired local HBM
// buffer.  The Host-DDR/Jetty fields are transport-scoped; each successful
// LocalBufferAcquire creates its own device-resident entity table with the
// local HBM address/token fields below filled for that caller-owned buffer.
typedef struct KvafBatchGetDirectUrmaEntity {
  uint32_t magic;
  uint32_t version;
  uint64_t remote_host_ddr_address;
  uint64_t remote_host_ddr_size;
  uint32_t remote_token_id;
  uint32_t remote_token_value;
  uint64_t local_hbm_address;
  uint64_t local_hbm_size;
  uint32_t local_token_id;
  uint32_t local_token_value;
  uint64_t sq_va;
  uint64_t sq_db_va;
  uint32_t jetty_id;
  uint32_t sq_depth;
  uint32_t tp_id;
  uint32_t wqe_size;
  uint8_t remote_eid[16];
  uint64_t cq_va;
  uint64_t cq_db_va;
  uint32_t jfc_id;
  uint32_t cq_depth;
  uint32_t cqe_size;
  uint32_t reserved0;
  uint64_t reserved[7];
} KvafBatchGetDirectUrmaEntity;

// Fixed-shape graph ABI. `output` and `channel_entity` must identify the same
// LocalBufferAcquire result. During capture/replay both addresses stay fixed;
// address/valid/index/offset/mask/output-index contents may change between
// replays.
typedef struct KvafExecuteBatchGetKernelArgs {
  uint64_t address_table;       // uint64[table_capacity], absolute Host DDR UBA
  uint64_t valid_table;         // uint8[table_capacity]
  uint64_t indices;             // int32[request_capacity]
  uint64_t offsets;             // int32[request_capacity], byte offset in entry
  uint64_t mask;                // uint8[request_capacity]
  uint64_t output_indices;      // int32[request_capacity]
  uint64_t output;              // caller-owned HBM output; must match entity
  uint64_t core_status;         // KvafBatchGetCoreStatus[block_count]
  uint64_t channel_entity;      // Entity[block_count * jettys_per_core]
  uint64_t queue_state;         // QueueState[block_count * jettys_per_core]
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t output_rows;
  uint32_t block_bytes;
  uint32_t block_count;
  // Zero preserves the phase-one ABI and selects one Jetty per AIV core.
  uint32_t jettys_per_core;
  // Optional fixed device ring: one Header + capacity Records per core.
  // These trailing fields preserve the pre-profile argument layout.
  uint64_t profile_ring;
  uint64_t profile_min_cycles;
  uint32_t profile_ring_capacity;
  uint32_t profile_ring_stride_bytes;
} KvafExecuteBatchGetKernelArgs;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchGetCoreStatus) == KVAF_AIV_BATCH_GET_STATUS_BYTES,
              "KvafBatchGetCoreStatus ABI changed");
static_assert(sizeof(KvafBatchGetCoreProfile) == 320,
              "KvafBatchGetCoreProfile ABI changed");
static_assert(sizeof(KvafBatchGetProfileRingHeader) == 32,
              "KvafBatchGetProfileRingHeader ABI changed");
static_assert(sizeof(KvafBatchGetProfileRecord) == 384,
              "KvafBatchGetProfileRecord ABI changed");
static_assert(sizeof(KvafBatchGetQueueState) == KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES,
              "KvafBatchGetQueueState ABI changed");
static_assert(sizeof(KvafBatchGetDirectUrmaEntity) == 192,
              "KvafBatchGetDirectUrmaEntity ABI changed");
static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 128,
              "KvafExecuteBatchGetKernelArgs ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_GET_ABI_H_

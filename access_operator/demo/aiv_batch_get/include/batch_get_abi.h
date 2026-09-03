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
  // One cold-bound, graph-visible control for the MTP forward-resident
  // Direct-HCCP route.  This deliberately keeps the deployed MTP BatchGet
  // wire ABI below (miss_count + token_bytes + tokens_per_block); it is not
  // compatible with the later mask/block_bytes experimental ABI.
  KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_BYTES =
      KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // Shared local-memory contract used by the AIV kernel and launcher.
  KVAF_AIV_BATCH_GET_LOCAL_MEMORY_BYTES = 160 * 1024,
};

static const uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC = 0x4b5641464247444fULL;
static const uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC =
    0x4b56414650524f46ULL;  // "KVAFPROF"
static const uint64_t KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC =
    0x4b5641464d545052ULL;  // "KVAFMTPR"
static const uint64_t KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION = 1ULL;

// Optional, per-core kernel timeline. The host enables collection by placing
// KVAF_AIV_BATCH_GET_PROFILE_MAGIC in `magic` before launch. The kernel keeps
// the magic in its output, so repeated launches can reuse the same status
// buffer. GetSystemCycle frequency is platform-specific and the caller must
// supply the clock MHz when converting cycles to time.
typedef struct KvafBatchGetCoreProfile {
  uint64_t magic;
  uint64_t kernel_start_cycle;
  uint64_t setup_done_cycle;
  // Legacy profile slot retained for ABI stability. The direct-prefix kernel
  // performs no compaction, so this equals setup_done_cycle.
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
// address/valid/index/offset/miss-count/output-index contents may change between
// replays.
typedef struct KvafExecuteBatchGetKernelArgs {
  uint64_t address_table;       // uint64[table_capacity], absolute Host DDR UBA
  uint64_t valid_table;         // uint8[table_capacity]
  uint64_t indices;             // int32[request_capacity]
  uint64_t offsets;             // int32[request_capacity], byte offset in entry
  uint64_t miss_count;          // int32[1], valid request-prefix length
  uint64_t output_indices;      // int32[request_capacity]
  uint64_t output;              // caller-owned HBM output; must match entity
  uint64_t core_status;         // KvafBatchGetCoreStatus[block_count]
  uint64_t channel_entity;      // Entity[block_count * jettys_per_core]
  uint64_t queue_state;         // QueueState[block_count * jettys_per_core]
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t output_rows;
  uint32_t token_bytes;         // bytes transferred for one token/WQE
  uint32_t tokens_per_block;    // tokens stored in one address-table block
  uint32_t block_count;
  // Zero preserves the phase-one ABI and selects one Jetty per AIV core.
  uint32_t jettys_per_core;
} KvafExecuteBatchGetKernelArgs;

// One graph-stable descriptor template for one Direct-HCCP queue in the MTP
// forward-resident route.  Every vector has the already deployed 112-byte
// ExecuteBatchGet ABI: `miss_count` is an NPU int32[1] written by the device
// executor, while token_bytes and tokens_per_block stay part of the cold
// control.  Do not replace miss_count with a mask here: MTP graph validation
// and the AIV kernel both consume the compact prefix count.
typedef struct KvafLayerResidentBatchGetDescriptorScratch {
  uint64_t address_table;
  uint64_t valid_table;
  uint64_t indices;
  uint64_t offsets;
  uint64_t miss_count;
  uint64_t output_indices;
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t token_bytes;
  uint32_t tokens_per_block;
} KvafLayerResidentBatchGetDescriptorScratch;

// Cold-bound Direct-HCCP state for one forward-owned MTP resident payload.
// Rows in [resident_row_base, resident_row_base + resident_row_count) are
// fixed by the caller before the forward starts; the device executor may only
// populate those rows.  The queue lease table protects only SQ/CQ ownership.
// It never owns, clears, remaps, or recycles resident rows.
typedef struct KvafLayerResidentBatchGetControl {
  uint64_t magic;
  uint64_t version;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint64_t core_status;
  uint64_t output;
  uint64_t queue_leases;
  uint64_t descriptor_templates;
  uint64_t queue_count;
  uint64_t output_rows;
  uint64_t resident_row_base;
  uint64_t resident_row_count;
  uint64_t jettys_per_core;
  uint64_t token_bytes;
  uint64_t tokens_per_block;
  uint64_t descriptor_template_count;
} KvafLayerResidentBatchGetControl;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchGetCoreStatus) == KVAF_AIV_BATCH_GET_STATUS_BYTES,
              "KvafBatchGetCoreStatus ABI changed");
static_assert(sizeof(KvafBatchGetCoreProfile) == 320,
              "KvafBatchGetCoreProfile ABI changed");
static_assert(sizeof(KvafBatchGetQueueState) == KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES,
              "KvafBatchGetQueueState ABI changed");
static_assert(sizeof(KvafBatchGetDirectUrmaEntity) == 192,
              "KvafBatchGetDirectUrmaEntity ABI changed");
static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 112,
              "KvafExecuteBatchGetKernelArgs ABI changed");
static_assert(sizeof(KvafLayerResidentBatchGetDescriptorScratch) == 64,
              "KvafLayerResidentBatchGetDescriptorScratch ABI changed");
static_assert(sizeof(KvafLayerResidentBatchGetControl) ==
                  KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_BYTES,
              "KvafLayerResidentBatchGetControl ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_GET_ABI_H_

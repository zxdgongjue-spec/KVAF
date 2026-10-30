#ifndef KVAF_AIV_BATCH_GET_ABI_H_
#define KVAF_AIV_BATCH_GET_ABI_H_

#include <stdint.h>

// Shared host/AIV ABI for the graph-safe ExecuteBatchGet kernel.  These
// records are copied verbatim between host code, HBM and the AIV kernel.
enum {
  KVAF_AIV_BATCH_GET_STATUS_BYTES = 5376,
  KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES = 64,
  KVAF_AIV_BATCH_GET_MAX_AIV_CORES = 48,
  KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE = 64,
  // Shared local-memory contract used by the AIV kernel and launcher.
  KVAF_AIV_BATCH_GET_LOCAL_MEMORY_BYTES = 160 * 1024,
};

static const uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC = 0x4b5641464247444fULL;
static const uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC =
    0x4b56414650524633ULL;  // "KVAFPRF3": serial CQ profiling
static const uint64_t KVAF_AIV_BATCH_GET_JETTY_TIMELINE_MAGIC =
    0x4b56414650525433ULL;  // "KVAFPRT3": round-robin CQ observation

// Optional V3 extension in the former reserved profile word. Zero denotes an
// older V3 kernel. Resource-table strides and all ABI record sizes stay fixed.
enum {
  KVAF_BATCH_GET_SCHEDULE_METADATA_V1 = 1U << 16U,
  KVAF_BATCH_GET_SCHEDULE_METADATA_V2 = 2U << 16U,
  KVAF_BATCH_GET_SCHEDULE_VERSION_SHIFT = 16U,
  KVAF_BATCH_GET_SCHEDULE_VERSION_MASK = 0xfU,
  KVAF_BATCH_GET_SCHEDULE_LEAN = 1U,
  KVAF_BATCH_GET_SCHEDULE_SINGLE_GROUP = 2U,
  KVAF_BATCH_GET_SCHEDULE_TAIL_REUSED = 4U,
  KVAF_BATCH_GET_SCHEDULE_GROUP_ENABLED = 8U,
  KVAF_BATCH_GET_SCHEDULE_POST_ENABLED = 16U,
  KVAF_BATCH_GET_SCHEDULE_MANAGED_SHIFT = 8U,
  KVAF_BATCH_GET_SCHEDULE_MANAGED_MASK = 0x7fU,
  KVAF_BATCH_GET_SCHEDULE_MIN_TAIL_SHIFT = 24U,
  KVAF_BATCH_GET_SCHEDULE_MIN_TAIL_MASK = 0xffU,
  KVAF_BATCH_GET_SCHEDULE_POST_COUNT_SHIFT = 32U,
};

// Actual SQ publications and software CQ observations, reset on every launch.
// CQ timestamps are observations after posting, not hardware completion times.
typedef struct KvafBatchGetJettyProfile {
  uint64_t submitted_reads;
  uint64_t submitted_posts;
  uint64_t completed_checkpoints;
  uint64_t first_doorbell_cycle;
  uint64_t last_doorbell_cycle;
  uint64_t first_cqe_seen_cycle;
  uint64_t last_cqe_seen_cycle;
  uint64_t probe_active_cycles;  // populated only in round-robin mode
} KvafBatchGetJettyProfile;

// Optional, per-core kernel timeline. The host enables collection by placing
// KVAF_AIV_BATCH_GET_PROFILE_MAGIC (or JETTY_TIMELINE_MAGIC) in `magic` before
// launch. The kernel keeps
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
  uint64_t submitted_group_count;
  uint64_t completed_group_count;
  uint64_t completion_count;  // CQEs consumed, not completed READs
  uint64_t post_active_cycles;  // sum of group submissions, excluding CQ waits
  uint64_t drain_wait_cycles;   // sum of all group completion waits
  uint64_t peak_submitted_not_drained;  // conservative per-core software peak
  // Retired small-post slots are always zero in current kernels. Keep these
  // words for V3 layout compatibility and historical log decoding.
  uint64_t small_post_max_requests;
  uint64_t selected_post_width;
  uint64_t small_post_group_count;
  uint64_t fused_group_count;
  uint64_t wqe_build_calls;
  uint64_t jetty_timeline_enabled;
  uint64_t profile_version;
  uint64_t schedule_metadata;
  KvafBatchGetJettyProfile jettys[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
} KvafBatchGetCoreProfile;

// ExecuteBatchGet uses only the first cache line on the normal path. The
// remaining cache lines carry the opt-in per-Jetty profiling payload.
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
  uint64_t poisoned;  // incomplete/failed submission; requires transport rebuild
  uint64_t reserved[2];
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
  // Total READ budget across this launch's cores. Zero selects the original
  // whole-batch path; otherwise must be >= 2 * block_count. Captured by value.
  // Consumes former padding. Host and V2 kernel MUST be rebuilt together;
  // the V3 kernel symbol also prevents loading the old status/profile stride.
  uint32_t batch_get_max_inflight_total;
} KvafExecuteBatchGetKernelArgs;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchGetCoreStatus) == KVAF_AIV_BATCH_GET_STATUS_BYTES,
              "KvafBatchGetCoreStatus ABI changed");
static_assert(sizeof(KvafBatchGetCoreProfile) == 5312,
              "KvafBatchGetCoreProfile ABI changed");
static_assert(sizeof(KvafBatchGetJettyProfile) == 64,
              "KvafBatchGetJettyProfile ABI changed");
static_assert(sizeof(KvafBatchGetQueueState) == KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES,
              "KvafBatchGetQueueState ABI changed");
static_assert(sizeof(KvafBatchGetDirectUrmaEntity) == 192,
              "KvafBatchGetDirectUrmaEntity ABI changed");
static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 112,
              "KvafExecuteBatchGetKernelArgs ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_GET_ABI_H_
